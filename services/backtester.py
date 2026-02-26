import pandas as pd
import pandas_ta as ta
import numpy as np

class VectorizedBacktester:
    def __init__(self, df: pd.DataFrame, initial_capital: float = 100000.0, risk_pct: float = 0.01):
        """
        Expects a pandas DataFrame with DatetimeIndex and basic OHLCV columns.
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.capital = initial_capital

    def _prepare_indicators(self, params: dict):
        """Compute the strategy indicators using dictionary parameters."""
        ema_fast = params.get('ema_fast', 9)
        ema_slow = params.get('ema_slow', 21)
        rsi_len = params.get('rsi_len', 14)

        self.df.ta.ema(length=ema_fast, append=True)
        self.df.ta.ema(length=ema_slow, append=True)
        self.df.ta.rsi(length=rsi_len, append=True)
        
        # Calculate daily VWAP
        if 'Date' not in self.df.columns:
            self.df['Date'] = self.df.index.date
        
        self.df['Typical_Price'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['VP'] = self.df['Typical_Price'] * self.df['Volume']
        self.df['Cumulative_VP'] = self.df.groupby('Date')['VP'].cumsum()
        self.df['Cumulative_Vol'] = self.df.groupby('Date')['Volume'].cumsum()
        self.df['VWAP'] = self.df['Cumulative_VP'] / self.df['Cumulative_Vol']
        
        self.df = self.df.dropna(subset=[f'EMA_{ema_fast}', f'EMA_{ema_slow}', f'RSI_{rsi_len}', 'VWAP'])

    def run_strategy(self, params: dict):
        """
        Execute vectorized simulation of a Mean Reversion + Trend strategy.
        Returns a dictionary of performance metrics and the trade log.
        """
        self._prepare_indicators(params)
        df = self.df
        
        ema_fast_col = f"EMA_{params.get('ema_fast', 9)}"
        ema_slow_col = f"EMA_{params.get('ema_slow', 21)}"
        rsi_col = f"RSI_{params.get('rsi_len', 14)}"
        
        # Entry Logic (Vectorized mask)
        # Buy: Fast EMA > Slow EMA (Trend), RSI < 40 (Oversold pull-back), Price > VWAP (Intraday Bullish)
        buy_condition = (df[ema_fast_col] > df[ema_slow_col]) & \
                        (df[rsi_col] < params.get('rsi_buy_threshold', 40)) & \
                        (df['Close'] > df['VWAP'])
                        
        # Sell Short: Fast EMA < Slow EMA (Trend), RSI > 60 (Overbought), Price < VWAP (Intraday Bearish)
        short_condition = (df[ema_fast_col] < df[ema_slow_col]) & \
                          (df[rsi_col] > params.get('rsi_short_threshold', 60)) & \
                          (df['Close'] < df['VWAP'])

        df['Signal'] = 0
        df.loc[buy_condition, 'Signal'] = 1
        df.loc[short_condition, 'Signal'] = -1
        
        # Simulate sequential execution of positions (Can't easily vectorize path-dependent SL/TP)
        # We will loop through valid signals to manage isolated trades
        trades = []
        open_trade = None
        
        sl_pct = params.get('sl_pct', 0.01)
        tp_pct = params.get('tp_pct', 0.02)
        
        for index, row in df.iterrows():
            if open_trade is None:
                if row['Signal'] == 1:
                    open_trade = {
                        'entry_time': index, 'type': 'BUY', 'entry_price': row['Close'],
                        'sl': row['Close'] * (1 - sl_pct), 'tp': row['Close'] * (1 + tp_pct)
                    }
                elif row['Signal'] == -1:
                    open_trade = {
                        'entry_time': index, 'type': 'SHORT', 'entry_price': row['Close'],
                        'sl': row['Close'] * (1 + sl_pct), 'tp': row['Close'] * (1 - tp_pct)
                    }
            else:
                # Check exit conditions
                if open_trade['type'] == 'BUY':
                    if row['Low'] <= open_trade['sl']:
                        open_trade['exit_time'] = index
                        open_trade['exit_price'] = open_trade['sl']
                        open_trade['pnl_pct'] = -sl_pct
                        trades.append(open_trade)
                        open_trade = None
                    elif row['High'] >= open_trade['tp']:
                        open_trade['exit_time'] = index
                        open_trade['exit_price'] = open_trade['tp']
                        open_trade['pnl_pct'] = tp_pct
                        trades.append(open_trade)
                        open_trade = None
                elif open_trade['type'] == 'SHORT':
                    if row['High'] >= open_trade['sl']:
                        open_trade['exit_time'] = index
                        open_trade['exit_price'] = open_trade['sl']
                        open_trade['pnl_pct'] = -sl_pct
                        trades.append(open_trade)
                        open_trade = None
                    elif row['Low'] <= open_trade['tp']:
                        open_trade['exit_time'] = index
                        open_trade['exit_price'] = open_trade['tp']
                        open_trade['pnl_pct'] = tp_pct
                        trades.append(open_trade)
                        open_trade = None

        return self._evaluate_metrics(trades)

    def _evaluate_metrics(self, trades: list):
        if not trades:
            return {
                "total_trades": 0, 
                "win_rate": 0, 
                "net_profit": 0, 
                "max_drawdown_pct": 0, 
                "final_equity": self.initial_capital,
                "trade_log": [],
                "equity_curve": [self.initial_capital]
            }
            
        wins = len([t for t in trades if t['pnl_pct'] > 0])
        total_trades = len(trades)
        win_rate = (wins / total_trades) * 100
        
        # Equity curve math
        equity = [self.initial_capital]
        for t in trades:
            risk_amount = equity[-1] * self.risk_pct
            # If trade PNL Pct is +0.02, and SL was 0.01, Reward is 2x risk amount.
            # Simplified sizing: Assume full risk is utilized if SL hits.
            amount_won = risk_amount if t['pnl_pct'] > 0 else -risk_amount
            # Simple compounded return
            new_eq = equity[-1] * (1 + t['pnl_pct'])
            equity.append(new_eq)
            t['pnl_abs'] = new_eq - equity[-2]
            
        final_equity = equity[-1]
        net_profit = final_equity - self.initial_capital
        
        # Max Drawdown
        peak = self.initial_capital
        mdd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > mdd:
                mdd = dd
                
        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "net_profit": round(net_profit, 2),
            "max_drawdown_pct": round(mdd * 100, 2),
            "final_equity": round(final_equity, 2),
            "trade_log": trades,
            "equity_curve": equity
        }
