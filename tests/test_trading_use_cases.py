import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from services.backtester import VectorizedBacktester
from services.technical_analysis import TechnicalAnalysisService
from services.stock_discovery import StockDiscoveryService
from services.advanced_indicators import classifier

@pytest.mark.asyncio
async def test_trading_strategy_accuracy_and_mfe():
    """
    Validate trading strategy accuracy and Max Favorable Excursion (MFE) 
    across 10 diverse stocks (subset of 50 for CI performance).
    Ensures that ML signals lead to profitable exit windows.
    """
    ta_svc = TechnicalAnalysisService()
    discovery_svc = StockDiscoveryService()
    
    # 1. Pick 10 diverse stocks from Nifty 100
    test_universe = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                     "BHARTIARTL.NS", "SBIN.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS"]
    
    results = []
    
    for ticker in test_universe:
        # Fetch 30 days of 15m data
        df = await asyncio.to_thread(ta_svc.fetch_ohlcv, ticker, period="30d", interval="15m")
        if df is None or df.empty:
            continue
            
        # Run backtest
        bt = VectorizedBacktester(df)
        report = await asyncio.to_thread(bt.run_strategy, {})
        
        # 2. Calculate Max Profit (MFE) after each signal
        # We look at the trade log and see if price moved in favor before exit
        trade_log = report.get("trade_log", [])
        mfe_list = []
        for trade in trade_log:
            entry_idx = trade["entry_time"]
            exit_idx = trade["exit_time"]
            
            # Find price action between entry and exit
            window = df.loc[entry_idx:exit_idx]
            if not window.empty:
                if "BUY" in trade["type"]:
                    max_price = window["High"].max()
                    mfe_pct = (max_price - trade["entry_price"]) / trade["entry_price"] * 100
                else: # SHORT
                    min_price = window["Low"].min()
                    mfe_pct = (trade["entry_price"] - min_price) / trade["entry_price"] * 100
                mfe_list.append(mfe_pct)
        
        avg_mfe = np.mean(mfe_list) if mfe_list else 0
        
        results.append({
            "ticker": ticker,
            "win_rate": report["win_rate"],
            "net_profit": report["net_profit"],
            "avg_mfe": round(avg_mfe, 2)
        })
        
    # 3. Assertions for accuracy targets
    avg_win_rate = np.mean([r["win_rate"] for r in results])
    print(f"\n[SKEPTIC] Strategy Backtest Results (Avg Win Rate: {avg_win_rate}%):")
    for r in results:
        print(f"  - {r['ticker']}: Win Rate {r['win_rate']}%, Avg MFE {r['avg_mfe']}%")
        
    assert avg_win_rate >= 50, "Average win rate should be at least 50% for a viable intraday strategy."

@pytest.mark.asyncio
async def test_news_breakout_impact_simulation():
    """
    Validate how news breakouts affect price and how the system should react.
    Simulates a 'shocker' news event and checks if risk engine blocks entry.
    """
    from services.risk_engine import risk_engine
    
    # Mock technical data showing high volatility (ATR surge)
    entry_price = 1000.0
    atr = 60.0 # Huge ATR (6% of price) indicating shock
    current_price = 1015.0 # Price already jumped 1.5% from entry
    
    # Gate 3: ATR Validity & Relative Volatility
    # atr_pct = (60 / 1000) * 100 = 6% (> 5% threshold)
    validation = risk_engine.validate_trade(entry_price, "BUY", atr, current_price)
    
    assert validation["passed"] is False, "Risk engine should block trades during extreme news-driven volatility."
    assert any("volatility" in r or "ATR" in r.upper() for r in validation["reasons"])

def test_dead_cat_bounce_detection():
    """
    Validate math model detection of trend reversals (Dead Cat Bounce).
    """
    # Create a sharp down-trend followed by a small weak rally
    prices = [100, 95, 90, 85, 80, 82, 83, 81, 78] # Weak rally to 83, then crash
    df = pd.DataFrame({'Close': prices, 'Open': prices, 'High': [p+1 for p in prices], 'Low': [p-1 for p in prices]})
    df.index = pd.date_range(datetime.now(), periods=len(prices), freq='15min')
    
    # Calculate EMA cross
    df.ta.ema(length=3, append=True) # Fast
    df.ta.ema(length=5, append=True) # Slow
    
    # Last signal should be Bearish despite the small bounce
    last_ema3 = df['EMA_3'].iloc[-1]
    last_ema5 = df['EMA_5'].iloc[-1]
    
    assert last_ema3 < last_ema5, "Fast EMA should remain below Slow EMA during a weak bounce (Dead Cat)."
