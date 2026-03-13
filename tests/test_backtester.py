import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.backtester import VectorizedBacktester

@pytest.fixture
def sample_df():
    """Generate 100 rows of dummy OHLCV data."""
    dates = [datetime(2024, 1, 1, 9, 15) + timedelta(minutes=i) for i in range(100)]
    df = pd.DataFrame({
        "Open": np.linspace(100, 110, 100),
        "High": np.linspace(101, 111, 100),
        "Low": np.linspace(99, 109, 100),
        "Close": np.linspace(100.5, 110.5, 100),
        "Volume": np.random.randint(1000, 5000, 100)
    }, index=dates)
    return df

def test_backtester_initialization(sample_df):
    bt = VectorizedBacktester(sample_df, initial_capital=50000, risk_pct=0.02)
    assert bt.initial_capital == 50000
    assert bt.risk_pct == 0.02
    assert len(bt.df) == 100

def test_prepare_indicators(sample_df):
    bt = VectorizedBacktester(sample_df)
    params = {"ema_fast": 5, "ema_slow": 10, "rsi_len": 14}
    bt._prepare_indicators(params)
    
    assert "EMA_5" in bt.df.columns
    assert "EMA_10" in bt.df.columns
    assert "RSI_14" in bt.df.columns
    assert "VWAP" in bt.df.columns
    # Check that rows with NaNs are dropped (approx 10 rows for slow EMA)
    assert len(bt.df) < 100

def test_run_strategy_no_trades(sample_df):
    # Data is purely trending up, so Mean Reversion (RSI < 40 + Price > VWAP) might not trigger
    bt = VectorizedBacktester(sample_df)
    params = {"ema_fast": 5, "ema_slow": 10, "rsi_len": 14, "rsi_buy_threshold": 20}
    metrics = bt.run_strategy(params)
    
    assert metrics["total_trades"] == 0
    assert metrics["final_equity"] == 100000

def test_run_strategy_tp_hits():
    # Synthetic data for TP hits
    dates = [datetime(2024, 1, 1, 9, 15) + timedelta(minutes=i) for i in range(20)]
    df = pd.DataFrame({
        "Open": [100]*10 + [110]*10,
        "High": [120]*10 + [111]*10, # BUY hits TP at row 0
        "Low":  [99]*20,
        "Close": [100]*10 + [110]*10,
        "Volume": [1000]*20
    }, index=dates)
    bt = VectorizedBacktester(df)
    def mock_prepare(params):
        bt.df["EMA_9"] = [105]*10 + [95]*10
        bt.df["EMA_21"] = [100]*10 + [100]*10
        bt.df["RSI_14"] = [30]*10 + [70]*10
        bt.df["VWAP"] = [90]*10 + [120]*10
        bt.df["Date"] = bt.df.index.date
        bt.df["LZ_Signal"] = [1]*10 + [-1]*10
    bt._prepare_indicators = mock_prepare
    params = {"ema_fast": 9, "ema_slow": 21, "rsi_len": 14, "sl_pct": 0.1, "tp_pct": 0.05}
    metrics = bt.run_strategy(params)
    
    types = [t["type"] for t in metrics["trade_log"]]
    # BUY TP = 100 * 1.05 = 105. High is 120. Hit!
    # SHORT TP = 110 * 0.95 = 104.5. Low is 99. Hit!
    assert "BUY" in types
    assert "SHORT" in types
    assert metrics["win_rate"] == 100.0

def test_run_strategy_with_real_trades():
    # Synthetic data for BUY and SHORT
    # 0-9: BUY signal (Fast > Slow, RSI < 40, Close > VWAP)
    # 10-14: SL Hit for BUY
    # 15-24: SHORT signal (Fast < Slow, RSI > 60, Close < VWAP)
    # 25-29: TP Hit for SHORT
    dates = [datetime(2024, 1, 1, 9, 15) + timedelta(minutes=i) for i in range(30)]
    
    # We need to construct OHLC such that TA indicators follow the pattern
    # For simplicity, we'll manually patch _prepare_indicators to set our columns
    # OR better: feed OHLC that natural TA likes.
    
    df = pd.DataFrame({
        "Open": [100]*10 + [100]*10 + [100]*10,
        "High": [101]*10 + [101]*10 + [101]*10,
        "Low":  [99]*10 + [90]*5 + [99]*5 + [80]*10, # Drop at 20-29 to hit TP for SHORT
        "Close": [100.5]*10 + [92]*5 + [100.5]*5 + [92]*10,
        "Volume": [1000]*30
    }, index=dates)

    
    bt = VectorizedBacktester(df)
    
    # Patch _prepare_indicators to force our signal state
    def mock_prepare(params):
        bt.df["EMA_9"] = [105]*10 + [105]*10 + [95]*10
        bt.df["EMA_21"] = [100]*10 + [100]*10 + [100]*10
        bt.df["RSI_14"] = [30]*10 + [30]*10 + [70]*10
        bt.df["VWAP"] = [100]*30
        bt.df["Date"] = bt.df.index.date
        bt.df["LZ_Signal"] = [1]*10 + [0]*10 + [-1]*10
        
    bt._prepare_indicators = mock_prepare
    
    params = {"ema_fast": 9, "ema_slow": 21, "rsi_len": 14, "sl_pct": 0.05, "tp_pct": 0.1}
    metrics = bt.run_strategy(params)
    
    # print(f"Signals: {bt.df['Signal'].value_counts().to_dict()}")
    # print(f"Trades: {[t['type'] for t in metrics['trade_log']]}")
    
    assert metrics["total_trades"] >= 2


    # Check trade log
    types = [t["type"] for t in metrics["trade_log"]]
    assert "BUY" in types
    assert "SHORT" in types

def test_evaluate_metrics_math():

    bt = VectorizedBacktester(pd.DataFrame())
    trades = [
        {"type": "BUY", "pnl_pct": 0.02}, # Win
        {"type": "SHORT", "pnl_pct": -0.01}, # Loss
    ]
    metrics = bt._evaluate_metrics(trades)
    
    assert metrics["total_trades"] == 2
    assert metrics["win_rate"] == 50.0
    # Final equity: 100000 * 1.02 = 102000, then 102000 * 0.99 = 100980
    assert metrics["final_equity"] == 100980.0
    assert metrics["net_profit"] == 980.0
