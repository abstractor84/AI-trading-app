import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from services.risk_engine import RiskEngine

@pytest.fixture
def engine():
    return RiskEngine(capital=100000, max_risk_per_trade=1000, max_daily_loss=5000)

def test_config_updates(engine):
    engine.update_config(200000, 2000, 10000)
    assert engine.capital == 200000
    assert engine.max_risk_per_trade == 2000
    assert engine.max_daily_loss == 10000
    
    engine.record_closed_pnl(-1000)
    assert engine.daily_pnl == -1000
    engine.reset_daily()
    assert engine.daily_pnl == 0.0

def test_compute_atr(engine):
    # None or small
    assert engine.compute_atr(None) == 0.0
    df_small = pd.DataFrame({"High": [100]*5, "Low": [95]*5, "Close": [98]*5})
    assert engine.compute_atr(df_small, period=14) == 0.0
    
    # Enough data
    df_med = pd.DataFrame({
        "High": [100, 105, 110, 115, 120],
        "Low": [95, 95, 100, 105, 110],
        "Close": [98, 100, 105, 110, 115]
    })
    # period 2 (len(df)=5 >= 3)
    assert engine.compute_atr(df_med, period=2) > 0

def test_pivots(engine):
    p = engine.compute_pivots(100, 90, 95)
    assert p["pivot"] == 95.0
    
    fib = engine.compute_fibonacci_pivots(100, 90, 95)
    assert fib["pivot"] == 95.0

def test_sl_target_logic(engine):
    res = engine.compute_sl_target(100, "BUY", 2.0)
    assert res["stop_loss"] == 97.0
    
    res = engine.compute_sl_target(100, "SHORT", 2.0)
    assert res["stop_loss"] == 103.0
    
    assert engine.compute_sl_target(100, "BUY", -1) is None
    assert engine.compute_sl_target(100, "HOLD", 2.0) is None

def test_position_sizing(engine):
    assert engine.compute_position_size(100, 98) == 500
    assert engine.compute_position_size(100, 100) == 0

def test_trailing_sl_exhaustive(engine):
    # Buy: Entry 100, ATR 2. SL 1.5*2=3.
    assert engine.compute_trailing_sl(100, 105, "BUY", 2.0) == 102.0 # > 2*ATR
    assert engine.compute_trailing_sl(100, 103, "BUY", 2.0) == 100.0 # > 1*ATR
    assert engine.compute_trailing_sl(100, 101, "BUY", 2.0) == 97.0  # < 1*ATR
    
    # Short Sell: Entry 100, ATR 2. SL 1.5*2=3 (103)
    # Price 95 -> Entry - Price = 5. move 5 >= 2*2(4). return 100 - 2*1 = 98.
    assert engine.compute_trailing_sl(100, 95, "SHORT SELL", 2.0) == 98.0
    # Price 97 -> Entry - Price = 3. move 3 >= 2. return 100.
    assert engine.compute_trailing_sl(100, 97, "SHORT SELL", 2.0) == 100.0
    # Price 99 -> Entry - Price = 1. return 100 + 3 = 103.
    assert engine.compute_trailing_sl(100, 99, "SHORT SELL", 2.0) == 103.0
    
    # Unsupported
    assert engine.compute_trailing_sl(100, 101, "EXIT", 2.0) == 100

def test_validate_trade_exhaustive(engine):
    # Set all failure conditions to hit all branches
    engine.daily_pnl = -10000 
    ta = {"ema_9": 100, "ema_21": 110} # Down trend
    # Action BUY, but trend is down -> Hits Trend Mismatch
    # vix 30 -> Hits VIX gate
    # atr 10 (10% of 100) -> Hits high ATR %
    # current_price 102 (2% drift) -> Hits Slippage gate
    res = engine.validate_trade(100, "BUY", 10.0, current_price=102, ta_data=ta, vix_value=30)
    assert not res["passed"]
    assert len(res["reasons"]) >= 5
    
    # Same for SHORT SELL with trend Up
    engine.reset_daily()
    ta_up = {"ema_9": 110, "ema_21": 100}
    res = engine.validate_trade(100, "SHORT SELL", 2.0, ta_data=ta_up)
    assert not res["passed"]
    assert "TREND MISMATCH" in str(res["reasons"])

def test_get_position_action_short_sl(engine):
    trade_short = {"entry_price": 100, "action": "SHORT SELL", "quantity": 10, "stop_loss": 103, "target_1": 96, "target_2": 90}
    # Current 104 -> SL HIT
    assert "SL HIT" in engine.get_position_action(trade_short, 104, 2.0, 300)["advice"]
    # Current 98 -> HOLD (but check pnl calculation for short)
    res = engine.get_position_action(trade_short, 98, 2.0, 300)
    assert res["pnl_per_share"] == 2.0
    assert res["pnl_total"] == 20.0


def test_get_position_action(engine):
    trade = {"entry_price": 100, "action": "BUY", "quantity": 10, "stop_loss": 97, "target_1": 104, "target_2": 110}
    assert engine.get_position_action(trade, 101, 2.0, 300)["advice"] == "HOLD"
    assert "SL HIT" in engine.get_position_action(trade, 96, 2.0, 300)["advice"]
    assert "T2 HIT" in engine.get_position_action(trade, 111, 2.0, 300)["advice"]
    assert "TRAIL" in engine.get_position_action(trade, 105, 2.0, 300)["advice"]
    assert "Power Hour" in engine.get_position_action(trade, 99, 2.0, 45)["advice"]
    assert "Day Close" in engine.get_position_action(trade, 101, 2.0, 10)["advice"]

def test_risk_engine_extra_branches(engine):
    with patch.object(engine, "compute_sl_target", return_value={"rr_ratio": 0.5, "stop_loss": 99, "risk_per_share": 1}):
        res = engine.validate_trade(100, "BUY", 2.0)
        assert "POOR R:R" in res["reasons"][0]

    with patch.object(engine, "compute_sl_target", return_value=None):
        res = engine.validate_trade(100, "BUY", 2.0)
        assert "Could not compute SL/Target" in res["reasons"][0]
