import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from services.state import AppState
from ws_handler import ConnectionManager
from background_engine import BackgroundEngine
from services.market_phase import MarketPhase

@pytest.fixture
def state():
    st = AppState()
    st.capital = 100000
    return st

@pytest.fixture
def manager():
    return ConnectionManager()

@pytest.fixture
def bg_engine(state, manager):
    return BackgroundEngine(state, manager)

@pytest.mark.asyncio
async def test_update_open_trades(bg_engine, state):
    state.open_trades = [
        {
            "id": "mock_id", "ticker": "RELIANCE.NS", "action": "BUY",
            "quantity": 10, "entry_price": 1000, "stop_loss": 980, 
            "target_1": 1020, "target_2": 1040, "status": "OPEN",
            "trailing_sl": 980, "projections": {}
        }
    ]
    
    phase_ctx = {"phase": MarketPhase.MID_MORNING_TREND, "mins_to_close": 120}
    
    # Mock ta_svc.fetch_ohlcv
    with patch("background_engine.ta_svc.fetch_ohlcv") as mock_fetch:
        import pandas as pd
        df = pd.DataFrame({"Close": [1010]}, index=pd.to_datetime(["2023-01-01"]))
        mock_fetch.return_value = df
        
        with patch("background_engine.risk_engine.compute_atr") as mock_atr, \
             patch("background_engine.risk_engine.compute_trailing_sl") as mock_tsl, \
             patch("background_engine.risk_engine.get_position_action") as mock_adv, \
             patch("background_engine.projection_svc.calculate_projections") as mock_proj:
                 
            mock_atr.return_value = 15.0
            mock_tsl.return_value = 990.0
            mock_adv.return_value = {"advice": "HOLD"}
            mock_proj.return_value = {"ensemble_target": 1030}
            
            await bg_engine._update_open_trades(phase_ctx)
            
            assert state.open_trades[0]["current_price"] == 1010
            assert state.open_trades[0]["pnl"] == 100  # (1010-1000)*10
            assert state.open_trades[0]["trailing_sl"] == 990
            assert state.open_trades[0]["risk_advice"]["advice"] == "HOLD"
            assert state.open_trades[0]["projections"]["ensemble_target"] == 1030

@pytest.mark.asyncio
async def test_maybe_call_ai_scan(bg_engine, state):
    state.open_trades = []
    phase_ctx = {"phase": MarketPhase.OPENING_VOLATILITY, "mins_to_close": 360, "is_trading_hours": True, "phase_label": "Opening"}
    
    with patch("background_engine.market_phase_svc.get_ai_schedule") as mock_sch, \
         patch("background_engine.discovery_svc._get_top_candidates", return_value=["K1"]), \
         patch("background_engine.ta_svc.analyze_stock") as mock_ta, \
         patch("background_engine.ta_svc.evaluate_math_probability", return_value=0.8), \
         patch("background_engine.ta_svc.fetch_ohlcv") as mock_fetch, \
         patch("background_engine.ta_svc.fetch_fundamentals") as mock_fund, \
         patch("background_engine.ta_svc.classify_signal") as mock_sig, \
         patch("background_engine.ai_advisor.scan_market") as mock_scan, \
         patch("services.news_sentiment.NewsSentimentService") as mock_news:
             
        mock_sch.return_value = {"prompt_type": "SCAN", "call_interval_mins": 5}
        mock_ta.return_value = {
            "close": 100, "vwap": 99, "ema_9": 101, "ema_21": 98, 
            "rsi_14": 60, "macd_hist": 2, "adx_14": 25, "vol_surge": 1.2,
            "bb_upper": 105, "bb_lower": 95
        }
        mock_fund.return_value = {"pe_ratio": 20}
        mock_sig.return_value = "STRONG BUY"
        mock_scan.return_value = [{"ticker": "K1", "action": "BUY", "confidence": 0.9}]
        mock_news.return_value.fetch_news.return_value = [{"title": "Good News"}]
        
        await bg_engine._maybe_call_ai(phase_ctx)
        
        assert bg_engine._ai_call_count_today == 1
        assert mock_scan.called
        assert len(state.ai_advisor_message["result"]) == 1
        assert state.ai_advisor_message["result"][0]["ticker"] == "K1"

@pytest.mark.asyncio
async def test_maybe_call_ai_position_review(bg_engine, state):
    state.open_trades = [{"ticker": "K1", "id": "t1", "action": "BUY", "quantity": 10, "entry_price": 100}]
    phase_ctx = {"phase": MarketPhase.MID_MORNING_TREND, "mins_to_close": 200, "is_trading_hours": True, "phase_label": "Mid-Morning"}
    
    with patch("background_engine.market_phase_svc.get_ai_schedule") as mock_sch, \
         patch("background_engine.ai_advisor.review_positions") as mock_rev, \
         patch("background_engine.ta_svc.analyze_stock") as mock_ta, \
         patch("background_engine.ta_svc.evaluate_math_probability", return_value=0.7), \
         patch("background_engine.ta_svc.classify_signal") as mock_sig:
             
        mock_sch.return_value = {"prompt_type": "SCAN", "call_interval_mins": 5}
        mock_rev.return_value = [{"ticker": "K1", "action": "HOLD", "urgency": "LOW"}]
        mock_ta.return_value = {
            "close": 105, "vwap": 104, "ema_9": 106, "ema_21": 103, 
            "rsi_14": 65, "macd_hist": 1, "adx_14": 22, "vol_surge": 1.1,
            "bb_upper": 110, "bb_lower": 100
        }
        mock_sig.return_value = "BUY"
        
        await bg_engine._maybe_call_ai(phase_ctx)
        
        assert bg_engine._ai_call_count_today == 1
        assert mock_rev.called
        assert state.ai_advisor_message["type"] == "POSITION_REVIEW"

@pytest.mark.asyncio
async def test_maybe_call_ai_exit_guidance(bg_engine, state):
    state.open_trades = [{"ticker": "K1", "id": "t1"}]
    phase_ctx = {"phase": MarketPhase.POWER_HOUR, "mins_to_close": 30, "is_trading_hours": True}
    
    with patch("background_engine.market_phase_svc.get_ai_schedule") as mock_sch, \
         patch("background_engine.ai_advisor.exit_guidance", new_callable=AsyncMock) as mock_exit:
             
        mock_sch.return_value = {"prompt_type": "EXIT_GUIDANCE", "call_interval_mins": 5}
        mock_exit.return_value = {"should_close_all": True}
        
        await bg_engine._maybe_call_ai(phase_ctx)
        
        assert bg_engine._ai_call_count_today == 1
        assert mock_exit.called

@pytest.mark.asyncio
async def test_update_open_trades_error(bg_engine, state):
    state.open_trades = [{"ticker": "FAIL.NS", "id": "t1"}]
    phase_ctx = {"phase": MarketPhase.MID_MORNING_TREND, "mins_to_close": 120}
    
    with patch("background_engine.ta_svc.fetch_ohlcv", side_effect=Exception("Data error")):
        # Should not raise exception
        await bg_engine._update_open_trades(phase_ctx)
        assert True

@pytest.mark.asyncio
async def test_maybe_call_ai_quota_exceeded(bg_engine, state):
    phase_ctx = {"phase": MarketPhase.MID_MORNING_TREND, "is_trading_hours": True}
    bg_engine._ai_call_count_today = 20
    
    with patch("background_engine.market_phase_svc.get_ai_schedule") as mock_sch:
        mock_sch.return_value = {"prompt_type": "SCAN", "call_interval_mins": 5}
        await bg_engine._maybe_call_ai(phase_ctx)
        assert bg_engine._ai_call_count_today == 20

@pytest.mark.asyncio
async def test_maybe_call_ai_consensus_fail(bg_engine, state):
    state.open_trades = []
    phase_ctx = {"phase": MarketPhase.OPENING_VOLATILITY, "is_trading_hours": True, "phase_label": "Opening", "mins_to_close": 360}
    
    with patch("background_engine.market_phase_svc.get_ai_schedule") as mock_sch, \
         patch("background_engine.discovery_svc._get_top_candidates", return_value=["K1"]), \
         patch("background_engine.ta_svc.analyze_stock") as mock_ta, \
         patch("background_engine.ta_svc.evaluate_math_probability", return_value=0.8), \
         patch("background_engine.ta_svc.classify_signal") as mock_sig, \
         patch("background_engine.ta_svc.fetch_ohlcv") as mock_fetch, \
         patch("background_engine.ai_advisor.scan_market") as mock_scan, \
         patch("services.news_sentiment.NewsSentimentService") as mock_news:
             
        mock_sch.return_value = {"prompt_type": "SCAN", "call_interval_mins": 5}
        mock_ta.return_value = {
            "close": 100, "vwap": 99, "ema_9": 101, "ema_21": 98, 
            "rsi_14": 60, "macd_hist": 2, "adx_14": 25, "vol_surge": 1.2,
            "bb_upper": 105, "bb_lower": 95
        }
        mock_sig.return_value = "STRONG SHORT SELL" # TA says SHORT SELL
        mock_scan.return_value = [{"ticker": "K1", "action": "BUY"}] # AI says BUY
        mock_news.return_value.fetch_news.return_value = []
        
        await bg_engine._maybe_call_ai(phase_ctx)
        
        # Result should be empty because of consensus block at line 309/311
        assert len(state.ai_advisor_message["result"]) == 0

@pytest.mark.asyncio
async def test_maybe_call_ai_low_prob(bg_engine, state):
    phase_ctx = {"phase": MarketPhase.MID_MORNING_TREND, "is_trading_hours": True, "phase_label": "Trend", "mins_to_close": 300}
    
    with patch("background_engine.market_phase_svc.get_ai_schedule") as mock_sch, \
         patch("background_engine.discovery_svc._get_top_candidates", return_value=["K1"]), \
         patch("background_engine.ta_svc.analyze_stock") as mock_ta, \
         patch("background_engine.ta_svc.evaluate_math_probability", return_value=0.3):
             
        mock_sch.return_value = {"prompt_type": "SCAN", "call_interval_mins": 5}
        mock_ta.return_value = {
            "close": 100, "vwap": 99, "ema_9": 101, "ema_21": 98, 
            "rsi_14": 60, "macd_hist": 2, "adx_14": 25, "vol_surge": 1.2
        }
        
        await bg_engine._maybe_call_ai(phase_ctx)
        assert bg_engine._ai_call_count_today == 1 # still increments before skip

@pytest.mark.asyncio
async def test_run_one_iteration_full(bg_engine, state, manager):
    # Mocking run loop to exit after one iteration, using MagicMock for to_thread calls
    with patch("background_engine.asyncio.sleep", side_effect=[None, Exception("Stop loop")]), \
         patch("background_engine.market_phase_svc.get_phase_context") as mock_ctx, \
         patch("background_engine.ta_svc.get_connection_status") as mock_conn, \
         patch("background_engine.discovery_svc.fetch_global_indices") as mock_glob, \
         patch.object(manager, "broadcast", new_callable=AsyncMock) as mock_broad:
             
        mock_ctx.return_value = {
            "phase": MarketPhase.MID_MORNING_TREND, 
            "is_trading_hours": True, 
            "transitioned": True,
            "mins_to_close": 300,
            "phase_label": "Test Phase"
        }
        mock_glob.return_value = {"india": {"NIFTY 50": {"value": 22000, "change_pct": 0.5}}}
        
        try:
            await bg_engine.run()
        except Exception as e:
            assert str(e) == "Stop loop"
            
        assert mock_broad.called
        # Check transition notification was sent
        assert any(args[0]["type"] == "notification" for args, _ in mock_broad.call_args_list)
        # Check state update was sent
        assert any(args[0]["type"] == "state_update" for args, _ in mock_broad.call_args_list)





def test_get_sleep_interval(bg_engine):
    assert bg_engine._get_sleep_interval(MarketPhase.PRE_MARKET_SETUP) == 60
    assert bg_engine._get_sleep_interval(MarketPhase.CLOSED) == 300
    assert bg_engine._get_sleep_interval("UNKNOWN") == 60


