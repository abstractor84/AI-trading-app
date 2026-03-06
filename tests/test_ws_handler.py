import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

from services.state import AppState
from ws_handler import ConnectionManager, handle_websocket
import services.upstox_streamer

class MockWebsocket:
    def __init__(self, actions):
        self.actions = actions
        self.sent_messages = []
        self.accepted = False
        self.closed = False
        
    async def accept(self):
        self.accepted = True
        
    async def receive_text(self):
        if not self.actions:
            from fastapi import WebSocketDisconnect
            raise WebSocketDisconnect(1000)
        return json.dumps(self.actions.pop(0))
        
    async def send_json(self, data):
        self.sent_messages.append(data)

@pytest.fixture
def state():
    return AppState()

@pytest.fixture
def manager():
    return ConnectionManager()

@pytest.mark.asyncio
async def test_ws_update_settings(state, manager):
    ws = MockWebsocket([{"action": "update_settings", "capital": 50000, "max_loss": 500, "search_engine": "tavily"}])
    await handle_websocket(ws, manager, state)
    
    assert state.capital == 50000
    assert state.max_loss_per_trade == 500
    assert state.search_engine == "tavily"
    
@pytest.mark.asyncio
async def test_ws_log_trade(state, manager):
    state.open_trades.clear()
    ws = MockWebsocket([{
        "action": "log_trade", "ticker": "TCS.NS", "trade_action": "BUY", "entry_price": 3500
    }])
    
    with patch("ws_handler.risk_engine.compute_atr", return_value=50.0), \
         patch("services.technical_analysis.TechnicalAnalysisService.fetch_ohlcv") as mock_fetch, \
         patch("services.technical_analysis.TechnicalAnalysisService.analyze_stock") as mock_analyze, \
         patch("ws_handler.risk_engine.validate_trade") as mock_val:
             
        import pandas as pd
        mock_fetch.return_value = pd.DataFrame({"Close": [3500]}, index=pd.to_datetime(["2023-01-01"]))
        mock_val.return_value = {
            "passed": True, 
            "levels": {"stop_loss": 3450, "target_1": 3600, "target_2": 3700, "risk_per_share": 50},
            "quantity": 10,
            "max_loss_this_trade": 500
        }
        
        await handle_websocket(ws, manager, state)
        assert len(state.open_trades) == 1

@pytest.mark.asyncio
async def test_ws_close_trade(state, manager):
    # Pre-populate an open trade
    state.open_trades.append({
        "id": "trade_1", "ticker": "TCS.NS", "action": "BUY", 
        "quantity": 10, "entry_price": 3500
    })
    
    ws = MockWebsocket([{"action": "close_trade", "trade_id": "trade_1", "exit_price": 3600}])
    
    with patch("services.state.AppState.close_trade") as mock_close:
        await handle_websocket(ws, manager, state)
        mock_close.assert_called_with("trade_1", 3600)

@pytest.mark.asyncio
async def test_get_chart_data(state, manager):
    ws = MockWebsocket([{"action": "get_chart_data", "ticker": "RELIANCE.NS"}])
    
    with patch("services.technical_analysis.TechnicalAnalysisService.fetch_ohlcv") as mock_fetch, \
         patch("services.price_projector.price_projector.generate_projection") as mock_proj, \
         patch("services.upstox_service.get_instrument_key") as mock_key, \
         patch("services.upstox_streamer.get_streamer") as mock_stream:
             
        import pandas as pd
        mock_fetch.return_value = pd.DataFrame({"Close": [2500]}, index=pd.to_datetime(["2023-01-01"]))
        mock_proj.return_value = {"timestamps": ["2023-01-01 15:00:00"], "projection": [2510]}
        mock_key.return_value = "NSE_EQ|INE1234"
        
        mock_str_inst = MagicMock()
        mock_stream.return_value = mock_str_inst
        
        # Mock ML functions to avoid loading scikit-learn heavily here
        with patch("services.advanced_indicators.classifier.classify_series", return_value=[]), \
             patch("services.advanced_indicators.adaptive_st.calculate", return_value={"value":[],"time":[]}), \
             patch("services.advanced_indicators.knn_forecaster.forecast", return_value=[]):
                 
             await handle_websocket(ws, manager, state)
             
             # Check if chart_data was sent
             has_chart = any(m["type"] == "chart_data" for m in ws.sent_messages)
             assert has_chart
             # Streamer should be subscribed
             mock_str_inst.subscribe.assert_called_with(["NSE_EQ|INE1234"])

@pytest.mark.asyncio
async def test_ws_trigger_scan(state, manager):
    ws = MockWebsocket([{"action": "trigger_scan"}])
    
    with patch("services.technical_analysis.TechnicalAnalysisService") as mock_ta_cls, \
         patch("services.stock_discovery.StockDiscoveryService") as mock_disc_cls, \
         patch("services.ai_scorer.ai_advisor.scan_market") as mock_ai_scan, \
         patch("services.news_sentiment.NewsSentimentService") as mock_news_cls:
             
        mock_disc = mock_disc_cls.return_value
        mock_disc._get_top_candidates.return_value = ["REL.NS"]
        
        mock_ta = mock_ta_cls.return_value
        # Mocking values so enrichment loop doesn't skip/fail
        mock_ta.analyze_stock.return_value = {"close": 2500, "vwap": 2490, "ema_9": 2480, "ema_21": 2470, "rsi_14": 60, "adx_14": 25, "vol_surge": 1.5}
        mock_ta.evaluate_math_probability.return_value = 0.8
        mock_ta.classify_signal.return_value = "STRONG BUY"
        mock_ta.fetch_ohlcv.return_value = MagicMock() # df
        
        # scan_market called via to_thread, so it must be a regular mock returning values
        mock_ai_scan.return_value = [{"ticker": "REL.NS", "action": "BUY", "confidence": 0.9}]
        
        await handle_websocket(ws, manager, state)
        
        # Check for scan_results in sent messages
        has_results = any(m["type"] == "scan_results" for m in ws.sent_messages)
        assert has_results
        results_msg = next(m for m in ws.sent_messages if m["type"] == "scan_results")
        assert len(results_msg["data"]) > 0


@pytest.mark.asyncio
async def test_ws_run_backtest(state, manager):
    ws = MockWebsocket([{"action": "run_backtest", "ticker": "REL.NS", "params": {"sma_fast": 10}}])
    
    with patch("services.technical_analysis.TechnicalAnalysisService.fetch_ohlcv") as mock_fetch, \
         patch("services.backtester.VectorizedBacktester.run_strategy") as mock_bt:
             
        import pandas as pd
        mock_fetch.return_value = pd.DataFrame({"Close": [2500]}, index=pd.to_datetime(["2023-01-01"]))
        mock_bt.return_value = {"total_return": 5.0, "trades": []}
        
        await handle_websocket(ws, manager, state)
        
        has_bt_result = any(m["type"] == "backtest_results" for m in ws.sent_messages)
        assert has_bt_result

@pytest.mark.asyncio
async def test_ws_ai_optimize(state, manager):
    ws = MockWebsocket([{"action": "ai_optimize", "ticker": "REL.NS", "params": {"sma_fast": 10}}])
    
    with patch("services.technical_analysis.TechnicalAnalysisService.fetch_ohlcv") as mock_fetch, \
         patch("services.strategy_tuner.StrategyTuner.optimize", new_callable=AsyncMock) as mock_opt:
             
        import pandas as pd
        mock_fetch.return_value = pd.DataFrame({"Close": [2500]}, index=pd.to_datetime(["2023-01-01"]))
        mock_opt.return_value = {"optimized_params": {"sma_fast": 12}, "improvement": 2.0}
        
        await handle_websocket(ws, manager, state)
        
        has_opt_result = any(m["type"] == "backtest_results" for m in ws.sent_messages)
        assert has_opt_result

@pytest.mark.asyncio
async def test_ws_get_status(state, manager):
    ws = MockWebsocket([{"action": "get_status"}])
    
    with patch("services.technical_analysis.TechnicalAnalysisService.get_connection_status") as mock_status:
        mock_status.return_value = {"broker": "connected", "feed": "live"}
        
        await handle_websocket(ws, manager, state)
        
        has_status = any(m["type"] == "connection_status" for m in ws.sent_messages)
        assert has_status


