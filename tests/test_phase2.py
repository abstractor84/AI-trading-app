import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.market_phase import MarketPhase, MarketPhaseService
from services.ai_scorer import AIAdvisorService
from services.stock_discovery import StockDiscoveryService
from services.news_sentiment import NewsSentimentService

# --- Test Market Phases ---
@patch("services.market_phase.datetime")
def test_pre_market_setup(mock_dt):
    mock_dt.now.return_value = datetime.combine(datetime.today(), time(9, 10))
    machine = MarketPhaseService()
    phase_ctx = machine.get_phase_context()
    assert phase_ctx["phase"] == MarketPhase.PRE_MARKET_SETUP.value
    assert "Setup" in phase_ctx["phase_label"]
    assert phase_ctx["mins_to_close"] > 300

@patch("services.market_phase.datetime")
def test_morning_rush(mock_dt):
    mock_dt.now.return_value = datetime.combine(datetime.today(), time(9, 20))
    machine = MarketPhaseService()
    phase_ctx = machine.get_phase_context()
    assert phase_ctx["phase"] == MarketPhase.OPENING_VOLATILITY.value

@patch("services.market_phase.datetime")
def test_lunch_chop(mock_dt):
    mock_dt.now.return_value = datetime.combine(datetime.today(), time(12, 30))
    machine = MarketPhaseService()
    phase_ctx = machine.get_phase_context()
    assert phase_ctx["phase"] == MarketPhase.LUNCH_CHOP.value

@patch("services.market_phase.datetime")
def test_power_hour(mock_dt):
    mock_dt.now.return_value = datetime.combine(datetime.today(), time(14, 45))
    machine = MarketPhaseService()
    phase_ctx = machine.get_phase_context()
    assert phase_ctx["phase"] == MarketPhase.POWER_HOUR.value
    assert phase_ctx["mins_to_close"] == 45

# --- Test Stock Discovery (DDGS logic) ---
@patch("services.news_sentiment._ddgs_fetch")
def test_ddgs_search_called(mock_ddgs):
    # Mock DuckDuckGo answers directly at the fetch layer
    mock_ddgs.return_value = ["News 1", "News 2"]
    
    svc = NewsSentimentService()
    results = svc.fetch_news("RELIANCE", search_engine="ddgs")
            
    # Verify DuckDuckGo is called when scanning
    mock_ddgs.assert_called_with("RELIANCE NSE India stock market news today")
    assert len(results) == 2
    assert "News 1" in results[0]


# --- Test AI Fallback Toggle ---
@patch("services.ai_scorer.quota_svc.check_quota")
@patch.object(AIAdvisorService, "_call_google")
def test_ai_fallback_disabled(mock_call_google, mock_quota):
    # Mock quota to pass
    mock_quota.return_value = {"can_call": True, "limit": 30, "remaining": 30}
    
    # Mock Google AI to immediately fail with a 429 error
    mock_call_google.side_effect = Exception("429 Too Many Requests")
    
    ai = AIAdvisorService()
    
    # Call with fallback disabled
    result = ai._call_ai("Test Prompt", "SCAN", "google", "gemini-2.5-pro", ai_fallback=False)
    
    # Verify it failed and returned the error, didn't fallback
    assert "error" in result
    assert "429" in result["error"]
    # Ensure it only tried once
    assert mock_call_google.call_count == 1

@patch("services.ai_scorer.quota_svc.check_quota")
@patch("services.ai_scorer.quota_svc.log_usage")
def test_ai_fallback_enabled(mock_log, mock_quota):
    # Mock quota to pass
    mock_quota.return_value = {"can_call": True, "limit": 30, "remaining": 30}
    
    ai = AIAdvisorService()
    
    with patch.object(ai, "_call_google") as mock_call_google:
        # Mock Google AI to throw 429 on 'gemini-2.5-pro', but succeed on 'gemini-2.5-flash'
        def side_effect(model_name, prompt):
            if model_name != "gemini-2.5-flash":
                raise Exception("429 Too Many Requests")
            return [{"ticker": "RELIANCE.NS", "action": "BUY"}]
            
        mock_call_google.side_effect = side_effect
        
        # Call with fallback enabled (default)
        result = ai._call_ai("Test Prompt", "SCAN", "google", "gemini-2.5-pro", ai_fallback=True)
        
        # Verify it successfully fell back to gemini-2.5-flash
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["ticker"] == "RELIANCE.NS"
        assert mock_call_google.call_count == 2
