import pytest
import json
import responses
from unittest.mock import patch, MagicMock

from services.ai_scorer import AIAdvisorService, ai_advisor
from datetime import datetime

@pytest.fixture
def svc():
    return AIAdvisorService()

def test_scan_market_basic(svc):
    candidates = [{"ticker": "RELIANCE.NS", "ta_data": {"close": 2500, "rsi_14": 50}, "math_prob": 0.8}]
    global_ctx = {"india": {"NIFTY 50": {"value": 22000, "change_pct": 0.5}}, "vix": {"value": 15}}
    phase_ctx = {"phase_label": "Opening", "mins_to_close": 300}
    
    with patch.object(svc, "_call_ai") as mock_call:
        mock_call.return_value = [{"ticker": "RELIANCE.NS", "action": "BUY"}]
        res = svc.scan_market(candidates, global_ctx, phase_ctx)
        assert res[0]["ticker"] == "RELIANCE.NS"

def test_review_positions_with_data(svc):
    trades = [{"ticker": "SBIN.NS", "action": "BUY", "quantity": 100, "entry_price": 600, "pnl": 50, "stop_loss": 590}]
    global_ctx = {"india": {"NIFTY 50": {"value": 22000}}, "vix": {"value": 15}}
    phase_ctx = {"phase_label": "Mid-Day", "mins_to_close": 180}
    with patch.object(svc, "_call_ai") as mock_call:
        mock_call.return_value = [{"ticker": "SBIN.NS", "action": "HOLD"}]
        res = svc.review_positions(trades, global_ctx, phase_ctx)
        assert res[0]["action"] == "HOLD"

def test_exit_guidance_with_data(svc):
    trades = [{"ticker": "TCS.NS", "pnl": 500, "action": "BUY"}]
    global_ctx = {"india": {"NIFTY 50": {"change_pct": 0.8}}}
    phase_ctx = {"phase_label": "Power Hour", "mins_to_close": 30}
    with patch.object(svc, "_call_ai") as mock_call:
        mock_call.return_value = {"should_close_all": True}
        res = svc.exit_guidance(trades, global_ctx, phase_ctx)
        assert res["should_close_all"] is True



def test_review_positions_empty(svc):
    res = svc.review_positions([], {}, {})
    assert "no open positions" in res["summary"].lower()

def test_exit_guidance_empty(svc):
    res = svc.exit_guidance([], {}, {})
    assert "no open positions" in res["advice"].lower()

@patch("services.ai_scorer.quota_svc.check_quota")
def test_call_ai_quota_exceeded(mock_quota, svc):
    mock_quota.return_value = {"can_call": False}
    res = svc._call_ai("prompt", "SCAN", "google", "gemini")
    assert "quota exceeded" in res["error"].lower()

@patch("services.ai_scorer.quota_svc.check_quota")
def test_call_ai_dispatch(mock_quota, svc):
    mock_quota.return_value = {"can_call": True}
    with patch.object(svc, "_call_google") as m_goog, \
         patch.object(svc, "_call_groq") as m_groq, \
         patch.object(svc, "_call_sambanova") as m_samba:
        
        svc._call_ai("p", "T", "google", "m")
        assert m_goog.called
        
        svc._call_ai("p", "T", "groq", "m")
        assert m_groq.called
        
        svc._call_ai("p", "T", "sambanova", "m")
        assert m_samba.called

@patch("requests.post")
def test_call_groq_success(mock_post, svc):
    svc.groq_key = "key"
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"choices": [{"message": {"content": '{"res": "ok"}'}}]}
    mock_post.return_value = mock_res
    
    res = svc._call_groq("model", "prompt")
    assert res["res"] == "ok"

@patch("requests.post")
def test_call_sambanova_success(mock_post, svc):
    svc.samba_key = "key"
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"choices": [{"message": {"content": '{"res": "samba_ok"}'}}]}
    mock_post.return_value = mock_res
    
    res = svc._call_sambanova("model", "prompt")
    assert res["res"] == "samba_ok"

def test_parse_json_markdown(svc):
    text = "Here is the result: ```json\n{\"key\": \"val\"}\n```"
    res = svc._parse_json_response(text)
    assert res["key"] == "val"

def test_parse_json_conversational(svc):
    text = "The AI says [{\"ticker\": \"AAPL\"}] in its reasoning."
    res = svc._parse_json_response(text)
    assert res[0]["ticker"] == "AAPL"

def test_parse_json_error(svc):
    res = svc._parse_json_response("Not a JSON")
    assert res["parse_error"] is True

@patch("services.ai_scorer.SessionLocal")
def test_log_interaction_exception(mock_session, svc):
    mock_session.side_effect = Exception("DB Fail")
    # Should not raise
    svc._log_interaction("T", "M", "S", {})
    assert True

def test_google_key_missing(svc):
    svc.google_key = None
    res = svc._call_google("m", "p")
    assert "error" in res

def test_groq_key_missing(svc):
    svc.groq_key = None
    res = svc._call_groq("m", "p")
    assert "error" in res

@patch("requests.post")
def test_call_google_fallback(mock_post, svc):
    svc.google_key = "k"
    mock_quota = MagicMock()
    mock_quota.return_value = {"can_call": True}
    with patch("services.ai_scorer.quota_svc.check_quota", mock_quota):
        # 429 on expensive model
        with patch.object(svc, "_call_google") as mock_goog:
            mock_goog.side_effect = [Exception("429"), {"fallback": "ok"}]
            res = svc._call_ai("p", "T", "google", "expensive", ai_fallback=True)
            assert res["fallback"] == "ok"
            assert mock_goog.call_count == 2

def test_call_ai_unknown_provider(svc):
    quota_mock = MagicMock(return_value={"can_call": True})
    with patch("services.ai_scorer.quota_svc.check_quota", quota_mock):
        res = svc._call_ai("p", "T", "unknown", "m")
        assert "unknown provider" in res["error"].lower()

def test_call_ai_quota_warning(svc):
    quota_mock = MagicMock(return_value={"can_call": True, "low_quota": True})
    with patch("services.ai_scorer.quota_svc.check_quota", quota_mock), \
         patch.object(svc, "_call_google", return_value={"res": "ok"}), \
         patch.object(svc, "_log_interaction"):
        res = svc._call_ai("p", "T", "google", "m")
        assert "quota_warning" in res

def test_call_ai_fallback_disabled(svc):
    quota_mock = MagicMock(return_value={"can_call": True})
    with patch("services.ai_scorer.quota_svc.check_quota", quota_mock), \
         patch.object(svc, "_call_google", side_effect=Exception("429")):
        res = svc._call_ai("p", "T", "google", "expensive", ai_fallback=False)
        assert "429" in res["error"]

def test_parse_json_list_fallback(svc):
    text = "The pick is [{\"ticker\": \"K\"}]"
    res = svc._parse_json_response(text)
    assert res[0]["ticker"] == "K"

def test_parse_json_dict_fallback(svc):
    text = "The score is {\"score\": 10}"
    res = svc._parse_json_response(text)
    assert res["score"] == 10

@patch("services.ai_scorer.genai.Client")
def test_call_google_real(mock_genai_client, svc):
    svc.google_key = "key"
    svc.google_client = mock_genai_client.return_value
    mock_response = MagicMock()
    mock_response.text = '{"res": "ok"}'
    svc.google_client.models.generate_content.return_value = mock_response
    
    res = svc._call_google("model", "prompt")
    assert res["res"] == "ok"

@responses.activate
def test_call_groq_error(svc):
    svc.groq_key = "key"
    responses.add(responses.POST, "https://api.groq.com/openai/v1/chat/completions", status=500)
    res = svc._call_groq("model", "prompt")
    assert "returned 500" in res["error"]

def test_call_ai_fallback_terminal_failure(svc):
    quota_mock = MagicMock(return_value={"can_call": True})
    with patch("services.ai_scorer.quota_svc.check_quota", quota_mock), \
         patch.object(svc, "_call_google", side_effect=Exception("429")) as mock_goog:
        # Both primary and fallback fail
        res = svc._call_ai("p", "T", "google", "expensive", ai_fallback=True)
        assert mock_goog.call_count == 2
        assert "error" in res

def test_parse_json_complex_fences(svc):
    text = "```\n{\"a\": 1}\n```"
    res = svc._parse_json_response(text)
    assert res["a"] == 1
    
    text = "Plain text then {\"b\": 2} more text"
    res = svc._parse_json_response(text)
    assert res["b"] == 2




