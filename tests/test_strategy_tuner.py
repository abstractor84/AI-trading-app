import pytest
import json
import pandas as pd
from unittest.mock import MagicMock, patch, patch
from services.strategy_tuner import StrategyTuner

@pytest.fixture
def tuner():
    with patch("services.strategy_tuner.genai.Client"):
        return StrategyTuner()

@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "High": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        "Open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "Volume": [1000] * 10
    }, index=pd.to_datetime(["2023-01-01"] * 10))

def test_optimize_zero_trades(tuner, mock_df):
    """Test optimized early exit/blind mutation when 0 trades are found."""
    initial_params = {"ema_fast": 9, "sl_pct": 0.01}
    
    with patch("services.strategy_tuner.VectorizedBacktester") as mock_bt_class:
        mock_bt = mock_bt_class.return_value
        mock_bt.run_strategy.return_value = {
            "net_profit": 0, "total_trades": 0, "max_drawdown_pct": 0
        }
        
        # Iteration 1 returns 0 trades, so it should do a blind mutation
        result = tuner.optimize("RELIANCE.NS", mock_df, initial_params, iterations=2)
        
        assert len(result["history"]) == 2
        # Rounding check 0.01 * 1.2 = 0.012
        assert result["history"][1]["params"]["sl_pct"] == 0.012

def test_optimize_google_provider(tuner, mock_df):
    """Test the full loop with Google AI provider mocked."""
    initial_params = {"ema_fast": 9, "sl_pct": 0.01}
    tuner.google_client = MagicMock()
    
    with patch("services.strategy_tuner.VectorizedBacktester") as mock_bt_class, \
         patch("services.strategy_tuner.quota_svc") as mock_quota:
        
        mock_bt = mock_bt_class.return_value
        # Use a side effect to return higher profit in second iteration
        mock_bt.run_strategy.side_effect = [
            {"net_profit": 500, "total_trades": 10, "max_drawdown_pct": 5, "win_rate": 0.5},
            {"net_profit": 500, "total_trades": 10, "max_drawdown_pct": 5, "win_rate": 0.5}, # bt.run_strategy is called twice per iteration
            {"net_profit": 700, "total_trades": 10, "max_drawdown_pct": 5, "win_rate": 0.5},
            {"net_profit": 700, "total_trades": 10, "max_drawdown_pct": 5, "win_rate": 0.5},
        ]
        
        mock_quota.check_quota.return_value = {"can_call": True}
        
        # Mock AI response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "analysis": "Test analysis",
            "new_params": {"ema_fast": 12, "sl_pct": 0.008}
        })
        tuner.google_client.models.generate_content.return_value = mock_response
        
        result = tuner.optimize("RELIANCE.NS", mock_df, initial_params, iterations=2, provider="google")
        
        assert len(result["history"]) == 2
        assert result["history"][1]["params"]["ema_fast"] == 12
        assert result["best_parameters"]["ema_fast"] == 12

def test_optimize_quota_fallback(tuner, mock_df):
    """Test genetic fallback when quota is exceeded."""
    initial_params = {"ema_fast": 9, "sl_pct": 0.01}
    
    with patch("services.strategy_tuner.VectorizedBacktester") as mock_bt_class, \
         patch("services.strategy_tuner.quota_svc") as mock_quota:
        
        mock_bt = mock_bt_class.return_value
        mock_bt.run_strategy.return_value = {
            "net_profit": 500, "total_trades": 10, "max_drawdown_pct": 5
        }
        
        mock_quota.check_quota.return_value = {"can_call": False}
        
        result = tuner.optimize("RELIANCE.NS", mock_df, initial_params, iterations=2)
        
        # Fallback: ema_fast + 1, sl_pct * 0.9
        assert result["history"][1]["params"]["ema_fast"] == 10
        assert result["history"][1]["params"]["sl_pct"] == pytest.approx(0.009)

def test_optimize_groq_provider(tuner, mock_df):
    """Test Groq provider branch."""
    tuner.groq_key = "test_key"
    initial_params = {"ema_fast": 9, "sl_pct": 0.01}
    
    with patch("services.strategy_tuner.VectorizedBacktester") as mock_bt_class, \
         patch("services.strategy_tuner.quota_svc") as mock_quota, \
         patch("requests.post") as mock_post:
        
        mock_bt = mock_bt_class.return_value
        mock_bt.run_strategy.return_value = {
            "net_profit": 500, "total_trades": 10, "max_drawdown_pct": 5
        }
        mock_quota.check_quota.return_value = {"can_call": True}
        
        mock_res = MagicMock()
        mock_res.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"analysis": "Groq test", "new_params": {"ema_fast": 15}})}}],
            "usage": {"total_tokens": 100}
        }
        mock_post.return_value = mock_res
        
        result = tuner.optimize("RELIANCE.NS", mock_df, initial_params, iterations=2, provider="groq")
        assert result["history"][1]["params"]["ema_fast"] == 15

def test_optimize_sambanova_provider(tuner, mock_df):
    """Test SambaNova provider branch."""
    tuner.samba_key = "test_key"
    initial_params = {"ema_fast": 9, "sl_pct": 0.01}
    
    with patch("services.strategy_tuner.VectorizedBacktester") as mock_bt_class, \
         patch("services.strategy_tuner.quota_svc") as mock_quota, \
         patch("requests.post") as mock_post:
        
        mock_bt = mock_bt_class.return_value
        mock_bt.run_strategy.return_value = {
            "net_profit": 500, "total_trades": 10, "max_drawdown_pct": 5
        }
        mock_quota.check_quota.return_value = {"can_call": True}
        
        mock_res = MagicMock()
        mock_res.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"analysis": "Samba test", "new_params": {"ema_fast": 20}})}}],
            "usage": {"total_tokens": 100}
        }
        mock_post.return_value = mock_res
        
        result = tuner.optimize("RELIANCE.NS", mock_df, initial_params, iterations=2, provider="sambanova")
        assert result["history"][1]["params"]["ema_fast"] == 20

def test_optimize_missing_keys(tuner, mock_df):
    """Test behavior when API keys are missing."""
    initial_params = {"ema_fast": 9, "sl_pct": 0.01}
    
    with patch("services.strategy_tuner.VectorizedBacktester") as mock_bt_class, \
         patch("services.strategy_tuner.quota_svc") as mock_quota:
        
        mock_bt = mock_bt_class.return_value
        mock_bt.run_strategy.return_value = {"net_profit": 0, "total_trades": 5, "max_drawdown_pct": 0}
        mock_quota.check_quota.return_value = {"can_call": True}
        
        # Google missing
        tuner.google_client = None
        result = tuner.optimize("TCS.NS", mock_df, initial_params, iterations=2, provider="google")
        assert result["history"][1]["params"]["ema_fast"] == 10 # Fallback
        
        # Groq missing
        tuner.groq_key = None
        result = tuner.optimize("TCS.NS", mock_df, initial_params, iterations=2, provider="groq")
        assert result["history"][1]["params"]["ema_fast"] == 11 # Fallback (previous + 1)
        
        # Samba missing
        tuner.samba_key = None
        result = tuner.optimize("TCS.NS", mock_df, initial_params, iterations=2, provider="sambanova")
        assert result["history"][1]["params"]["ema_fast"] == 12

def test_optimize_exception_handling(tuner, mock_df):
    """Test fallback when an exception occurs during AI call."""
    initial_params = {"ema_fast": 9, "sl_pct": 0.01}
    tuner.google_client = MagicMock()
    tuner.google_client.models.generate_content.side_effect = Exception("AI Error")
    
    with patch("services.strategy_tuner.VectorizedBacktester") as mock_bt_class, \
         patch("services.strategy_tuner.quota_svc") as mock_quota:
        
        mock_bt = mock_bt_class.return_value
        mock_bt.run_strategy.return_value = {"net_profit": 500, "total_trades": 10, "max_drawdown_pct": 5}
        mock_quota.check_quota.return_value = {"can_call": True}
        
        result = tuner.optimize("RELIANCE.NS", mock_df, initial_params, iterations=2)
        assert result["history"][1]["params"]["ema_fast"] == 10

