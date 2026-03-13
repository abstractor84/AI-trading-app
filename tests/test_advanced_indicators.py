import pytest
import pandas as pd
import numpy as np
from services.advanced_indicators import LorentzianClassifier, AdaptiveSuperTrend, KNNTrendForecaster

@pytest.fixture
def sample_df():
    dates = pd.date_range(start="2023-01-01", periods=150, freq="5min")
    df = pd.DataFrame({
        "Open": np.random.uniform(100, 110, 150),
        "High": np.random.uniform(110, 120, 150),
        "Low": np.random.uniform(90, 100, 150),
        "Close": np.random.uniform(100, 110, 150),
        "Volume": np.random.uniform(1000, 5000, 150)
    }, index=dates)
    return df

def test_lorentzian_prepare_features(sample_df):
    lc = LorentzianClassifier()
    features = lc.prepare_features(sample_df)
    assert features is not None
    assert "rsi" in features.columns
    assert "wt" in features.columns
    assert "vol" in features.columns

def test_lorentzian_classify(sample_df):
    lc = LorentzianClassifier(k=3, lookback=100)
    result = lc.classify_series(sample_df, window=10)
    assert len(result) > 0
    assert "signal" in result[-1]
    assert "score" in result[-1]
    assert result[-1]["signal"] in [-1, 0, 1]

def test_adaptive_supertrend(sample_df):
    ast = AdaptiveSuperTrend()
    result = ast.calculate(sample_df, params={"atr_period": 5})
    assert result is not None
    assert "value" in result
    assert "regime" in result
    assert len(result["value"]) > 0

def test_lorentzian_signal_logic(sample_df):
    """Verify Lorentzian only signals when filters (EMA, ST, ADX) align."""
    lc = LorentzianClassifier(k=5, lookback=100)
    # Force a clear trend in the sample data for testing
    sample_df['Close'] = np.linspace(100, 150, 150) # Strong uptrend
    sample_df['High'] = sample_df['Close'] + 1
    sample_df['Low'] = sample_df['Close'] - 1
    
    result = lc.classify_series(sample_df, window=20)
    
    # In a strong manual uptrend, signals should be either 0 or 1, never -1
    signals = [r['signal'] for r in result]
    assert -1 not in signals, "Lorentzian signaled SHORT in a clear uptrend"

def test_adaptive_st_regimes(sample_df):
    """Verify Adaptive SuperTrend correctly identifies volatility regimes."""
    ast = AdaptiveSuperTrend()
    # Add a high-volatility spike
    sample_df.iloc[-10:, sample_df.columns.get_loc('High')] += 50
    sample_df.iloc[-10:, sample_df.columns.get_loc('Low')] -= 50
    
    result = ast.calculate(sample_df, window=50)
    regimes = result['regime']
    
    # Last 10 should likely be regime 3 (High volatility)
    assert 3 in regimes[-10:], "Failed to detect high volatility regime"

