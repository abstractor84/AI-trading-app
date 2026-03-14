import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.advanced_indicators import classifier, adaptive_st

@pytest.fixture
def sample_df():
    """Generate 100 rows of dummy OHLCV data."""
    dates = [datetime(2024, 1, 1, 9, 15) + timedelta(minutes=5*i) for i in range(100)]
    df = pd.DataFrame({
        "Open": np.linspace(100, 110, 100) + np.random.randn(100),
        "High": np.linspace(101, 111, 100) + np.random.randn(100),
        "Low": np.linspace(99, 109, 100) + np.random.randn(100),
        "Close": np.linspace(100.5, 110.5, 100) + np.random.randn(100),
        "Volume": np.random.randint(1000, 5000, 100)
    }, index=dates)
    return df

def test_lorentzian_adaptive_lookback(sample_df):
    """Verify LZ works on short datasets (100 rows) with adaptive lookback."""
    # Default lookback is 2000, but we only have 100 rows.
    # The adaptive logic should handle this.
    res = classifier.classify_series(sample_df, window=50)
    assert len(res) > 0
    assert "signal" in res[0]
    assert "time" in res[0]

def test_adaptive_supertrend_basic(sample_df):
    """Verify SuperTrend works on short datasets."""
    res = adaptive_st.calculate(sample_df, window=50)
    assert res is not None
    assert "value" in res
    assert len(res["value"]) <= 50
    assert len(res["trend"]) == len(res["value"])

def test_lorentzian_classification_direct(sample_df):
    """Verify direct classification (single bar) works."""
    res = classifier.classify(sample_df)
    assert "signal" in res
    assert "score" in res
    assert res["signal"] in ["BUY", "SHORT SELL", "NEUTRAL"]
