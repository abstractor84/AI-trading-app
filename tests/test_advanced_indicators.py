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

def test_knn_forecaster(sample_df):
    knn = KNNTrendForecaster(k=2, sequence_length=5)
    forecast = knn.forecast(sample_df)
    assert forecast is not None
    assert len(forecast) == 10

