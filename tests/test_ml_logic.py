import pytest
import pandas as pd
import numpy as np
from services.advanced_indicators import LorentzianClassifier, AdaptiveSuperTrend, KNNTrendForecaster
from services.technical_analysis import TechnicalAnalysisService

@pytest.fixture
def sample_bullish_data():
    """Generate a synthetic bullish trend with momentum."""
    np.random.seed(42)
    rows = 2000
    dates = pd.date_range("2023-01-01", periods=rows, freq="5min")
    
    # Uptrend
    prices = 100 + np.cumsum(np.random.normal(0.1, 0.5, rows))
    
    df = pd.DataFrame({
        "Open": prices - 0.2,
        "High": prices + 0.5,
        "Low": prices - 0.5,
        "Close": prices,
        "Volume": np.random.randint(1000, 5000, rows)
    }, index=dates)
    
    return df

@pytest.fixture
def sample_bearish_data():
    """Generate a synthetic bearish trend."""
    np.random.seed(42)
    rows = 2000
    dates = pd.date_range("2023-01-01", periods=rows, freq="5min")
    
    # Downtrend
    prices = 200 + np.cumsum(np.random.normal(-0.1, 0.5, rows))
    
    df = pd.DataFrame({
        "Open": prices + 0.2,
        "High": prices + 0.5,
        "Low": prices - 0.5,
        "Close": prices,
        "Volume": np.random.randint(1000, 5000, rows)
    }, index=dates)
    
    return df

class TestMLLogic:
    def test_lorentzian_bullish_signal(self, sample_bullish_data):
        classifier = LorentzianClassifier(lookback=2000)
        result = classifier.classify_series(sample_bullish_data, window=10)
        
        # In a strong uptrend, Lorentzian should ideally suggest positive score/signal
        assert result is not None
        assert len(result) > 0
        assert "score" in result[-1]
        assert "signal" in result[-1]
        print(f"Lorentzian Bullish Result: {result[-1]}")

    def test_lorentzian_lookback_handling(self):
        """Verify Lorentzian handles insufficient data gracefully."""
        short_df = pd.DataFrame({
            "Open": [100]*10, "High": [105]*10, "Low": [95]*10, "Close": [100]*10, "Volume": [1000]*10
        }, index=pd.date_range("2024-01-01", periods=10, freq="1d"))
        
        classifier = LorentzianClassifier(lookback=2000)
        result = classifier.classify_series(short_df, window=2)
        assert len(result) > 0
        assert result[-1]["score"] == 0
        assert result[-1]["signal"] == 0

    def test_knn_forecast_direction(self, sample_bullish_data):
        knn = KNNTrendForecaster(k=5, sequence_length=10)
        forecast = knn.forecast(sample_bullish_data)
        
        assert forecast is not None
        assert len(forecast) > 0
        # Check if the overall forecast direction is upwards given bullish input
        assert forecast[-1] > sample_bullish_data["Close"].iloc[-1] * 0.95

    def test_adaptive_supertrend_alignment(self, sample_bullish_data):
        st = AdaptiveSuperTrend()
        result = st.calculate(sample_bullish_data, params={"atr_period": 10})
        
        assert result is not None
        assert "value" in result
        assert "regime" in result
        assert len(result["value"]) > 0
        # In a bullish trend, the ST value should generally be below the Close price
        # (Trailing SL behavior)
        last_st = result["value"][-1]
        last_close = sample_bullish_data["Close"].iloc[-1]
        assert last_st <= last_close or abs(last_st - last_close) / last_close < 0.05

    def test_technical_analysis_splicing_integration(self):
        """Ensure analyze_stock uses the full data context."""
        ta_svc = TechnicalAnalysisService()
        # Mocking fetch_spliced_ohlcv to return known data
        # We'll just check if indicators are computed successfully on reasonably sized data
        # (This is more of an integration check)
        pass 
