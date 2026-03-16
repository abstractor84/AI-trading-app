import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch
from services.price_projector import PriceProjector, price_projector

@pytest.fixture
def sample_df():
    # Create 100 minutes of price data starting at 10:00 AM today
    now = datetime.now()
    market_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
    times = [market_start + timedelta(minutes=i) for i in range(100)]
    df = pd.DataFrame({
        "Open": np.linspace(100, 110, 100),
        "High": np.linspace(101, 111, 100),
        "Low": np.linspace(99, 109, 100),
        "Close": np.linspace(100.5, 110.5, 100),
        "Volume": [1000] * 100
    }, index=times)
    return df

def test_generate_projection_basic(sample_df):
    """Test basic projection generation."""
    # SKEPTIC: Ensure we are within market hours for projection
    with patch("services.price_projector.datetime") as mock_dt:
        mock_dt.now.return_value = datetime.now().replace(hour=11, minute=0)
        result = price_projector.generate_projection(sample_df, interval_minutes=1)
        
        assert "projection" in result
        assert "upper_band" in result
        assert "lower_band" in result
        assert len(result["projection"]) > 0
        assert len(result["projection"]) == len(result["timestamps"])
        assert result["current_price"] == pytest.approx(110.5)

def test_insufficient_data():
    """Test error handling for small dataframes."""
    df = pd.DataFrame({"Close": [100, 101]}, index=pd.to_datetime(["2023-01-01 09:15", "2023-01-01 09:16"]))
    result = price_projector.generate_projection(df)
    assert "error" in result
    assert result["error"] == "Insufficient data for projection"

def test_arima_fallback(sample_df):
    """Test projection when ARIMA is forced to fail or unavailable."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("services.price_projector.HAS_ARIMA", False)
        result = price_projector.generate_projection(sample_df)
        assert "ARIMA(2,1,2)" not in result["models_used"]
        assert "Fourier" in result["models_used"]

def test_fourier_logic(sample_df):
    """Test Fourier decomposition component directly."""
    prices = sample_df["Close"].values
    n_forecast = 30
    forecast = price_projector._fourier_projection(prices, n_forecast)
    
    assert len(forecast) == n_forecast
    assert isinstance(forecast, np.ndarray)
    # Check that it starts near the last price
    assert forecast[0] == pytest.approx(prices[-1], rel=0.05)

def test_gravity_logic(sample_df):
    """Test VWAP Gravity component."""
    prices = sample_df["Close"].values
    vwap = 105.0
    n_forecast = 30
    forecast = price_projector._gravity_projection(prices, n_forecast, vwap)
    
    assert len(forecast) == n_forecast
    # Gravity should pull toward 105.0 over time
    last_val = forecast[-1]
    # Current is 110.5, VWAP is 105.0. It should move toward 105.0.
    assert abs(last_val - 105.0) < abs(prices[-1] - 105.0)

def test_monte_carlo_bands(sample_df):
    """Test Monte Carlo simulation bands."""
    prices = sample_df["Close"].values
    n_forecast = 30
    center_line = np.linspace(110, 115, n_forecast)
    upper, lower = price_projector._monte_carlo_bands(prices, n_forecast, center_line)
    
    assert len(upper) == n_forecast
    assert len(lower) == n_forecast
    # Upper band should be above lower band
    assert np.all(upper >= lower)

def test_compute_vwap(sample_df):
    """Verify VWAP calculation."""
    val = price_projector._compute_vwap(sample_df)
    # Basic check: VWAP should be roughly average of Close prices for this steady uptrend
    assert 100 < val < 112
    
    # Empty volume fallback
    df_no_vol = sample_df.copy()
    df_no_vol["Volume"] = 0
    val2 = price_projector._compute_vwap(df_no_vol)
    assert val2 == pytest.approx(sample_df["Close"].iloc[-1])

def test_fallback_yesterday_data(sample_df):
    """Test fallback to tail(100) when no data for today is found."""
    # Set dates to yesterday
    yesterday = datetime.now() - timedelta(days=1)
    sample_df.index = [yesterday - timedelta(minutes=i) for i in range(100)]
    
    result = price_projector.generate_projection(sample_df)
    assert "projection" in result
    assert result["current_price"] == pytest.approx(sample_df["Close"].iloc[-1])

def test_n_forecast_fallback(sample_df):
    """Test forecast window when time is after market close."""
    # SKEPTIC: Update sample_df to end at 4 PM
    now = datetime.now()
    market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    sample_df.index = [market_end - timedelta(minutes=100-i) for i in range(100)]
    
    # Mandate forbids ghost data after 15:30. Should be 0.
    result = price_projector.generate_projection(sample_df)
    assert len(result["projection"]) == 0
    assert "None (After Market Close)" in result["models_used"] 

def test_explicit_n_forecast(sample_df):
    """Test that explicit n_forecast is respected."""
    result = price_projector.generate_projection(sample_df, n_forecast=50)
    assert len(result["projection"]) == 50

def test_arima_exception(sample_df):
    """Test ARIMA exception handling."""
    with patch("services.price_projector.ARIMA") as mock_arima:
        mock_arima.side_effect = Exception("ARIMA Failed")
        result = price_projector._arima_projection(sample_df["Close"].values, 10)
        assert result is None

def test_vwap_exception():
    """Test VWAP calculation exception."""
    df = pd.DataFrame({"Close": [100]}) # Missing cols
    val = price_projector._compute_vwap(df)
    assert val == 100.0

