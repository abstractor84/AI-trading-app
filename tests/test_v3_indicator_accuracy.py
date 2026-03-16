import pytest
import pandas as pd
import numpy as np
from services.advanced_indicators import classifier, adaptive_st, knn_forecaster

def make_trending_data(trend="up", length=200):
    """Generate mock data with a clear trend for indicator testing."""
    dates = pd.date_range(start="2023-01-01", periods=length, freq="5min")
    noise = np.random.normal(0, 0.2, length) 
    if trend == "up":
        price = np.linspace(100, 160, length) + noise 
    else:
        price = np.linspace(160, 100, length) + noise
    
    df = pd.DataFrame({
        "Open": (price - 0.5).astype(float),
        "High": (price + 1.0).astype(float),
        "Low": (price - 1.0).astype(float),
        "Close": price.astype(float),
        "Volume": (1000.0 + np.random.randint(-100, 100, length)).astype(float)
    }, index=dates)
    return df

def test_lorentzian_accuracy():
    """Verify Lorentzian signals are present and logical."""
    # Strong uptrend with some noise
    df_up = make_trending_data("up", length=1200)
    signals = classifier.classify_series(df_up, window=400, params={"threshold": 0.2})
    
    assert len(signals) == 400
    buy_signals = [s for s in signals if s['signal'] == 1]
    # We expect at least one signal in a strong 400-bar window
    assert len(buy_signals) >= 1

def test_knn_accuracy():
    """Verify KNN signals correlate with price movement."""
    df_up = make_trending_data("up", length=800)
    shading = knn_forecaster.get_historical_shading(df_up, window=400)
    
    assert len(shading) == 400
    trends = [s['trend'] for s in shading]
    up_ratio = trends.count(1) / len(trends)
    assert up_ratio > 0.6 

def test_knn_reversal_response():
    """Verify KNN detects trend reversals when trained on mixed data."""
    # Data with multiple cycles to ensure training data (first 200 bars) has both UP and DOWN
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="5min")
    # Cycle every 150 bars
    x = np.linspace(0, 4 * np.pi, 1000)
    price = 100 + 20 * np.sin(x) + np.random.normal(0, 0.2, 1000)
    
    df = pd.DataFrame({
        "Open": (price - 0.5).astype(float), 
        "High": (price + 1.0).astype(float), 
        "Low": (price - 1.0).astype(float), 
        "Close": price.astype(float), 
        "Volume": 1000.0
    }, index=dates)
    
    # window=800 means it trains on the first 200 bars.
    # 200 bars is > 150, so it sees one full swing and the start of another.
    shading = knn_forecaster.get_historical_shading(df, window=800)
    assert len(shading) == 800
    
    trends = [s['trend'] for s in shading]
    markers = [s for s in shading if s['marker'] != 0]
    
    # In oscillatory data, we expect multiple markers
    assert len(markers) >= 2, f"Should have multiple markers in oscillatory data, got {len(markers)}"
    
    # Verify we have both types of trends
    assert trends.count(1) > 50
    assert trends.count(-1) > 50

def test_st_regime_mapping():
    """Verify SuperTrend regimes adapt to volatility."""
    df = make_trending_data("up", length=400)
    # Add a high volatility spike (ensuring float ops)
    df.iloc[200:220] = df.iloc[200:220] * 1.5 
    
    res = adaptive_st.calculate(df, window=200)
    assert res is not None
    assert "regime" in res
    assert len(set(res['regime'])) > 1 

def test_knn_recent_marker_presence():
    """SKEPTIC: Ensure KNN doesn't just show markers at the start. Verified in last 20% of window."""
    # Create 1000 bars of data with multiple trend changes
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="5min")
    x = np.linspace(0, 10 * np.pi, 1000)
    price = 100 + 10 * np.sin(x) + np.random.normal(0, 0.1, 1000)
    
    df = pd.DataFrame({
        "Open": price - 0.5, "High": price + 1.0, "Low": price - 1.0, "Close": price, "Volume": 1000
    }, index=dates).astype(float)
    
    # window=800
    shading = knn_forecaster.get_historical_shading(df, window=800)
    assert len(shading) == 800
    
    # Check last 160 bars (20% of 800)
    recent = shading[-160:]
    markers = [s['marker'] for s in recent if s['marker'] != 0]
    
    assert len(markers) >= 1, "Should have at least one KNN marker in the most recent 20% of the chart"
