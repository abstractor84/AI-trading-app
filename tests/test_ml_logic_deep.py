import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.advanced_indicators import classifier, adaptive_st

def generate_complex_data(rows=500, noise_level=0.1, fake_breakouts=True):
    """
    Generate complex price data with trend, noise, and fake breakouts.
    Used to stress test ML indicator accuracy and whipsaw resistance.
    """
    dates = [datetime(2024, 1, 1, 9, 15) + timedelta(minutes=5*i) for i in range(rows)]
    x = np.linspace(0, 10 * np.pi, rows)
    
    # Base trend: Sine wave
    base_price = 100 + 10 * np.sin(x)
    
    # Add Noise
    prices = base_price + np.random.randn(rows) * noise_level
    
    # Add Fake Breakouts / Dead Cat Bounces
    if fake_breakouts:
        # Near a trough (approx row 75), add a quick spike that fails
        prices[70:80] += 2.0
        # Near a peak (approx row 225), add a quick dip that recovers
        prices[220:230] -= 2.0

    df = pd.DataFrame({
        "Open": prices + np.random.randn(rows) * 0.05,
        "High": prices + 0.5,
        "Low": prices - 0.5,
        "Close": prices,
        "Volume": np.random.randint(1000, 5000, rows)
    }, index=dates)
    return df

def test_ml_signal_timing_and_accuracy():
    """
    Verify that BUY/SELL signals appear at logical trend pivot points.
    Checks timing accuracy: Signal should trigger near troughs/peaks.
    """
    df = generate_complex_data(rows=500, noise_level=0.05, fake_breakouts=False)
    
    # 1. Test Lorentzian Classifier Timing
    lz_results = classifier.classify_series(df, window=400)
    assert len(lz_results) > 0
    
    # Convert to series for easier analysis
    lz_df = pd.DataFrame(lz_results)
    buy_signals = lz_df[lz_df['signal'] == 1]
    sell_signals = lz_df[lz_df['signal'] == -1]
    
    # Expected Troughs: Sine wave sin(x) troughs at 1.5pi, 3.5pi, 5.5pi...
    # For linspace(0, 10pi, 500), 1.5pi is at index (1.5/10)*500 = 75
    # Peaks at 0.5pi, 2.5pi... 0.5pi is at index 25
    
    # Check if we have signals near expected pivots
    # (Allowing some delay due to ML windowing/confirmation)
    has_buy_near_trough = any(abs(idx - 75) < 20 for idx in buy_signals.index)
    has_sell_near_peak = any(abs(idx - 125) < 20 for idx in sell_signals.index) # 2.5pi = 125
    
    # In deterministic sine data, ML should be highly accurate
    assert has_buy_near_trough, "ML failed to detect trough-based BUY signal"
    assert has_sell_near_peak, "ML failed to detect peak-based SELL signal"

def test_ml_whipsaw_resistance():
    """
    Verify that ML indicators ignore frequent small direction changes (noise).
    Ensures 'fake breakouts' don't trigger frequent contradictory signals.
    """
    # High noise + fake breakouts
    df = generate_complex_data(rows=500, noise_level=0.5, fake_breakouts=True)
    
    # 1. Adaptive SuperTrend Stability
    st_res = adaptive_st.calculate(df, window=500)
    st_trend = np.array(st_res['trend'])
    
    # Count trend flips. Frequent flips (> 10 in 500 bars of sine wave) indicate poor noise resistance.
    flips = np.count_nonzero(st_trend[1:] != st_trend[:-1])
    # Expected pivots in 10pi range: 10 flips. If > 20, it's reacting to noise.
    assert flips < 20, f"SuperTrend too sensitive to noise: {flips} flips detected"

    # 2. Lorentzian Signal Consistency
    lz_res = classifier.classify_series(df, window=400)
    lz_signals = [r['signal'] for r in lz_res]
    
    # Check for 'rapid oscillation' (BUY then SELL within 5 bars)
    rapid_flips = 0
    for i in range(len(lz_signals)-5):
        segment = lz_signals[i:i+5]
        if 1 in segment and -1 in segment:
            rapid_flips += 1
            
    assert rapid_flips < 5, f"Lorentzian shows too many rapid flips: {rapid_flips}"

def test_ml_indicators_combined_plotting_data():
    """
    Verify all indicators generate valid plotting data simultaneously.
    Ensures zero regressions in multi-indicator data payloads.
    """
    df = generate_complex_data(rows=200)
    
    lz = classifier.classify_series(df, window=100)
    st = adaptive_st.calculate(df, window=100)
    
    assert len(lz) == 100
    assert len(st['value']) == 100
    assert len(st['trend']) == 100
    
    # Verify alignment
    for i in range(len(lz)):
        # LZ time is unix, df index is datetime
        lz_time = lz[i]['time']
        expected_time = int(df.index[-(100-i)].timestamp())
        assert abs(lz_time - expected_time) < 2, f"Time mismatch at index {i}"
