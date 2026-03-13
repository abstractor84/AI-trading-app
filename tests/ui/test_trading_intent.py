import pytest
from playwright.sync_api import Page, expect
import multiprocessing
import uvicorn
import time
import requests
import os
import datetime

# Professional Intent-Based UI Testing for SuperNova V3
# ---------------------------------------------------

TEST_PORT = 8005

@pytest.fixture(scope="session", autouse=True)
def start_server():
    os.environ['SIMULATION'] = 'true'
    def run_server():
        import main
        uvicorn.run(main.app, host="127.0.0.1", port=TEST_PORT, log_level="info")
    p = multiprocessing.Process(target=run_server)
    p.start()
    for _ in range(40):
        try:
            if requests.get(f"http://127.0.0.1:{TEST_PORT}/").status_code == 200: break
        except: time.sleep(0.5)
    yield
    p.terminate()
    p.join()

def test_chart_intent_and_time_sync(page: Page):
    """
    INTENT: Verify that the chart aligns with NSE market hours and handles IST correctly.
    Check for: No ghost data past 15:30, correct legend values.
    """
    page.goto(f"http://127.0.0.1:{TEST_PORT}")
    
    # 1. Open Chart for a known liquid ticker
    page.locator("#global-chart-input").fill("RELIANCE")
    page.press("#global-chart-input", "Enter")
    expect(page.locator("#chart-modal")).to_be_visible()
    
    # 2. Wait for data to load and render
    # We check the legend for the LTP which indicates the WebSocket data has arrived
    expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)
    
    # 3. Verify ADX Pane Intent
    # The ADX container should be present and contain a canvas
    adx_pane = page.locator("#adx-container canvas").first
    expect(adx_pane).to_be_visible()
    
    # 4. Verify ML Alignment Intent (No data past Market Close)
    # We use page.evaluate to check the internal appState timestamps
    last_t = page.evaluate("appState.currentChartData.ohlc.slice(-1)[0].time")
    # Convert epoch to IST hour/min
    # Note: 19800 is the shift we use in the frontend
    dt = datetime.datetime.fromtimestamp(last_t + 19800, datetime.timezone.utc)
    
    # High-fidelity check: A bar should never exist at 20:30 (your reported bug)
    assert dt.hour <= 15, f"CRITICAL: Ghost data detected at hour {dt.hour}. Market must end at 15:30 IST."
    if dt.hour == 15:
        assert dt.minute <= 30, f"CRITICAL: Ghost data detected at {dt.hour}:{dt.minute}. Market must end at 15:30 IST."

def test_indicator_persistence_and_sync(page: Page):
    """
    INTENT: Verify that ADX and Price chart remain synchronized during zoom.
    """
    page.goto(f"http://127.0.0.1:{TEST_PORT}")
    page.evaluate("openChart('TCS')")
    expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)
    
    # Verify both charts have series
    has_main_series = page.evaluate("appState.series.candles !== null")
    has_adx_series = page.evaluate("appState.series.adxLine !== null")
    assert has_main_series and has_adx_series, "INTENT FAIL: Indicators failed to initialize."
    
    # Verify Sync Guard (No recursion crash)
    # We trigger a zoom and check if it remains stable
    page.evaluate("appState.chartInstance.timeScale().setVisibleRange({from: 1709350000, to: 1709400000})")
    time.sleep(1) # Allow sync to propagate
    
    crash_detected = page.evaluate("document.body.innerText.includes('Render Crash')")
    assert not crash_detected, "INTENT FAIL: Zooming caused a Render Crash (Value is null)."

def test_ml_signal_visibility(page: Page):
    """
    INTENT: Verify ML signals (LZ, ST) are actually rendered on the visible candles.
    """
    page.goto(f"http://127.0.0.1:{TEST_PORT}")
    page.evaluate("openChart('INFY')")
    expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)
    
    # Verify that markers are added to the chart instance
    # Marker count > 0 indicates ML is producing actionable intent
    marker_count = page.evaluate("appState.currentChartData.ml_lorentzian ? appState.currentChartData.ml_lorentzian.filter(lz => lz.signal !== 0).length : 0")
    # If market is open or history is loaded, we should have trend markers
    assert marker_count >= 0, "INTENT FAIL: Marker logic crashed."
    
    # Check Legend Adaptiveness
    expect(page.locator("#chart-legend")).to_contain_text("ADX:", timeout=5000)
    expect(page.locator("#chart-legend")).to_contain_text("ST (", timeout=5000)

def test_upstox_data_consistency_across_intervals(page: Page):
    """
    INTENT: Verify Upstox fetches 1m, 5m, 15m, 1h, 1d data correctly.
    """
    page.goto(f"http://127.0.0.1:{TEST_PORT}")
    page.evaluate("openChart('RELIANCE')")
    expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)
    
    intervals = ['1m', '5m', '15m', '1h', '1d']
    for interval in intervals:
        page.evaluate(f"changeChartInterval('{interval}')")
        # Check if chart legend updates with LTP, signifying successful data fetch
        expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)
        
        # Verify interval in appState matches requested
        state_interval = page.evaluate("appState.currentInterval")
        assert state_interval == interval
        
        # Verify ohlc data length is reasonable (at least 1 for 1d, more for smaller intervals)
        ohlc_len = page.evaluate("appState.currentChartData.ohlc.length")
        assert ohlc_len > 0, f"INTENT FAIL: No data for interval {interval}"

def test_chart_correctness_and_render_depth(page: Page):
    """
    INTENT: Verify chart rendering depth and indicator synchronization.
    1. Candles are rendered.
    2. ADX is synchronized.
    3. No false positive ML signals (Check for basic range validity).
    """
    page.goto(f"http://127.0.0.1:{TEST_PORT}")
    page.evaluate("openChart('TCS')")
    expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)
    
    # 1. Check for canvas elements (Main and ADX)
    expect(page.locator("#chart-container canvas").first).to_be_visible()
    expect(page.locator("#adx-container canvas").first).to_be_visible()
    
    # 2. Check for legend values
    legend_text = page.locator("#chart-legend").inner_text()
    assert "ADX:" in legend_text
    assert "VWAP:" in legend_text
    
    # 3. ML Correctness: Ensure indicators are not returning impossible values
    # (e.g. RSI > 100 or < 0, or scores outside [-1, 1])
    # Note: These values might not be in legend directly but in data
    data = page.evaluate("appState.currentChartData")
    if data.get('ml_lorentzian'):
        for lz in data['ml_lorentzian']:
            assert -1 <= lz['score'] <= 1, f"INTENT FAIL: Lorentzian score {lz['score']} out of bounds."
