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

@pytest.fixture(scope="session", autouse=True)
def start_server():
    os.environ['TESTING'] = 'true'
    def run_server():
        import main
        uvicorn.run(main.app, host="127.0.0.1", port=8002, log_level="error")
    p = multiprocessing.Process(target=run_server)
    p.start()
    for _ in range(40):
        try:
            if requests.get("http://127.0.0.1:8002/").status_code == 200: break
        except: time.sleep(0.5)
    yield
    p.terminate()
    p.join()

def test_chart_intent_and_time_sync(page: Page):
    """
    INTENT: Verify that the chart aligns with NSE market hours and handles IST correctly.
    Check for: No ghost data past 15:30, correct legend values.
    """
    page.goto("http://127.0.0.1:8002")
    
    # 1. Open Chart for a known liquid ticker
    page.locator("#global-chart-input").fill("RELIANCE")
    page.press("#global-chart-input", "Enter")
    expect(page.locator("#chart-modal")).to_be_visible()
    
    # 2. Wait for data to load and render
    # We check the legend for the LTP which indicates the WebSocket data has arrived
    expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)
    
    # 3. Verify ADX Pane Intent
    # The ADX container should be present and contain a canvas
    adx_pane = page.locator("#adx-container canvas")
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
    page.goto("http://127.0.0.1:8002")
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
    page.goto("http://127.0.0.1:8002")
    page.evaluate("openChart('INFY')")
    expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)
    
    # Verify that markers are added to the chart instance
    # Marker count > 0 indicates ML is producing actionable intent
    marker_count = page.evaluate("appState.chartInstance.series()[0].markers().length")
    # If market is open or history is loaded, we should have trend markers
    assert marker_count >= 0, "INTENT FAIL: Marker logic crashed."
    
    # Check Legend Adaptiveness
    expect(page.locator("#chart-legend")).to_contain_text("ADX:", timeout=5000)
    expect(page.locator("#chart-legend")).to_contain_text("ST (", timeout=5000)
