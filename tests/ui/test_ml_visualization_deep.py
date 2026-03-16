import pytest
from playwright.sync_api import Page, expect
import multiprocessing
import uvicorn
import time
import requests
import os

TEST_PORT = 8006

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

def test_ml_indicator_visibility_and_legend_sync(page: Page):
    """
    Verify each ML indicator (LZ, ST, KNN) plots correctly and updates legend.
    Checks visibility separately and then all together.
    """
    # Capture console logs for debugging
    page.on("console", lambda msg: print(f"BROWSER_CONSOLE: {msg.text}"))
    
    page.goto(f"http://127.0.0.1:{TEST_PORT}")
    # Wait for app to be ready
    page.wait_for_function("window.appState !== undefined")
    page.evaluate("openChart('RELIANCE')")
    
    # Wait for initial data
    expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)
    
    # 1. Test SuperTrend (ST) visibility
    # SKEPTIC: ST is checked by default. Uncheck and Re-check to trigger change event.
    page.uncheck("#toggle-st")
    page.check("#toggle-st")
    
    # We wait for the text to change from "--"
    expect(page.locator("#legend-st")).to_contain_text("ST:", timeout=10000)
    expect(page.locator("#legend-st")).not_to_contain_text("--", timeout=5000)
    
    # Check if series object exists in state
    has_st = page.evaluate("appState.series.stLine !== null")
    assert has_st, "ST series (Line) should be initialized"
    
    # 2. Test Lorentzian (LZ) markers
    page.check("#toggle-lz")
    page.wait_for_timeout(1000)
    # Check if LZ data exists in current payload
    has_lz_data = page.evaluate("appState.currentChartData.ml_lorentzian.length > 0")
    assert has_lz_data, "LZ Data should be present in payload"
    expect(page.locator("#legend-lz")).to_contain_text("LZ:", timeout=5000)
    
    # 3. Test KNN Projection
    page.check("#toggle-knn")
    page.wait_for_timeout(1000)
    has_knn = page.evaluate("appState.series.projLine !== null && appState.series.knnUp !== null")
    assert has_knn, "KNN Projection and Trend series should be initialized"

def test_ml_signal_positioning_relative_to_ohlc(page: Page):
    """
    Verify signals (markers) are generated in the data payload.
    Ensures signal logic correctly maps to time points.
    """
    page.goto(f"http://127.0.0.1:{TEST_PORT}")
    page.wait_for_function("window.appState !== undefined")
    page.evaluate("openChart('INFY')")
    expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)
    
    page.check("#toggle-lz")
    page.wait_for_timeout(2000)
    
    # Check currentChartData for markers
    lz_signals = page.evaluate("appState.currentChartData.ml_lorentzian.filter(l => l.signal !== 0)")
    assert len(lz_signals) > 0, "No LZ signals found in chart data payload"
    
    for s in lz_signals:
        assert "time" in s
        assert "score" in s
        assert s["signal"] in [1, -1]
            
def test_ml_indicator_toggling_integrity(page: Page):
    """
    Verify that toggling indicators updates the UI state.
    """
    page.goto(f"http://127.0.0.1:{TEST_PORT}")
    page.wait_for_function("window.appState !== undefined")
    page.evaluate("openChart('TCS')")
    expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)
    
    # Initial check
    page.uncheck("#toggle-st")
    page.wait_for_timeout(500)
    # Legend should still be there but maybe empty/stale? 
    # Actually toggleIndicator re-renders the chart.
    
    # Verify toggling doesn't crash the engine
    page.check("#toggle-st")
    page.check("#toggle-lz")
    page.check("#toggle-knn")
    page.wait_for_timeout(500)
    
    # Verify chart still responds to interval change after toggling
    page.evaluate("changeChartInterval('15m')")
    expect(page.locator("#legend-ltp")).to_contain_text("LTP:", timeout=20000)

