import pytest
import asyncio
import os
from playwright.sync_api import Page, expect

@pytest.fixture(scope="module")
def server():
    # SKEPTIC: Ensure simulation is ON for deterministic data
    os.environ["SIMULATION"] = "true"
    os.environ["ENABLE_SIMULATION"] = "true"
    os.environ["UPSTOX_ACCESS_TOKEN"] = "mocked_token"
    import subprocess
    import time
    # Kill any existing process on 8000
    subprocess.run(["fuser", "-k", "8000/tcp"], capture_output=True)
    time.sleep(1)
    proc = subprocess.Popen(["./venv/bin/python3", "main.py"], env=os.environ)
    time.sleep(10) # Sufficient wait for startup
    yield
    proc.terminate()
    proc.wait()

def test_knn_label_and_legend_visibility(page: Page, server):
    """Verify KNN labels (K-BUY/SELL) and legend text are visible."""
    # Capture console logs
    page.on("console", lambda msg: print(f"BROWSER: {msg.text}"))
    page.on("pageerror", lambda exc: print(f"BROWSER ERROR: {exc}"))
    
    page.goto("http://localhost:8000")
    # Wait for WebSocket
    expect(page.locator("#ws-status")).to_have_text("Live", timeout=15000)
    
    # Open chart directly via JS
    page.evaluate("openChart('RELIANCE.NS')")
    
    # 1. Check Legend (Wait for WebSocket refresh)
    # SKEPTIC: KNN: -- updates to KNN: BUY or KNN: SELL or KNN: NEUTRAL
    # We wait for the text to NOT be "KNN: --"
    page.wait_for_function("document.getElementById('legend-knn').textContent !== 'KNN: --'", timeout=20000)
    expect(page.locator("#legend-knn")).to_contain_text("KNN:", timeout=5000)
    
    # 3. Verify Marker Distribution (SKEPTIC: Ensure signals are not just at the start)
    # The chart has 1000 bars in simulation. window=1000.
    # We check if there are markers in the last 200 bars.
    recent_marker_count = page.evaluate("""() => {
        const data = appState.currentChartData.ml_knn;
        const total = data.length;
        const recent = data.slice(total - 200);
        return recent.filter(p => p.marker !== 0).length;
    }""")
    assert recent_marker_count > 0, f"Should have at least one KNN marker in recent 200 bars, found {recent_marker_count}"
    
    # 4. Toggle KNN and verify it still renders
    page.uncheck("#toggle-knn")
    page.check("#toggle-knn")
    expect(page.locator("#legend-knn")).to_be_visible()

def test_crosshair_sync_and_tooltip(page: Page, server):
    """Verify crosshair sync and time tooltip visibility on both charts."""
    page.goto("http://localhost:8000")
    page.wait_for_timeout(3000)
    page.evaluate("openChart('RELIANCE.NS')")
    
    # Wait for charts to initialize
    page.wait_for_selector("#chart-container canvas", timeout=15000)
    
    # Verify ADX timeScale is configured to show tooltips
    is_time_visible = page.evaluate("() => appState.adxChart && appState.adxChart.options().timeScale.visible")
    assert is_time_visible == True, "ADX timeScale must be visible for tooltip"
    
    # Verify crosshair mode is Magnet (1)
    ch_mode = page.evaluate("() => appState.chartInstance.options().crosshair.mode")
    assert ch_mode == 1, "Crosshair mode should be Magnet (1)"

def test_y_axis_declutter(page: Page, server):
    """Verify only LTP label is visible on Y-axis."""
    page.goto("http://localhost:8000")
    page.wait_for_timeout(3000)
    page.evaluate("openChart('RELIANCE.NS')")
    
    page.wait_for_selector("#chart-container canvas", timeout=15000)
    
    # Ensure auxiliary series have labels hidden
    is_st_hidden = page.evaluate("() => appState.series.stUp.options().lastValueVisible === false")
    is_knn_hidden = page.evaluate("() => appState.series.knnUp.options().lastValueVisible === false")
    
    assert is_st_hidden, "ST Y-axis label should be hidden"
    assert is_knn_hidden, "KNN Y-axis label should be hidden"

def test_settings_realtime_impact(page: Page, server):
    """Verify changing settings triggers refresh and updates UI."""
    page.goto("http://localhost:8000")
    page.wait_for_timeout(3000)
    page.evaluate("openChart('RELIANCE.NS')")
    
    # Wait for settings to be loadable
    page.wait_for_selector("#gear-knn", timeout=10000)
    page.click("#gear-knn")
    
    # Fill settings
    page.fill("#knn-window", "300")
    # Click Apply in the specific modal
    page.click("#knn-settings button.primary")
    
    # Verify toast
    expect(page.locator("#toast-container")).to_contain_text("KNN settings applied", timeout=15000)
    
    # Verify state update
    val = page.evaluate("appState.indicatorSettings.knn.window")
    assert val == 300
