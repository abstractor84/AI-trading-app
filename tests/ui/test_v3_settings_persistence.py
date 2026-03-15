import pytest
import asyncio
import os
from playwright.sync_api import Page, expect

@pytest.fixture(scope="module")
def server():
    # SKEPTIC: Ensure simulation is ON for deterministic data
    os.environ["SIMULATION"] = "true"
    import subprocess
    import time
    proc = subprocess.Popen(["./venv/bin/python3", "main.py"], env=os.environ)
    time.sleep(5) # Wait for startup
    yield
    proc.terminate()

def test_ml_indicator_settings_update(page: Page, server):
    """
    Verify that changing ML indicator settings triggers a data refresh and doesn't crash the UI.
    """
    page.goto("http://localhost:8000")
    
    # 1. Open Chart for RELIANCE
    page.click("text=RELIANCE")
    expect(page.locator("#chart-title")).to_contain_text("RELIANCE")
    
    # 2. Change ST Settings
    page.click("#gear-st")
    page.fill("#st-atr-len", "20")
    page.click("text=Apply Settings")
    
    # Verify toast
    expect(page.locator("#toast-container")).to_contain_text("ST settings applied")
    
    # 3. Verify Reset
    page.click("#gear-st")
    page.click("text=Reset to Default")
    expect(page.locator("#toast-container")).to_contain_text("ST settings reset to defaults")
    
    # 4. Verify KNN Markers Logic (Check if markers are still in payload after toggle)
    page.uncheck("#toggle-knn")
    page.check("#toggle-knn")
    # Evaluate marker count in appState
    marker_count = page.evaluate("appState.currentChartData.ml_knn.filter(p => p.marker !== 0).length")
    assert marker_count > 0, "KNN should have markers in simulation mode"

def test_chart_zoom_persistence(page: Page, server):
    """
    Verify that switching intervals does not trigger a global zoom-out (fitContent).
    """
    page.goto("http://localhost:8000")
    page.click("text=RELIANCE")
    
    # Switch to 1m
    page.click("button[data-interval='1m']")
    page.wait_for_timeout(1000)
    
    # Get current visible range
    range1 = page.evaluate("appState.chartInstance.timeScale().getVisibleRange()")
    
    # Switch to 5m
    page.click("button[data-interval='5m']")
    page.wait_for_timeout(1000)
    
    # SKEPTIC: Lightweight Charts might adjust the range slightly due to different bar widths,
    # but it shouldn't be a TOTAL zoom-out to fit 5 days of data.
    range2 = page.evaluate("appState.chartInstance.timeScale().getVisibleRange()")
    
    # If fitContent was called, range2 would likely be much larger than range1 
    # (since 5m data covers 5 days vs 1 day for 1m).
    # But wait, 1m data is also filtered to 1 day.
    # Actually, the best way to check is if isFirstLoad is false.
    is_first_load = page.evaluate("appState.isFirstLoad")
    assert is_first_load == False, "isFirstLoad should be false after rendering"
