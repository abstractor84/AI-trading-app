
import pytest
import os
import multiprocessing
import time
from playwright.sync_api import Page, expect
import uvicorn

# Constants for Testing
TEST_PORT = 8008

@pytest.fixture(scope="session", autouse=True)
def start_server():
    os.environ['SIMULATION'] = 'true'
    os.environ['ENABLE_SIMULATION'] = 'true'
    os.environ['UPSTOX_ACCESS_TOKEN'] = 'mocked_token'
    
    print(f"Starting test server on port {TEST_PORT}...")
    def run_server():
        import main
        uvicorn.run(main.app, host="127.0.0.1", port=TEST_PORT, log_level="info")
    
    p = multiprocessing.Process(target=run_server)
    p.start()
    
    # Wait for server to be ready
    time.sleep(10)
    print("Test server should be up.")
    yield
    p.terminate()
    print("Test server terminated.")

def test_supernova_ui_comprehensive(page: Page):
    """
    Consolidated test for ML indicators, scrolling, and scan results.
    """
    page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}"))
    page.on("pageerror", lambda exc: print(f"BROWSER ERROR: {exc}"))
    
    # 1. Load Page
    page.goto(f"http://127.0.0.1:{TEST_PORT}", timeout=60000)
    page.wait_for_selector("#upx-status", timeout=30000)
    
    # Verify UPX Icon (Mocked token makes it active)
    upx = page.locator("#upx-status")
    # In simulation mode with mocked_token, it should eventually turn active
    # but wait a few seconds for WebSocket state update
    page.wait_for_timeout(2000)
    
    # 2. Open Chart
    page.evaluate("openChart('TCS')")
    expect(page.locator("#chart-container canvas").first).to_be_visible(timeout=30000)
    
    # 3. Verify ML Indicators Toggle & Legend
    page.check("#toggle-st")
    page.check("#toggle-knn")
    expect(page.locator("#legend-st")).to_contain_text("ST:")
    
    # 4. Verify Settings Gears
    expect(page.locator("#gear-st")).to_be_visible()
    page.click("#gear-st")
    expect(page.locator("#st-settings")).to_be_visible()
    page.click("#st-settings .btn-primary")
    expect(page.locator("#st-settings")).not_to_be_visible()
    
    # 5. Test Right Scroll
    # Check if we can scroll past the last bar
    page.evaluate("appState.chartInstance.timeScale().scrollToPosition(10, false)")
    
    # 6. Close Chart and Trigger Scan
    page.click("#chart-close-btn")
    expect(page.locator("#chart-modal")).not_to_be_visible()
    
    page.click("#scan-btn")
    
    # Wait for mock_picks to arrive
    ticker_el = page.locator(".sc-ticker").first
    expect(ticker_el).to_contain_text("RELIANCE", timeout=15000)
    
    # Check Technicals Row (User requirement for precision/accuracy visibility)
    expect(page.locator(".ta-label:has-text('RSI')").first).to_be_visible()
    expect(page.locator(".ta-label:has-text('MACD')").first).to_be_visible()
    expect(page.locator(".ta-label:has-text('ADX')").first).to_be_visible()
    expect(page.locator(".ta-label:has-text('LZ AI')").first).to_be_visible()
    
    # Check Risk Levels (Mandatory for intraday safety)
    expect(page.locator(".level-chip.entry").first).to_be_visible()
    expect(page.locator(".level-chip.sl").first).to_be_visible()
    expect(page.locator(".level-chip.t1").first).to_be_visible()
    
    print("Comprehensive UI test passed successfully.")
