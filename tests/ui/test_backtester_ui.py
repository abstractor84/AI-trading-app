import pytest
import time
import multiprocessing
import uvicorn
import requests
import os
from playwright.sync_api import Page, expect

TEST_PORT = 8000

@pytest.fixture(scope="session", autouse=True)
def start_server():
    os.environ['SIMULATION'] = 'true'
    def run_server():
        import main
        uvicorn.run(main.app, host="127.0.0.1", port=TEST_PORT, log_level="info")
    p = multiprocessing.Process(target=run_server)
    p.start()
    # Wait for startup
    for _ in range(40):
        try:
            if requests.get(f"http://127.0.0.1:{TEST_PORT}/").status_code == 200: break
        except: time.sleep(0.5)
    yield
    p.terminate()
    p.join()

def test_backtester_ui_renders_results(page: Page):
    page.goto(f"http://127.0.0.1:{TEST_PORT}")
    
    # Switch to Backtester tab
    page.click("#nav-backtest-btn")
    expect(page.locator("#page-backtest")).to_be_visible()
    
    # Fill config
    page.fill("#bt-ticker", "RELIANCE.NS")
    page.select_option("#bt-period", "30d")
    
    # Run backtest
    page.click("#run-backtest-btn")
    
    # Wait for results
    # Backend sends 'backtest_results' which triggers handleBacktestResults
    # Results container should show metrics
    expect(page.locator(".bt-summary-card")).to_be_visible(timeout=60000)
    expect(page.locator(".bt-stat strong").first).not_to_have_text("0%", timeout=10000)
    
    # Detailed trade log should be visible
    expect(page.locator(".bt-log-wrapper table")).to_be_visible()
