import pytest
import asyncio
import os
from playwright.sync_api import Page, expect

TEST_PORT = 8011

@pytest.fixture(scope="session", autouse=True)
def start_server():
    os.environ['SIMULATION'] = 'true'
    os.environ['ENABLE_SIMULATION'] = 'true'
    os.environ['UPSTOX_ACCESS_TOKEN'] = 'mocked_token'

    def run_server():
        import main
        import uvicorn
        uvicorn.run(main.app, host="127.0.0.1", port=TEST_PORT, log_level="error")
    
    import multiprocessing
    import time
    p = multiprocessing.Process(target=run_server)
    p.start()
    time.sleep(5)
    
    # Mock some data in DB
    from database import SessionLocal
    from models import Trade
    from datetime import datetime, timedelta
    
    with SessionLocal() as db:
        # Delete if exists to avoid unique constraint error
        db.query(Trade).filter_by(id="MOCK-ID-123").delete()
        # Create a mock closed trade
        t = Trade(
            id="MOCK-ID-123",
            ticker="MOCK_HIST.NS",
            action="BUY",
            quantity=10,
            entry_price=1234.56,
            exit_price=1300.00,
            pnl=654.40,
            status="CLOSED",
            timestamp=datetime.utcnow() - timedelta(days=2),
            close_time=datetime.utcnow() - timedelta(days=1),
            stop_loss=1200.0,
            target_1=1300.0,
            target_2=1350.0,
            trailing_sl=1250.0
        )
        db.add(t)
        db.commit()

    yield
    p.terminate()
    p.join()


def test_trade_history_rendering_and_formatting(page: Page):
    """Verify that 90-day trade history renders with correct formatting and clickable tickers."""
    page.goto(f"http://127.0.0.1:{TEST_PORT}")
    
    # Navigate to history page
    page.click("#nav-history-btn")
    
    # Wait for the table to populate (ws handler returns data)
    page.wait_for_selector(".history-row", timeout=10000)
    
    # Check if the mock trade is present by finding the row containing MOCK_HIST
    row = page.locator(".history-row:has-text('MOCK_HIST')").first
    expect(row).to_be_visible()
    
    # Validate formatting (Indian Rupee formatting with commas, e.g., 1,234.56)
    # The 'fmt' function is used in JS, so it should output "1,234.56"
    text_content = row.inner_text()
    assert "1,234.56" in text_content, f"Entry price should be properly formatted in '{text_content}'"
    assert "1,300.00" in text_content, f"Exit price should be properly formatted in '{text_content}'"
    assert "654.40" in text_content, f"PnL should be properly formatted in '{text_content}'"
    
    # Verify ticker is clickable and has correct class
    ticker_span = row.locator(".hist-ticker")
    expect(ticker_span).to_have_text("MOCK_HIST")
    
    # Click ticker to open chart
    ticker_span.click()
    
    # Chart should open
    expect(page.locator("#chart-title")).to_contain_text("MOCK_HIST", timeout=5000)
