#!/usr/bin/env python3
"""
Test live ticks on chart - verify real-time updates work.
"""
from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    # Capture console logs
    console_logs = []
    page.on('console', lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))
    
    # Navigate to the app
    page.goto('http://localhost:8000/')
    page.wait_for_load_state('networkidle')
    print("✓ Page loaded")
    
    # Wait a bit for initial data
    time.sleep(2)
    
    # Use the global search to find a stock
    search_input = page.locator('#global-chart-input')
    search_btn = page.locator('#global-chart-btn')
    
    if search_input.is_visible():
        print("✓ Search bar found")
        search_input.fill('RELIANCE')
        search_btn.click()
        time.sleep(2)
        
        # Check if chart modal appeared
        chart_modal = page.locator('#chart-modal')
        if chart_modal.is_visible():
            print("✓ Chart modal opened")
            
            # Click 1m interval button
            page.locator('.tf-btn[data-interval="1m"]').click()
            print("✓ Clicked 1m interval")
            time.sleep(3)
            
            # Take screenshot of chart
            page.screenshot(path='/tmp/chart_1m.png', full_page=True)
            print("✓ Screenshot saved to /tmp/chart_1m.png")
            
            # Wait and take another to see if ticks update
            time.sleep(5)
            page.screenshot(path='/tmp/chart_1m_2.png', full_page=True)
            print("✓ Second screenshot saved to /tmp/chart_1m_2.png")
            
            # Check for any errors in console
            errors = [l for l in console_logs if 'error' in l.lower()]
            if errors:
                print(f"Console errors: {errors[:5]}")
            else:
                print("✓ No console errors")
        else:
            print("Chart modal did not open")
    else:
        print("Search bar not found")
        
    browser.close()
    print("Test complete")