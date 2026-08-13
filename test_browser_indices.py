#!/usr/bin/env python3
"""
Browser automation test to verify Indian indices are displaying correctly in the UI.

This script:
1. Opens the dashboard in a browser
2. Checks if all 7 Indian indices are displayed
3. Verifies there are no "Waiting for..." messages
4. Takes screenshots for verification
"""

import sys
import time
from playwright.sync_api import sync_playwright

# Server is already running on port 8000
BASE_URL = "http://localhost:8000"

def main():
    print("="*80)
    print("Browser Automation: Verify Indian Indices Display")
    print("="*80)
    
    with sync_playwright() as p:
        # Launch browser in headless mode
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        
        # Collect console logs
        console_logs = []
        page.on("console", lambda msg: console_logs.append(f"{msg.type}: {msg.text}"))
        
        # Navigate to the app
        print("\n[1] Navigating to dashboard...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        print(f"    Page title: {page.title()}")
        
        # Wait for the market pulse section to render
        time.sleep(3)
        
        # Take a screenshot of the full page
        page.screenshot(path="/tmp/indices_before.png", full_page=True)
        print("    Saved screenshot to /tmp/indices_before.png")
        
        # Check for Indian indices in the DOM
        print("\n[2] Checking Indian indices display...")
        
        # Get the market pulse section content
        india_pulse = page.locator("#market-pulse-india").inner_html() if page.locator("#market-pulse-india").count() > 0 else ""
        
        print(f"    Indian indices HTML (first 500 chars):")
        print(f"    {india_pulse[:500]}")
        
        # Expected indices
        expected_indices = [
            "GIFT Nifty", "Nifty 50", "Sensex", "Bank Nifty", 
            "FinNifty", "Nifty Midcap 100", "Nifty Smallcap 100"
        ]
        
        # Check which indices are displayed
        found_indices = []
        missing_indices = []
        
        for idx in expected_indices:
            if idx in india_pulse:
                found_indices.append(idx)
            else:
                missing_indices.append(idx)
        
        print(f"\n    Found indices: {found_indices}")
        print(f"    Missing indices: {missing_indices}")
        
        # Check for "Waiting" messages (indicates data not loaded)
        waiting_patterns = [
            "Waiting for",
            "Loading",
            "No data",
            "unavailable"
        ]
        
        print("\n[3] Checking for loading/error states...")
        has_loading = False
        for pattern in waiting_patterns:
            if pattern.lower() in india_pulse.lower():
                print(f"    ⚠️ Found pattern: {pattern}")
                has_loading = True
        
        if not has_loading:
            print("    ✅ No loading/error patterns found")
        
        # Check if we have actual data (looking for numbers)
        import re
        numbers = re.findall(r'\d+[\d,\.]*', india_pulse)
        if len(numbers) > 5:
            print(f"    ✅ Found {len(numbers)} numeric values - data appears loaded")
        else:
            print(f"    ❌ Only found {len(numbers)} numeric values - possible data issue")
        
        # Check for error messages in console
        print("\n[4] Checking console for errors...")
        errors = [log for log in console_logs if "error" in log.lower()]
        if errors:
            print(f"    ⚠️ Found {len(errors)} error messages:")
            for e in errors[:5]:
                print(f"        {e}")
        else:
            print("    ✅ No console errors")
        
        # Final verdict
        print("\n" + "="*80)
        print("VERIFICATION RESULT")
        print("="*80)
        
        if not missing_indices and not has_loading:
            print("✅ SUCCESS: All 7 Indian indices are displaying correctly!")
        else:
            print(f"❌ ISSUES FOUND:")
            if missing_indices:
                print(f"   - Missing indices: {missing_indices}")
            if has_loading:
                print(f"   - Still loading or no data")
        
        # Take final screenshot
        page.screenshot(path="/tmp/indices_after.png", full_page=True)
        print("\nScreenshots saved:")
        print("  - /tmp/indices_before.png (initial)")
        print("  - /tmp/indices_after.png (final)")
        
        browser.close()
        
        return 0 if not missing_indices and not has_loading else 1


if __name__ == "__main__":
    sys.exit(main())
