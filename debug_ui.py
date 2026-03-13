from playwright.sync_api import sync_playwright
import time
import os
import subprocess

def run():
    # Start the test server
    os.environ['SIMULATION'] = 'true'
    proc = subprocess.Popen(["./venv/bin/python", "-c", "import uvicorn; import main; uvicorn.run(main.app, host='127.0.0.1', port=8005)"])
    time.sleep(3)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.on("console", lambda msg: print(f"BROWSER: {msg.text}"))
            page.goto("http://127.0.0.1:8005")
            time.sleep(2)
            print("Clicking chart open")
            page.evaluate("openChart('RELIANCE')")
            time.sleep(5)
            
            # Print legend
            print("Legend HTML:", page.locator("#chart-legend").inner_html())
            browser.close()
    finally:
        proc.terminate()

if __name__ == "__main__":
    run()
