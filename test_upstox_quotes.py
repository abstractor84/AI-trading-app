import os
import requests
token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()
if token:
    url = f"https://api.upstox.com/v3/market-quote/quotes?instrument_key=NSE_INDEX%7CNifty%2050"
    resp = requests.get(url, headers={"Accept": "application/json", "Authorization": f"Bearer {token}"})
    print(resp.json())
