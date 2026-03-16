import os
import requests
token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()
print(f"Token present: {bool(token)}")
if token:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    url = "https://api.upstox.com/v3/historical-candle/intraday/NSE_EQ%7CINE002A01018/minutes/5"
    resp = requests.get(url, headers=headers)
    print(resp.status_code, resp.text[:200])
