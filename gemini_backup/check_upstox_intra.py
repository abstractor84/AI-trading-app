import os
import sys
import logging
import requests
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
from services.technical_analysis import _upstox_svc

_upstox_svc.reload_token()

from services.upstox_service import get_instrument_key
key = get_instrument_key("TCS.NS")
encoded_key = requests.utils.quote(key, safe='')
headers = _upstox_svc._headers()

print("Testing V3 HISTORICAL with today 1minute...")
today = datetime.today().strftime("%Y-%m-%d")
url = f"https://api.upstox.com/v3/historical-candle/{encoded_key}/minutes/1/{today}/{today}"
print(f"V3 INTRA URL: {url}")
resp = requests.get(url, headers=headers, timeout=10)
if resp.status_code == 200:
    data = resp.json().get("data", {}).get("candles", [])
    print(f"V3 INTRA Success! Rows: {len(data)}")
    if data:
        print(data[:2])
else:
    print(f"V3 INTRA FAILED {resp.status_code}: {resp.text}")
