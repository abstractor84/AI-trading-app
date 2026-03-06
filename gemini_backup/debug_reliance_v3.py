import os
import sys
import logging
import requests
from datetime import datetime
# Add current directory to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
from services.technical_analysis import _upstox_svc

_upstox_svc.reload_token()

from services.upstox_service import get_instrument_key
ticker = "RELIANCE.NS"
key = get_instrument_key(ticker)
encoded_key = requests.utils.quote(key, safe='')
headers = _upstox_svc._headers()

print(f"Testing V3 INTRA for {ticker} (Key: {key})...")
url_intra = f"https://api.upstox.com/v3/historical-candle/intraday/{encoded_key}/minutes/1"
resp = requests.get(url_intra, headers=headers)
print(f"Intraday URL: {url_intra}")
if resp.status_code == 200:
    data = resp.json().get("data", {}).get("candles", [])
    print(f"Rows: {len(data)}")
    if data: print(data[:2])
else:
    print(f"Failed {resp.status_code}: {resp.text}")

print("\nTesting V3 HISTORICAL TODAY...")
today = datetime.today().strftime("%Y-%m-%d")
url_hist = f"https://api.upstox.com/v3/historical-candle/{encoded_key}/minutes/1/{today}/{today}"
resp = requests.get(url_hist, headers=headers)
print(f"Historical URL: {url_hist}")
if resp.status_code == 200:
    data = resp.json().get("data", {}).get("candles", [])
    print(f"Rows: {len(data)}")
    if data: print(data[:2])
else:
    print(f"Failed {resp.status_code}: {resp.text}")
