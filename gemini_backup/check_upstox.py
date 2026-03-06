import os
import sys
import logging
import requests
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
from services.technical_analysis import _upstox_svc

print(f"Current Token (masked): {_upstox_svc.access_token[:5] if _upstox_svc.access_token else 'None'}...")
print(f"Is Authenticated: {_upstox_svc.is_authenticated}")

print("Triggering reload_token()...")
_upstox_svc.reload_token()

print(f"Post-Reload Token (masked): {_upstox_svc.access_token[:5] if _upstox_svc.access_token else 'None'}...")
print(f"Post-Reload Is Authenticated: {_upstox_svc.is_authenticated}")

from services.upstox_service import get_instrument_key
key = get_instrument_key("TCS.NS")
print(f"Instrument Key for TCS.NS: {key}")

yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

print("Testing V3 INTRA 1minute...")
# URL: https://api.upstox.com/v3/historical-candle/intraday/{instrumentKey}/{unit}/{interval}
url = f"https://api.upstox.com/v3/historical-candle/intraday/{encoded_key}/minutes/1"
print(f"V3 INTRA URL: {url}")
resp = requests.get(url, headers=headers, timeout=10)
if resp.status_code == 200:
    data = resp.json().get("data", {}).get("candles", [])
    print(f"V3 INTRA Success! Rows: {len(data)}")
else:
    print(f"V3 INTRA FAILED {resp.status_code}: {resp.text}")

profile = _upstox_svc.fetch_profile()
print(f"Profile: {profile}")
