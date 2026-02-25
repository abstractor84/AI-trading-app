import requests
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("UPSTOX_ACCESS_TOKEN")

if token:
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    url = "https://api.upstox.com/v2/historical-candle/intraday/NSE_EQ|INE002A01018/5minute"
    try:
        res = requests.get(url, headers=headers)
        print("Status", res.status_code)
        print(res.json().get('data', {}).get('candles', [])[:2])
    except Exception as e:
        print("Err:", e)
else:
    print("No Token")
