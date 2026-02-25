import requests
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("UPSTOX_ACCESS_TOKEN")
api_key = os.getenv("UPSTOX_API_KEY")

print("Key:", api_key)
print("Token:", token)

# Let's try downloading the instrument list if needed, or testing an endpoint
if not token:
    print("No token! Authentication might be the issue.")

class UpstoxTest:
    def __init__(self):
        self.access_token = token
    def _headers(self):
        return {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
    def test_candle(self, key):
         # v2/historical-candle/NSE_EQ|INE002A01018/5minute/2023-11-12/2023-11-05
         url = f"https://api.upstox.com/v2/historical-candle/{key}/5minute/2026-02-25/2026-02-15"
         res = requests.get(url, headers=self._headers())
         print(f"Res for {key}:", res.status_code, res.text[:200])

ut = UpstoxTest()
if token:
    ut.test_candle("NSE_EQ|INE002A01018")
    ut.test_candle("NSE_EQ|RELIANCE")

