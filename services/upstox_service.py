import os
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class UpstoxService:
    def __init__(self):
        self.api_key = os.getenv("UPSTOX_API_KEY")
        self.api_secret = os.getenv("UPSTOX_API_SECRET")
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN") # Will be managed via OAuth flow in real app
        self.base_url = "https://api.upstox.com/v2"
        self.is_authenticated = bool(self.access_token)
        
        # Upstox specific instrument keys mapping
        # E.g. RELIANCE is "NSE_EQ|INE002A01018"
        # Since we use symbol names typically, we'll need a mapping layer or dynamic fetching
        self.instrument_cache = {}
        
    def _headers(self):
        return {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }

    def fetch_historical_candles(self, instrument_key: str, interval: str = "5minute", to_date: str = "", from_date: str = ""):
        """Fetch historical candle data from Upstox API."""
        if not self.is_authenticated:
            logger.warning("Upstox: Not authenticated. Using fallback YFinance.")
            return None
            
        url = f"{self.base_url}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
        try:
            response = requests.get(url, headers=self._headers(), timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('data', {}).get('candles', [])
            elif response.status_code == 429:
                logger.error("Upstox: Rate limit exceeded.")
            else:
                logger.error(f"Upstox API Error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Upstox Request Exception: {e}")
        return None

    def fetch_market_quote(self, instrument_key: str):
        """Fetch near real-time quote data (LTP) from Upstox API."""
        if not self.is_authenticated:
            return None
            
        params = {'instrument_key': instrument_key}
        url = f"{self.base_url}/market-quote/quotes"
        try:
            response = requests.get(url, headers=self._headers(), params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Upstox Request Exception: {e}")
        return None
