import os
import gzip
import json
import requests
import logging
import pandas as pd
import pytz
import time
from datetime import datetime, timedelta
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Instrument key resolution
# Upstox uses keys like "NSE_EQ|INE002A01018" to identify instruments.
# We build a reverse-lookup table from the BOD JSON file published by Upstox.
# -----------------------------------------------------------------------------

_instrument_cache = {}
_reverse_instrument_cache = {}
_last_cache_load = 0

def _load_instrument_cache() -> None:
    """Download and cache the NSE instruments JSON from Upstox."""
    global _instrument_cache, _reverse_instrument_cache, _last_cache_load
    now = time.time()
    if _instrument_cache and (now - _last_cache_load < 86400):
        return  # already loaded today

    url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with gzip.open(BytesIO(resp.content), 'rt', encoding='utf-8') as f:
            instruments = json.load(f)

        _instrument_cache.clear()
        _reverse_instrument_cache.clear()
        for inst in instruments:
            # Each instrument object has trading_symbol and instrument_key
            symbol = inst.get("trading_symbol", "")
            key = inst.get("instrument_key", "")
            if symbol and key:
                symbol_up = symbol.upper()
                _reverse_instrument_cache[key] = symbol_up
                _reverse_instrument_cache[key.replace("|", ":")] = symbol_up
                
                # Cache standard symbol
                _instrument_cache[symbol_up] = key
                _instrument_cache[f"{symbol_up}.NS"] = key
                
                # Handle common Upstox variants (e.g. TATAMOTORS-EQ vs TATAMOTORS)
                if "-EQ" in symbol_up:
                    base = symbol_up.replace("-EQ", "")
                    _instrument_cache[base] = key
                    _instrument_cache[f"{base}.NS"] = key

        logger.info(f"Upstox: Loaded {len(_instrument_cache)} NSE instruments.")
        
        # SKEPTIC: Manually inject GIFT Nifty which is often missing from public BOD list but available via API
        _instrument_cache["GIFT NIFTY"] = "NSE_INDEX|GIFT Nifty"
        _instrument_cache["GIFT NIFTY.NS"] = "NSE_INDEX|GIFT Nifty"
        _reverse_instrument_cache["NSE_INDEX|GIFT Nifty"] = "GIFT NIFTY"
        
        _last_cache_load = now
    except Exception as e:
        logger.error(f"Upstox: Failed to load instrument cache: {e}")


def get_instrument_key(ticker: str) -> str | None:
    """Resolve a Yahoo Finance ticker like 'RELIANCE.NS' to an Upstox key."""
    if not ticker: return None
    # SKEPTIC: If it's already a key (contains |), return it.
    if "|" in ticker: return ticker
    _load_instrument_cache()
    return _instrument_cache.get(ticker.upper())


def get_symbol_from_key(instrument_key: str) -> str:
    """Reverse lookup from instrument key to symbol."""
    _load_instrument_cache()
    # Handle both pipe and colon separators
    return _reverse_instrument_cache.get(instrument_key, 
           _reverse_instrument_cache.get(instrument_key.replace("|", ":"), 
           instrument_key.split("|")[-1].split(":")[-1]))


class UpstoxService:
    """
    V3 REST API Service for Upstox (2026 Edition)
    --------------------------------------------
    Provides methods for fetching OHLCV data and market quotes.
    Handles automatic authentication checks and provides clean dataframes.
    If credentials are missing or token is invalid, it fails gracefully allowing
    caller (TechnicalAnalysisService) to fall back to yfinance.
    """

    BASE_URL = "https://api.upstox.com/v3"

    def __init__(self):
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        self._is_authenticated = None

    @property
    def is_authenticated(self) -> bool:
        if self._is_authenticated is not None:
            return self._is_authenticated
        return bool(os.getenv("UPSTOX_ACCESS_TOKEN"))

    @is_authenticated.setter
    def is_authenticated(self, value: bool):
        self._is_authenticated = value

    def _headers(self) -> dict:
        token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip().replace('"', '').replace("'", "")
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def _handle_api_error(self, resp: requests.Response, context: str):
        """Centralized API error handling with auth-state tracking."""
        if resp.status_code == 401:
            logger.error(f"Upstox Auth Failure ({context}): Token invalid or expired. Switching to fallback mode.")
            self._is_authenticated = False
        elif resp.status_code == 400:
            # We already log specific 400s in callers, but this catches leftovers
            logger.debug(f"Upstox 400 ({context}): {resp.text[:200]}")
        else:
            logger.error(f"Upstox API Error {resp.status_code} ({context}): {resp.text[:200]}")

    def reload_token(self) -> None:
        """Reload the token from environment variables (useful after manual login)."""
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        self._is_authenticated = None # Reset so property re-evaluates
        # Validation will be checked on next API call or via validate_token()

    def validate_token(self) -> bool:
        """Actively verify if the token is still valid with a lightweight API call."""
        if not self.is_authenticated:
            return False
        url = f"{self.BASE_URL}/user/profile"
        try:
            # Note: user/profile might still be under v2 or v3, checking v3 first
            resp = requests.get(url, headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                return True
            # Fallback to v2 if v3 profile is not yet available
            resp_v2 = requests.get("https://api.upstox.com/v2/user/profile", headers=self._headers(), timeout=5)
            return resp_v2.status_code == 200
        except Exception:
            return False

    def fetch_profile(self) -> dict | None:
        """Fetch user profile details to verify connection."""
        if not self.is_authenticated:
            return None
        url = f"{self.BASE_URL}/user/profile"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                return resp.json().get("data")
            # Fallback to v2
            resp_v2 = requests.get("https://api.upstox.com/v2/user/profile", headers=self._headers(), timeout=5)
            if resp_v2.status_code == 200:
                return resp_v2.json().get("data")
        except Exception:
            pass
        return None

    def _candles_to_df(self, candles: list) -> pd.DataFrame | None:
        """Helper: Converts raw Upstox candle list to a standardized DataFrame."""
        if not candles or len(candles) == 0:
            return None
        # Upstox V3 candle format: [timestamp, open, high, low, close, volume, open_interest]
        df = pd.DataFrame(candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Localize to UTC then convert to IST
        try:
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')
        except Exception:
            # Fallback for already correct or weird formats in tests
            pass
            
        df.set_index('timestamp', inplace=True)
        # Convert all to float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Sort by time
        df.sort_index(inplace=True)
        # Drop columns we don't need
        if 'OI' in df.columns:
            df.drop(columns=['OI'], inplace=True)
        
        df.dropna(subset=['Close'], inplace=True)
        return df if not df.empty else None

    def fetch_intraday_candles(self, instrument_key: str, interval: str = "1", unit: str = "minutes") -> pd.DataFrame | None:
        """
        Fetch intraday bars for the current session (V3).
        Endpoint: /v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}
        """
        if not self.is_authenticated:
            return None

        # Clean interval (e.g. '1minute' -> '1', 'day' -> '1')
        num_interval = "".join(filter(str.isdigit, interval)) or "1"
        
        encoded_key = requests.utils.quote(instrument_key, safe='')
        url = f"{self.BASE_URL}/historical-candle/intraday/{encoded_key}/{unit}/{num_interval}"
        
        try:
            resp = requests.get(url, headers=self._headers(), timeout=8)
            if resp.status_code == 200:
                candles = resp.json().get("data", {}).get("candles", [])
                return self._candles_to_df(candles)
            else:
                self._handle_api_error(resp, "intraday")
        except Exception as e:
            logger.error(f"Upstox fetch_intraday_candles exception: {e}")
        return None

    def fetch_historical_candles(
        self,
        instrument_key: str,
        unit: str = "minutes",
        interval: str = "1",
        days: int = 5,
    ) -> pd.DataFrame | None:
        """
        Fetch historical OHLCV data for the last `days` days (V3).
        Endpoint: GET /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}
        """
        if not self.is_authenticated:
            return None

        # V3 requires to_date before from_date
        to_date = datetime.today().strftime("%Y-%m-%d")
        from_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        num_interval = "".join(filter(str.isdigit, interval)) or "1"
        encoded_key = requests.utils.quote(instrument_key, safe='')
        
        url = f"{self.BASE_URL}/historical-candle/{encoded_key}/{unit}/{num_interval}/{to_date}/{from_date}"
        logger.debug(f"SKEPTIC: Upstox Historical Call for {instrument_key}")
        
        try:
            resp = requests.get(url, headers=self._headers(), timeout=8)
            if resp.status_code == 200:
                candles = resp.json().get("data", {}).get("candles", [])
                return self._candles_to_df(candles)
            elif resp.status_code == 400 and "INDEX" in instrument_key:
                # SKEPTIC: Known restriction for some indices on Upstox
                logger.warning(f"SKEPTIC: Upstox Historical Data restricted for {instrument_key} (400). Skipping.")
            else:
                self._handle_api_error(resp, "historical")
        except Exception as e:
            logger.error(f"Upstox fetch_historical_candles exception: {e}")
        return None

    def fetch_ohlcv(self, ticker: str, days: int = 5, interval: str = "1", unit: str = "minutes") -> pd.DataFrame | None:
        """
        High-level method (V3): fetches intraday + historical and merges.
        """
        if not self.is_authenticated:
            return None

        # SKEPTIC: Upstox V3 is strict about plural units.
        if unit == "minute": unit = "minutes"
        if unit == "hour": unit = "hours"
        if unit == "day": unit = "days"
        if unit == "week": unit = "weeks"
        if unit == "month": unit = "months"

        key = get_instrument_key(ticker)
        if not key:
            return None

        # API Limits (V3):
        if unit == "minutes" and days > 20: days = 20
        if unit == "hours" and days > 90: days = 90

        frames = []

        # SKEPTIC: Intraday endpoint is only for sub-day granularities.
        if unit in ["minutes", "hours"]:
            td = self.fetch_intraday_candles(key, interval, unit)
            if td is not None and not td.empty:
                frames.append(td)

        # Historical data for prior days
        if days >= 1:
            hist = self.fetch_historical_candles(key, unit, interval, days=days)
            if hist is not None and not hist.empty:
                frames.append(hist)

        if not frames:
            return None

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.dropna(inplace=True)
        return combined

    def fetch_market_quote(self, instrument_key: str) -> dict | None:
        """Fetch real-time snapshot quote using REST API (V3). Returns V2-compatible structure. Supports comma-separated keys."""
        if not self.is_authenticated:
            return None

        # SKEPTIC: We use the 'quotes' endpoint instead of 'ltp' to get 'close' price for change calculation
        url = f"{self.BASE_URL}/market-quote/quotes?instrument_key={requests.utils.quote(instrument_key, safe='')}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                raw_data = resp.json().get("data", {})

                normalized_data = {}
                keys_requested = instrument_key.split(",")
                for key in keys_requested:
                    alt_key = key.replace("|", ":")
                    target_key = key if key in raw_data else alt_key if alt_key in raw_data else None
                    if target_key:
                        normalized_data[key] = raw_data[target_key]
                    else:
                        logger.warning(f"Upstox: {key} not found in quote data.")

                if normalized_data:
                    return {
                        "status": "success",
                        "data": normalized_data
                    }
            else:
                self._handle_api_error(resp, "market_quote")
        except Exception as e:
            logger.error(f"Upstox fetch_market_quote exception: {e}")
        return None
upstox_client = UpstoxService()
