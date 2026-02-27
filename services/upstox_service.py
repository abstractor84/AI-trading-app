import os
import gzip
import json
import requests
import logging
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Cache for instrument key lookup: {symbol -> instrument_key}
_instrument_cache: dict[str, str] = {}

# -----------------------------------------------------------------------------
# Instrument key resolution
# Upstox uses keys like "NSE_EQ|INE002A01018" to identify instruments.
# We build a reverse-lookup table from the BOD JSON file published by Upstox.
# -----------------------------------------------------------------------------

def _load_instrument_cache() -> None:
    """Download and cache the NSE instruments JSON from Upstox."""
    global _instrument_cache
    if _instrument_cache:
        return  # already loaded

    url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with gzip.open(BytesIO(resp.content), 'rt', encoding='utf-8') as f:
            instruments = json.load(f)

        for inst in instruments:
            # Each instrument object has trading_symbol and instrument_key
            symbol = inst.get("trading_symbol", "")
            key = inst.get("instrument_key", "")
            if symbol and key:
                _instrument_cache[symbol.upper()] = key

        logger.info(f"Upstox: Loaded {len(_instrument_cache)} NSE instruments.")
    except Exception as e:
        logger.error(f"Upstox: Failed to load instrument cache: {e}")


def get_instrument_key(ticker: str) -> str | None:
    """Resolve a Yahoo Finance ticker like 'RELIANCE.NS' to an Upstox key."""
    _load_instrument_cache()
    clean = ticker.replace(".NS", "").upper()
    key = _instrument_cache.get(clean)
    if not key:
        logger.warning(f"Upstox: No instrument key found for '{clean}'.")
    return key


# -----------------------------------------------------------------------------
# Main service class
# -----------------------------------------------------------------------------

class UpstoxService:
    """
    A data service that fetches OHLCV data from the Upstox v2 REST API.

    Authentication:
      Upstox uses short-lived OAuth2 Bearer tokens. The token must be stored in
      the UPSTOX_ACCESS_TOKEN env variable (obtained via the Upstox login flow
      or manually copied from https://account.upstox.com/developer/apps).

    Fallback:
      All methods return None when unauthenticated or on API error, allowing the
      caller (TechnicalAnalysisService) to fall back to yfinance.
    """

    BASE_URL = "https://api.upstox.com/v2"

    def __init__(self):
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        self.is_authenticated = bool(self.access_token)
        if not self.is_authenticated:
            logger.warning(
                "Upstox: UPSTOX_ACCESS_TOKEN not set. "
                "Provide a token via .env to enable Upstox data. yfinance will be used as fallback."
            )

    def _headers(self) -> dict:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def validate_token(self) -> bool:
        """Actively verify if the token is still valid with a lightweight API call."""
        if not self.access_token:
            return False
        url = f"{self.BASE_URL}/user/profile"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                self.is_authenticated = True
                return True
            else:
                logger.error(f"Upstox token validation failed: {resp.status_code} {resp.text}")
                self.is_authenticated = False
                return False
        except Exception as e:
            logger.error(f"Upstox validation exception: {e}")
            self.is_authenticated = False
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
        except Exception as e:
            logger.error(f"Upstox fetch_profile exception: {e}")
        return None

    def reload_token(self):
        """Reload token from environment variable and re-validate."""
        import os
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        self.is_authenticated = False
        if self.access_token:
            logger.info("Upstox token reloaded. Validating...")
            self.validate_token()
        else:
            logger.warning("Upstox token reload failed: No token in env.")

    # ------------------------------------------------------------------
    # Low-level API helpers
    # ------------------------------------------------------------------

    def _candles_to_df(self, candles: list) -> pd.DataFrame:
        """
        Convert raw Upstox candles to a DataFrame that matches yfinance format.

        Candle format: [Timestamp, Open, High, Low, Close, Volume, OI]
        """
        if not candles:
            return pd.DataFrame()

        rows = []
        for c in candles:
            ts = pd.to_datetime(c[0])
            rows.append({
                "Open":   float(c[1]),
                "High":   float(c[2]),
                "Low":    float(c[3]),
                "Close":  float(c[4]),
                "Volume": int(c[5]),
            })
        timestamps = [pd.to_datetime(c[0]) for c in candles]
        df = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps))
        df.index.name = "Datetime"
        df.sort_index(inplace=True)

        # Ensure timezone-aware index (IST = UTC+5:30)
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")

        return df

    def fetch_intraday_candles(self, instrument_key: str, interval: str = "5minute") -> pd.DataFrame | None:
        """
        Fetch today's intraday OHLCV data.
        Endpoint: GET /v2/historical-candle/intraday/{instrument_key}/{interval}
        """
        if not self.is_authenticated:
            return None

        encoded_key = requests.utils.quote(instrument_key, safe='')
        url = f"{self.BASE_URL}/historical-candle/intraday/{encoded_key}/{interval}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=8)
            if resp.status_code == 200:
                candles = resp.json().get("data", {}).get("candles", [])
                return self._candles_to_df(candles)
            elif resp.status_code == 401:
                logger.error("Upstox: Unauthorised — access token expired or invalid.")
            elif resp.status_code == 429:
                logger.error("Upstox: Rate limit hit.")
            else:
                logger.error(f"Upstox intraday error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error(f"Upstox fetch_intraday_candles exception: {e}")
        return None

    def fetch_historical_candles(
        self,
        instrument_key: str,
        interval: str = "5minute",
        days: int = 5,
    ) -> pd.DataFrame | None:
        """
        Fetch historical OHLCV data for the last `days` days.
        Endpoint: GET /v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}
        """
        if not self.is_authenticated:
            return None

        to_date = datetime.today().strftime("%Y-%m-%d")
        from_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        encoded_key = requests.utils.quote(instrument_key, safe='')
        url = f"{self.BASE_URL}/historical-candle/{encoded_key}/{interval}/{to_date}/{from_date}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=8)
            if resp.status_code == 200:
                candles = resp.json().get("data", {}).get("candles", [])
                return self._candles_to_df(candles)
            elif resp.status_code == 401:
                logger.error("Upstox: Unauthorised — access token expired or invalid.")
            elif resp.status_code == 429:
                logger.error("Upstox: Rate limit hit.")
            else:
                logger.error(f"Upstox historical error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error(f"Upstox fetch_historical_candles exception: {e}")
        return None

    def fetch_ohlcv(self, ticker: str, days: int = 5, interval: str = "5minute") -> pd.DataFrame | None:
        """
        High-level method: resolves ticker → instrument_key, then fetches
        intraday (today's session) + historical (last `days` days) and merges.
        Returns a DataFrame with the same columns as yfinance or None on failure.
        """
        if not self.is_authenticated:
            return None

        key = get_instrument_key(ticker)
        if not key:
            return None

        frames = []

        # Today's intraday data (real-time, lower latency)
        td = self.fetch_intraday_candles(key, interval)
        if td is not None and not td.empty:
            frames.append(td)

        # Historical data for prior days
        if days > 1:
            hist = self.fetch_historical_candles(key, interval, days=days)
            if hist is not None and not hist.empty:
                frames.append(hist)

        if not frames:
            return None

        combined = pd.concat(frames).sort_index()
        # Drop duplicates (timestamps that appear in both intraday and historical)
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.dropna(inplace=True)
        return combined

    def fetch_market_quote(self, instrument_key: str) -> dict | None:
        """Fetch near real-time LTP quote."""
        if not self.is_authenticated:
            return None
        params = {"instrument_key": instrument_key}
        url = f"{self.BASE_URL}/market-quote/quotes"
        try:
            resp = requests.get(url, headers=self._headers(), params=params, timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Upstox fetch_market_quote exception: {e}")
        return None
