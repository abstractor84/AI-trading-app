import os
import gzip
import json
import requests
import logging
import pandas as pd
import pytz
import time
import random
import threading
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from dotenv import load_dotenv
from functools import wraps
from enum import Enum
from typing import Optional, Dict, Any, List

load_dotenv()

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Error categorization for better debugging and monitoring."""
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    VALIDATION = "validation"
    API = "api"
    PARSING = "parsing"
    UNKNOWN = "unknown"


def categorize_error(response_status: int = None, response_text: str = None, exception: Exception = None) -> ErrorCategory:
    """Categorize an error based on response status or exception type."""
    if response_status:
        if response_status == 401 or response_status == 403:
            return ErrorCategory.AUTHENTICATION
        elif response_status == 429:
            return ErrorCategory.RATE_LIMIT
        elif response_status == 400 or response_status == 422:
            return ErrorCategory.VALIDATION
        elif response_status >= 500:
            return ErrorCategory.API
    
    if exception:
        if isinstance(exception, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return ErrorCategory.NETWORK
        elif isinstance(exception, (json.JSONDecodeError, ValueError)):
            return ErrorCategory.PARSING
    
    return ErrorCategory.UNKNOWN


def create_error_payload(
    endpoint: str,
    method: str,
    status_code: int = None,
    response_text: str = None,
    exception: Exception = None,
    request_params: dict = None,
    correlation_id: str = None
) -> Dict[str, Any]:
    """Create a structured error payload for logging."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    payload = {
        "correlation_id": correlation_id,
        "endpoint": endpoint,
        "method": method,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if request_params:
        # Sanitize sensitive params
        safe_params = {k: v for k, v in request_params.items() if k.lower() not in ['password', 'token', 'secret', 'key']}
        payload["request_params"] = safe_params
    
    if status_code:
        payload["status_code"] = status_code
        payload["error_category"] = categorize_error(response_status=status_code).value
    
    if response_text:
        payload["response_text"] = response_text[:500] if len(response_text) > 500 else response_text
    
    if exception:
        payload["exception_type"] = type(exception).__name__
        payload["exception_message"] = str(exception)
        if status_code is None:
            payload["error_category"] = categorize_error(exception=exception).value
    
    return payload

# Rate limit tracking
global_rate_limit_state = {
    "is_rate_limited": False,
    "rate_limited_at": None,
    "retry_after": 0,
    "last_success": None,
    "consecutive_failures": 0,
    "lock": threading.Lock()
}

# Rate limit configuration
RATE_LIMIT_COOLDOWN_SECONDS = 60  # Initial cooldown
MAX_RATE_LIMIT_COOLDOWN = 300  # Max 5 minutes
MAX_CONSECUTIVE_FAILURES = 5


def is_rate_limited():
    """Check if we're currently rate limited."""
    with global_rate_limit_state["lock"]:
        if not global_rate_limit_state["is_rate_limited"]:
            return False
        
        # Check if cooldown period has passed
        if global_rate_limit_state["rate_limited_at"]:
            elapsed = time.time() - global_rate_limit_state["rate_limited_at"]
            if elapsed >= global_rate_limit_state["retry_after"]:
                global_rate_limit_state["is_rate_limited"] = False
                global_rate_limit_state["rate_limited_at"] = None
                global_rate_limit_state["retry_after"] = 0
                logger.info("Rate limit cooldown period expired, resetting rate limit state")
                return False
        
        return True


def get_rate_limit_status():
    """Get current rate limit status for UI display."""
    with global_rate_limit_state["lock"]:
        if not global_rate_limit_state["is_rate_limited"]:
            return {
                "is_rate_limited": False,
                "remaining_cooldown": 0,
                "last_success": global_rate_limit_state["last_success"],
                "consecutive_failures": global_rate_limit_state["consecutive_failures"]
            }
        
        elapsed = time.time() - (global_rate_limit_state["rate_limited_at"] or time.time())
        remaining = max(0, global_rate_limit_state["retry_after"] - elapsed)
        
        return {
            "is_rate_limited": True,
            "remaining_cooldown": int(remaining),
            "rate_limited_at": global_rate_limit_state["rate_limited_at"],
            "retry_after": global_rate_limit_state["retry_after"],
            "last_success": global_rate_limit_state["last_success"],
            "consecutive_failures": global_rate_limit_state["consecutive_failures"]
        }


def handle_rate_limit(retry_after: int = None):
    """Handle rate limiting with exponential backoff."""
    with global_rate_limit_state["lock"]:
        global_rate_limit_state["is_rate_limited"] = True
        global_rate_limit_state["rate_limited_at"] = time.time()
        global_rate_limit_state["consecutive_failures"] += 1
        
        # Calculate retry_after with exponential backoff and jitter
        if retry_after:
            base_cooldown = retry_after
        else:
            base_cooldown = RATE_LIMIT_COOLDOWN_SECONDS * (
                2 ** min(global_rate_limit_state["consecutive_failures"], 4)
            )
        
        # Add jitter (random 0-25% of base)
        jitter = random.uniform(0, base_cooldown * 0.25)
        global_rate_limit_state["retry_after"] = min(
            base_cooldown + jitter, 
            MAX_RATE_LIMIT_COOLDOWN
        )
        
        logger.warning(
            f"Rate limit triggered! Cooldown: {global_rate_limit_state['retry_after']:.1f}s "
            f"(failures: {global_rate_limit_state['consecutive_failures']})"
        )


def reset_rate_limit():
    """Reset rate limit state after successful request."""
    with global_rate_limit_state["lock"]:
        global_rate_limit_state["is_rate_limited"] = False
        global_rate_limit_state["rate_limited_at"] = None
        global_rate_limit_state["retry_after"] = 0
        global_rate_limit_state["consecutive_failures"] = 0
        global_rate_limit_state["last_success"] = time.time()


def wait_if_rate_limited():
    """Wait if rate limited (returns True if waited)."""
    if is_rate_limited():
        status = get_rate_limit_status()
        wait_time = status["remaining_cooldown"]
        if wait_time > 0:
            logger.info(f"Waiting {wait_time}s for rate limit cooldown...")
            time.sleep(min(wait_time, 5))  # Max sleep 5 seconds to not block too long
            return True
    return False


def rate_limit_handling(func):
    """Decorator to add rate limit handling to API calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check rate limit before making request
        if is_rate_limited():
            wait_if_rate_limited()
        
        try:
            result = func(*args, **kwargs)
            # Success - reset failure count
            reset_rate_limit()
            return result
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited - extract retry-after header if present
                retry_after = None
                if e.response.headers:
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            retry_after = int(retry_after)
                        except ValueError:
                            retry_after = None
                handle_rate_limit(retry_after)
                wait_if_rate_limited()
                # Retry once after waiting
                result = func(*args, **kwargs)
                reset_rate_limit()
                return result
            else:
                raise
        except Exception as e:
            # Network errors - increment failure count
            with global_rate_limit_state["lock"]:
                global_rate_limit_state["consecutive_failures"] += 1
                if global_rate_limit_state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
                    handle_rate_limit()
            raise
    return wrapper


# -----------------------------------------------------------------------------
# Instrument key resolution
# Upstox uses keys like "NSE_EQ|INE002A01018" to identify instruments.
# We build a reverse-lookup table from the BOD JSON file published by Upstox.
# -----------------------------------------------------------------------------

# Exchange URLs for instrument downloads
EXCHANGE_URLS = {
    "NSE": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    "BSE": "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz",
    "MCX": "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz"
}

# Search API endpoint
SEARCH_API_URL = "https://api.upstox.com/v2/instruments/search"

# Cache for search results
_search_cache = {}
_search_cache_time = {}
SEARCH_CACHE_TTL = 300  # 5 minutes

_instrument_cache = {}
_reverse_instrument_cache = {}
_last_cache_load = 0

def _load_instrument_cache(exchange: str = "NSE") -> None:
    """Download and cache instruments JSON from Upstox for a specific exchange."""
    global _instrument_cache, _reverse_instrument_cache, _last_cache_load
    now = time.time()
    
    # Check if we have cached data for this exchange recently
    cache_key = f"_instrument_cache_{exchange}"
    cache = globals().get(cache_key, {})
    reverse_cache = globals().get(f"_reverse_instrument_cache_{exchange}", {})
    last_load = globals().get(f"_last_cache_load_{exchange}", 0)
    
    if cache and (now - last_load < 86400):
        return  # already loaded today

    url = EXCHANGE_URLS.get(exchange, EXCHANGE_URLS["NSE"])
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with gzip.open(BytesIO(resp.content), 'rt', encoding='utf-8') as f:
            instruments = json.load(f)

        # Build exchange-specific cache
        new_cache = {}
        new_reverse_cache = {}
        
        for inst in instruments:
            # Each instrument object has trading_symbol and instrument_key
            symbol = inst.get("trading_symbol", "")
            key = inst.get("instrument_key", "")
            if symbol and key:
                symbol_up = symbol.upper()
                new_reverse_cache[key] = symbol_up
                new_reverse_cache[key.replace("|", ":")] = symbol_up
                
                # Add exchange suffix for disambiguation
                exchange_suffix = "." + exchange
                new_cache[symbol_up] = key
                new_cache[f"{symbol_up}{exchange_suffix}"] = key
                
                # Handle common Upstox variants (e.g. TATAMOTORS-EQ vs TATAMOTORS)
                if "-EQ" in symbol_up:
                    base = symbol_up.replace("-EQ", "")
                    new_cache[base] = key
                    new_cache[f"{base}{exchange_suffix}"] = key

        # Store in exchange-specific globals
        globals()[cache_key] = new_cache
        globals()[f"_reverse_instrument_cache_{exchange}"] = new_reverse_cache
        globals()[f"_last_cache_load_{exchange}"] = now
        
        # Also merge into main cache
        _instrument_cache.update(new_cache)
        _reverse_instrument_cache.update(new_reverse_cache)

        logger.info(f"Upstox: Loaded {len(new_cache)} {exchange} instruments.")

    except Exception as e:
        logger.error(f"Upstox: Failed to load {exchange} instrument cache: {e}")


def load_all_exchanges() -> None:
    """Load instruments for all supported exchanges."""
    for exchange in EXCHANGE_URLS.keys():
        _load_instrument_cache(exchange)
    logger.info(f"Upstox: Loaded instruments for all exchanges. Total cache size: {len(_instrument_cache)}")


def get_instrument_key(ticker: str, use_search: bool = True) -> str | None:
    """Resolve a Yahoo Finance ticker like 'RELIANCE.NS' to an Upstox key.
    
    Args:
        ticker: The ticker symbol to resolve (e.g., 'RELIANCE.NS', 'RELIANCE')
        use_search: If True, try search API first (more efficient). If False, use cache only.
    """
    if not ticker: return None
    # SKEPTIC: If it's already a key (contains | or :), return it.
    if "|" in ticker or ":" in ticker: return ticker
    
    # First try cache
    _load_instrument_cache()
    result = _instrument_cache.get(ticker.upper())
    if result:
        return result
    
    # Try with .NS suffix removed
    clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")
    result = _instrument_cache.get(clean_ticker)
    if result:
        return result
    
    # If use_search is True and cache miss, try search API
    if use_search:
        # Try search API for this ticker
        try:
            from services.upstox_service import upstox_client
            search_results = upstox_client.search_instruments(query=clean_ticker)
            if search_results and len(search_results) > 0:
                # Return the first matching instrument key
                first_result = search_results[0]
                if isinstance(first_result, dict):
                    key = first_result.get("instrument_key")
                    if key:
                        logger.debug(f"Found instrument key via search API: {ticker} -> {key}")
                        return key
        except Exception as e:
            logger.debug(f"Search API lookup failed for {ticker}: {e}")
    
    return None


def get_symbol_from_key(instrument_key: str) -> str:
    """Reverse lookup from instrument key to symbol."""
    _load_instrument_cache()
    # Handle both pipe and colon separators
    return _reverse_instrument_cache.get(instrument_key, 
           _reverse_instrument_cache.get(instrument_key.replace("|", ":"), 
           instrument_key.split("|")[-1].split(":")[-1]))


class UpstoxService:
    """
    REST API Service for Upstox (Hybrid V2/V3)
    --------------------------------------------
    Uses V3 for Historical data and V2 for Profiles and Quotes.
    Enhanced with rate limit handling.
    """

    BASE_URL = "https://api.upstox.com/v2"
    V3_URL = "https://api.upstox.com/v3"
    SEARCH_URL = "https://api.upstox.com/v2/instruments/search"

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

    def _handle_api_error(self, resp: requests.Response, context: str, request_params: dict = None):
        """
        Centralized API error handling with auth-state tracking and rate limit detection.
        
        Enhanced with structured error payloads and correlation IDs for better debugging.
        """
        correlation_id = str(uuid.uuid4())
        
        # Create structured error payload
        error_payload = create_error_payload(
            endpoint=context,
            method="GET",
            status_code=resp.status_code,
            response_text=resp.text,
            request_params=request_params,
            correlation_id=correlation_id
        )
        
        if resp.status_code == 401:
            logger.error(
                f"Upstox Auth Failure ({context}): Token invalid or expired. "
                f"Switching to fallback mode. (correlation_id={correlation_id})"
            )
            self._is_authenticated = False
            error_payload["action_taken"] = "set_authenticated_false"
        elif resp.status_code == 429:
            # Rate limited
            retry_after = resp.headers.get('Retry-After')
            retry_seconds = int(retry_after) if retry_after else 60
            handle_rate_limit(retry_seconds)
            logger.warning(
                f"Upstox Rate Limited ({context}): Retry after {retry_seconds}s "
                f"(correlation_id={correlation_id})"
            )
            error_payload["retry_after"] = retry_seconds
            error_payload["action_taken"] = "rate_limit_handled"
        elif resp.status_code == 400:
            # We already log specific 400s in callers, but this catches leftovers
            logger.debug(f"Upstox 400 ({context}): {resp.text[:200]} (correlation_id={correlation_id})")
            error_payload["action_taken"] = "logged"
        else:
            logger.error(
                f"Upstox API Error {resp.status_code} ({context}): {resp.text[:200]} "
                f"(correlation_id={correlation_id})"
            )
            error_payload["action_taken"] = "error_logged"
        
        # Log structured error payload for debugging
        logger.debug(f"Upstox API Error Details: {json.dumps(error_payload)}")

    def reload_token(self) -> None:
        """Reload the token from environment variables (useful after manual login)."""
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        self._is_authenticated = None # Reset so property re-evaluates
        # Validation will be checked on next API call or via validate_token()
        
        # Clear the connection status cache to force a fresh check
        try:
            from services.technical_analysis import clear_connection_status_cache
            clear_connection_status_cache()
        except ImportError as e:
            logger.warning(f"Could not import clear_connection_status_cache: {e}")

    def validate_token(self) -> bool:
        """Actively verify if the token is still valid with a lightweight API call."""
        if not self.is_authenticated:
            return False
        # Profile is still v2
        url = "https://api.upstox.com/v2/user/profile"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                reset_rate_limit()
                return True
            elif resp.status_code == 429:
                handle_rate_limit()
            return False
        except Exception:
            return False

    def fetch_profile(self) -> dict | None:
        """Fetch user profile details to verify connection."""
        if not self.is_authenticated:
            return None
        url = "https://api.upstox.com/v2/user/profile"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=5)
            if resp.status_code == 200:
                reset_rate_limit()
                return resp.json().get("data")
            elif resp.status_code == 429:
                handle_rate_limit()
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

    @rate_limit_handling
    def fetch_intraday_candles(self, instrument_key: str, interval: str = "1", unit: str = "minutes") -> pd.DataFrame | None:
        """
        Fetch intraday bars for the current session (V3).
        Endpoint: /v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}
        
        Enhanced with request/response logging and correlation IDs.
        """
        correlation_id = str(uuid.uuid4())
        
        if not self.is_authenticated:
            logger.warning(f"Upstox fetch_intraday_candles: Not authenticated (correlation_id={correlation_id})")
            return None

        # Clean interval (e.g. '1minute' -> '1', 'day' -> '1')
        num_interval = "".join(filter(str.isdigit, interval)) or "1"
        
        # FIXED: Correct URL format is /v3/historical-candle/intraday/{key}/{unit}/{interval}
        encoded_key = requests.utils.quote(instrument_key, safe='')
        url = f"{self.V3_URL}/historical-candle/intraday/{encoded_key}/{unit}/{num_interval}"
        
        # Enhanced request logging
        logger.debug(
            f"Upstox fetch_intraday_candles request: {url} "
            f"(correlation_id={correlation_id}, key={instrument_key})"
        )
        
        try:
            resp = requests.get(url, headers=self._headers(), timeout=8)
            
            # Response logging
            logger.debug(
                f"Upstox fetch_intraday_candles response: status={resp.status_code} "
                f"(correlation_id={correlation_id})"
            )
            
            if resp.status_code == 200:
                candles = resp.json().get("data", {}).get("candles", [])
                logger.info(f"Fetched {len(candles)} intraday candles for {instrument_key} (correlation_id={correlation_id})")
                return self._candles_to_df(candles)
            else:
                self._handle_api_error(resp, "intraday", request_params={"instrument_key": instrument_key, "interval": interval, "unit": unit})
        except requests.exceptions.ConnectionError as e:
            error_payload = create_error_payload(
                endpoint="intraday",
                method="GET",
                exception=e,
                request_params={"instrument_key": instrument_key, "interval": interval, "unit": unit},
                correlation_id=correlation_id
            )
            logger.error(f"Upstox fetch_intraday_candles connection error: {json.dumps(error_payload)}")
        except requests.exceptions.Timeout as e:
            error_payload = create_error_payload(
                endpoint="intraday",
                method="GET",
                exception=e,
                request_params={"instrument_key": instrument_key, "interval": interval, "unit": unit},
                correlation_id=correlation_id
            )
            logger.error(f"Upstox fetch_intraday_candles timeout: {json.dumps(error_payload)}")
        except Exception as e:
            error_payload = create_error_payload(
                endpoint="intraday",
                method="GET",
                exception=e,
                request_params={"instrument_key": instrument_key, "interval": interval, "unit": unit},
                correlation_id=correlation_id
            )
            logger.error(f"Upstox fetch_intraday_candles exception: {json.dumps(error_payload)}")
        return None

    @rate_limit_handling
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
        
        Enhanced with request/response logging and correlation IDs.
        """
        correlation_id = str(uuid.uuid4())
        
        if not self.is_authenticated:
            logger.warning(f"Upstox fetch_historical_candles: Not authenticated (correlation_id={correlation_id})")
            return None

        # V3 requires to_date before from_date
        to_date = datetime.today().strftime("%Y-%m-%d")
        from_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        num_interval = "".join(filter(str.isdigit, interval)) or "1"
        encoded_key = requests.utils.quote(instrument_key, safe='')
        
        # FIXED: Correct URL format is /v3/historical-candle/{key}/{unit}/{interval}/{to_date}/{from_date}
        # SKEPTIC: For indices, use V2 endpoint instead of V3 (V3 returns 400 error for indices)
        if "INDEX" in instrument_key:
            url = f"{self.BASE_URL}/historical-candle/{encoded_key}/{unit}/{num_interval}/{to_date}/{from_date}"
        else:
            url = f"{self.V3_URL}/historical-candle/{encoded_key}/{unit}/{num_interval}/{to_date}/{from_date}"
        
        logger.debug(
            f"SKEPTIC: Upstox Historical Call for {instrument_key} using "
            f"{'V2' if 'INDEX' in instrument_key else 'V3'} (correlation_id={correlation_id})"
        )
        
        try:
            resp = requests.get(url, headers=self._headers(), timeout=8)
            
            # Response logging
            logger.debug(
                f"Upstox fetch_historical_candles response: status={resp.status_code} "
                f"(correlation_id={correlation_id})"
            )
            
            if resp.status_code == 200:
                candles = resp.json().get("data", {}).get("candles", [])
                logger.info(f"Fetched {len(candles)} historical candles for {instrument_key} (correlation_id={correlation_id})")
                return self._candles_to_df(candles)
            elif resp.status_code == 400 and "INDEX" in instrument_key:
                # SKEPTIC: Known restriction for some indices on Upstox
                logger.warning(
                    f"SKEPTIC: Upstox Historical Data restricted for {instrument_key} (400). "
                    f"Skipping. (correlation_id={correlation_id})"
                )
            else:
                self._handle_api_error(resp, "historical", request_params={
                    "instrument_key": instrument_key,
                    "unit": unit,
                    "interval": interval,
                    "days": days
                })
        except requests.exceptions.ConnectionError as e:
            error_payload = create_error_payload(
                endpoint="historical",
                method="GET",
                exception=e,
                request_params={"instrument_key": instrument_key, "unit": unit, "interval": interval, "days": days},
                correlation_id=correlation_id
            )
            logger.error(f"Upstox fetch_historical_candles connection error: {json.dumps(error_payload)}")
        except requests.exceptions.Timeout as e:
            error_payload = create_error_payload(
                endpoint="historical",
                method="GET",
                exception=e,
                request_params={"instrument_key": instrument_key, "unit": unit, "interval": interval, "days": days},
                correlation_id=correlation_id
            )
            logger.error(f"Upstox fetch_historical_candles timeout: {json.dumps(error_payload)}")
        except Exception as e:
            error_payload = create_error_payload(
                endpoint="historical",
                method="GET",
                exception=e,
                request_params={"instrument_key": instrument_key, "unit": unit, "interval": interval, "days": days},
                correlation_id=correlation_id
            )
            logger.error(f"Upstox fetch_historical_candles exception: {json.dumps(error_payload)}")
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
            # More descriptive error logging
            logger.warning(f"Upstox: Could not resolve instrument key for ticker: {ticker}. Instrument cache may need refresh.")
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
            else:
                logger.debug(f"Upstox: No intraday data for {ticker} ({key})")

        # Historical data for prior days
        if days >= 1:
            hist = self.fetch_historical_candles(key, unit, interval, days=days)
            if hist is not None and not hist.empty:
                frames.append(hist)
            else:
                logger.debug(f"Upstox: No historical data for {ticker} ({key})")

        if not frames:
            # More descriptive error logging
            logger.warning(f"Upstox: No data returned for {ticker} ({key}). Token valid: {self.is_authenticated}")
            return None

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.dropna(inplace=True)
        return combined

    @rate_limit_handling
    def search_instruments(
        self,
        query: str = None,
        exchanges: list = None,
        segments: list = None,
        instrument_types: list = None,
        expiry: str = None,
        page_number: int = 1,
        records: int = 100
    ) -> list | None:
        """
        Search for instruments using Upstox Search API.
        
        Endpoint: GET https://api.upstox.com/v2/instruments/search
        
        Args:
            query: Search query (e.g., 'RELIANCE', 'NIFTY')
            exchanges: List of exchanges to search (e.g., ['NSE', 'BSE', 'MCX'])
            segments: List of segments (e.g., ['EQ', 'FUT', 'OPT'])
            instrument_types: List of instrument types (e.g., ['stock', 'index', 'futures', 'options'])
            expiry: Expiry date for derivatives (e.g., '2024-12-26')
            page_number: Page number for pagination (default: 1)
            records: Number of records per page (default: 100, max: 100)
        
        Returns:
            List of matching instruments with instrument_key, trading_symbol, etc.
            
        Enhanced with request/response logging and correlation IDs.
        """
        correlation_id = str(uuid.uuid4())
        
        if not self.is_authenticated:
            logger.warning(f"Upstox search_instruments: Not authenticated (correlation_id={correlation_id})")
            return None

        # Build cache key for search results
        cache_params = (query, tuple(exchanges) if exchanges else None, tuple(segments) if segments else None)
        cache_key = str(cache_params)
        
        # Check cache first
        now = time.time()
        if cache_key in _search_cache:
            cache_age = now - _search_cache_time.get(cache_key, 0)
            if cache_age < SEARCH_CACHE_TTL:
                logger.debug(f"Returning cached search results for: {query} (correlation_id={correlation_id})")
                return _search_cache[cache_key]

        params = {}
        if query:
            params["q"] = query
        if exchanges:
            params["exchanges"] = ",".join(exchanges)
        if segments:
            params["segments"] = ",".join(segments)
        if instrument_types:
            params["instrument_types"] = ",".join(instrument_types)
        if expiry:
            params["expiry"] = expiry
        params["page_number"] = page_number
        params["records"] = min(records, 100)  # Cap at 100

        # Request logging
        logger.debug(
            f"Upstox search_instruments request: query='{query}', exchanges={exchanges}, "
            f"segments={segments} (correlation_id={correlation_id})"
        )
        
        try:
            url = self.SEARCH_URL
            resp = requests.get(url, headers=self._headers(), params=params, timeout=10)
            
            # Response logging
            logger.debug(
                f"Upstox search_instruments response: status={resp.status_code}, "
                f"query='{query}' (correlation_id={correlation_id})"
            )
            
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("data", [])
                
                # Cache the results
                _search_cache[cache_key] = results
                _search_cache_time[cache_key] = now
                
                logger.info(
                    f"Search API found {len(results)} instruments for query: '{query}' "
                    f"(correlation_id={correlation_id})"
                )
                return results
            else:
                self._handle_api_error(resp, "search_instruments", request_params=params)
        except requests.exceptions.ConnectionError as e:
            error_payload = create_error_payload(
                endpoint="search_instruments",
                method="GET",
                exception=e,
                request_params=params,
                correlation_id=correlation_id
            )
            logger.error(f"Upstox search_instruments connection error: {json.dumps(error_payload)}")
        except requests.exceptions.Timeout as e:
            error_payload = create_error_payload(
                endpoint="search_instruments",
                method="GET",
                exception=e,
                request_params=params,
                correlation_id=correlation_id
            )
            logger.error(f"Upstox search_instruments timeout: {json.dumps(error_payload)}")
        except Exception as e:
            error_payload = create_error_payload(
                endpoint="search_instruments",
                method="GET",
                exception=e,
                request_params=params,
                correlation_id=correlation_id
            )
            logger.error(f"Upstox search_instruments exception: {json.dumps(error_payload)}")
        
        return None

    @rate_limit_handling
    def fetch_market_quote(self, instrument_key: str) -> dict | None:
        """
        Fetch real-time snapshot quote using REST API (Hybrid V2/V3). 
        Returns V2-compatible structure. Supports comma-separated keys.
        
        Enhanced with request/response logging and correlation IDs.
        """
        correlation_id = str(uuid.uuid4())
        
        if not self.is_authenticated:
            logger.warning(f"Upstox fetch_market_quote: Not authenticated (correlation_id={correlation_id})")
            return None

        # Ensure instrument cache is loaded for key lookups
        _load_instrument_cache()

        # SKEPTIC: Explicitly use V2 endpoint for quotes as V3 is currently unstable or restricted.
        encoded_key = requests.utils.quote(instrument_key, safe='')
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_key}"
        
        # Request logging
        logger.debug(
            f"Upstox fetch_market_quote request: keys={instrument_key} "
            f"(correlation_id={correlation_id})"
        )
        
        try:
            resp = requests.get(url, headers=self._headers(), timeout=5)
            
            # Response logging
            logger.debug(
                f"Upstox fetch_market_quote response: status={resp.status_code} "
                f"(correlation_id={correlation_id})"
            )
            
            if resp.status_code == 200:
                raw_data = resp.json().get("data", {})

                normalized_data = {}
                keys_requested = instrument_key.split(",")
                for key in keys_requested:
                    # Try multiple key formats that Upstox might return
                    # Format 1: NSE_INDEX|Sensex
                    # Format 2: NSE_INDEX:Sensex
                    # Format 3: Sensex
                    # Format 4: NSE_INDEX|Nifty 50 -> may be returned as just "Nifty 50"
                    # Format 5: NSE_EQ:TCS (exchange:symbol) - NEW: Upstox API now returns this format instead of ISIN-based
                    
                    # Get the symbol from the instrument cache for this key
                    symbol = None
                    # Try to find the symbol using the reverse cache
                    if key in _reverse_instrument_cache:
                        symbol = _reverse_instrument_cache[key]
                    else:
                        # Also try with colon separator
                        colon_key = key.replace("|", ":")
                        if colon_key in _reverse_instrument_cache:
                            symbol = _reverse_instrument_cache[colon_key]
                        else:
                            # FIXED: Try to find by ISIN - extract ISIN from key and search in cache
                            # ISIN is the part after | (e.g., INE467B01029 from NSE_EQ|INE467B01029)
                            isin = key.split("|")[-1] if "|" in key else None
                            if isin and isin in _reverse_instrument_cache:
                                symbol = _reverse_instrument_cache[isin]
                                logger.debug(f"Upstox: Found symbol '{symbol}' via ISIN direct lookup for key '{key}'")
                            elif isin:
                                # Search through cache for this ISIN
                                for cache_key, cache_symbol in _reverse_instrument_cache.items():
                                    if isin in cache_key:
                                        symbol = cache_symbol
                                        logger.debug(f"Upstox: Found symbol '{symbol}' via ISIN partial match for key '{key}'")
                                        break
                    
                    # Extract exchange prefix (e.g., "NSE_EQ" from "NSE_EQ|INE467B01029")
                    exchange_prefix = key.split("|")[0] if "|" in key else (key.split(":")[0] if ":" in key else "")
                    
                    formats_to_try = [
                        key,
                        key.replace("|", ":"),
                        key.replace("|", ""),
                        key.split("|")[-1] if "|" in key else key,  # Just the name part (ISIN)
                        key.split("|")[-1].replace(" ", "") if "|" in key else key.replace(" ", ""),
                    ]
                    
                    # Add the symbol-based format (e.g., "NSE_EQ:TCS") that Upstox API now returns
                    if symbol and exchange_prefix:
                        symbol_format = f"{exchange_prefix}:{symbol}"
                        formats_to_try.append(symbol_format)
                        logger.debug(f"Upstox: Added symbol-based format '{symbol_format}' for key '{key}'")
                    target_key = None
                    for fmt in formats_to_try:
                        if fmt in raw_data:
                            target_key = fmt
                            break
                    
                    if target_key:
                        normalized_data[key] = raw_data[target_key]
                        logger.debug(f"Upstox: Found {key} as {target_key}")
                    else:
                        # Known indices that may not be available in Upstox - use exact matching
                        key_name = key.split('|')[-1].split(':')[-1].strip()
                        if key_name in ['Nifty Midcap 100', 'Nifty Smallcap 100']:
                            logger.info(f"Upstox: Index '{key_name}' is not available (known limitation). Skipping.")
                        else:
                            logger.warning(f"Upstox: {key} not found in quote data. Available keys: {list(raw_data.keys())[:5]}")

                if normalized_data:
                    logger.info(f"Fetched quotes for {len(normalized_data)} instruments (correlation_id={correlation_id})")
                    return {
                        "status": "success",
                        "data": normalized_data
                    }
            else:
                if resp.status_code == 404:
                    logger.error(
                        f"SKEPTIC: 404 for keys: {instrument_key} | URL: {url} "
                        f"(correlation_id={correlation_id})"
                    )
                self._handle_api_error(resp, "market_quote", request_params={"instrument_key": instrument_key})
        except requests.exceptions.ConnectionError as e:
            error_payload = create_error_payload(
                endpoint="market_quote",
                method="GET",
                exception=e,
                request_params={"instrument_key": instrument_key},
                correlation_id=correlation_id
            )
            logger.error(f"Upstox fetch_market_quote connection error: {json.dumps(error_payload)}")
        except requests.exceptions.Timeout as e:
            error_payload = create_error_payload(
                endpoint="market_quote",
                method="GET",
                exception=e,
                request_params={"instrument_key": instrument_key},
                correlation_id=correlation_id
            )
            logger.error(f"Upstox fetch_market_quote timeout: {json.dumps(error_payload)}")
        except Exception as e:
            error_payload = create_error_payload(
                endpoint="market_quote",
                method="GET",
                exception=e,
                request_params={"instrument_key": instrument_key},
                correlation_id=correlation_id
            )
            logger.error(f"Upstox fetch_market_quote exception: {json.dumps(error_payload)}")
        return None


# Cached data for graceful degradation
_market_quote_cache = {}
_market_quote_cache_time = {}
CACHE_TTL_SECONDS = 60  # Cache quotes for 60 seconds

def get_cached_quote(instrument_key: str) -> dict | None:
    """Get cached quote if available and not expired."""
    global _market_quote_cache, _market_quote_cache_time
    
    if instrument_key in _market_quote_cache:
        cache_age = time.time() - _market_quote_cache_time.get(instrument_key, 0)
        if cache_age < CACHE_TTL_SECONDS:
            return _market_quote_cache[instrument_key]
    
    return None


def set_cached_quote(instrument_key: str, quote_data: dict) -> None:
    """Cache a quote for graceful degradation."""
    global _market_quote_cache, _market_quote_cache_time
    _market_quote_cache[instrument_key] = quote_data
    _market_quote_cache_time[instrument_key] = time.time()


def get_quote_with_fallback(instrument_key: str) -> dict | None:
    """
    Get quote with graceful degradation: try live, fallback to cache.
    Returns quote data and a flag indicating if it's from cache (stale).
    """
    # Try live first
    if not is_rate_limited():
        quote = upstox_client.fetch_market_quote(instrument_key)
        if quote and quote.get("status") == "success":
            # Cache the successful result
            set_cached_quote(instrument_key, quote)
            return {"data": quote.get("data"), "is_stale": False}
    
    # Fallback to cache if rate limited or request failed
    cached = get_cached_quote(instrument_key)
    if cached:
        logger.info(f"Using cached quote for {instrument_key} (rate limited or API error)")
        return {"data": cached.get("data"), "is_stale": True}
    
    return None


upstox_client = UpstoxService()
