"""
YFinance Helper Module - Optimized for intraday trading.

Features:
- Aggressive caching (5-15 minutes for intraday data)
- Rate limiting (max 1-2 requests per minute)
- Exponential backoff with max retries
- Batch requests when possible
- Quota awareness

Priority: Upstox should ALWAYS be used first for NSE data, yfinance as fallback only.
"""

import yfinance as yf
import pandas as pd
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict
from functools import wraps

# SKEPTIC: Use curl_cffi for proper browser fingerprinting to avoid Yahoo rate limiting
# Simply setting headers is NOT enough - Yahoo now checks TLS fingerprints
try:
    from curl_cffi import requests as curl_requests
    _USE_CURL_CFFI = True
except ImportError:
    _USE_CURL_CFFI = False
    import requests

def get_yfinance_session():
    """Create a session that impersonates a browser to avoid Yahoo rate limiting.
    
    CRITICAL: Yahoo now checks TLS fingerprints. Simply setting User-Agent header is NOT enough.
    We must use curl_cffi for proper browser fingerprinting.
    """
    if _USE_CURL_CFFI:
        # Use curl_cffi for proper browser fingerprinting
        session = curl_requests.Session(impersonate="chrome")
        return session
    else:
        # Fallback to requests if curl_cffi not installed
        import requests
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        return session

# Global session for yfinance requests
_yfinance_session = get_yfinance_session()

# Log warning if curl_cffi is not available
if not _USE_CURL_CFFI:
    logger.error("CRITICAL: curl_cffi not installed! Yahoo rate limiting WILL occur without proper TLS fingerprinting. Install with: pip install curl_cffi")

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_TTL_INTRADAY = 5 * 60  # 5 minutes for intraday (5m, 15m, 1h)
CACHE_TTL_DAILY = 15 * 60     # 15 minutes for daily data
CACHE_TTL_POSITIONS = 60 * 60 # 1 hour for positions/holdings data

# Max cache age for rate limit fallback (in seconds)
# If cache is older than this when rate limited, return None instead of stale data
MAX_CACHE_AGE_RATE_LIMIT = 300  # 5 minutes

# Rate limiting
# SKEPTIC: Increased from 2 to 30 requests per minute to match yfinance's ~2000/hour limit
# This allows faster data fetching while still respecting rate limits
MAX_REQUESTS_PER_MINUTE = 30
REQUEST_COOLDOWN = 2  # seconds between requests (reduced from 30 for faster fetching)

# Cache size limits to prevent unbounded growth
MAX_CACHE_SIZE = 200
CLEANUP_THRESHOLD = 150


class YFinanceCache:
    """Thread-safe cache for yfinance data."""
    
    def __init__(self):
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._lock = threading.RLock()
    
    def _maybe_cleanup(self):
        """Remove oldest expired entries when cache exceeds size limit."""
        if len(self._cache) >= MAX_CACHE_SIZE:
            current_time = datetime.now()
            # Remove expired entries
            expired_keys = [
                k for k, (ts, _) in self._cache.items()
                if (current_time - ts).total_seconds() >= self._get_ttl(k)
            ]
            for key in expired_keys:
                del self._cache[key]
            
            # If still over limit, remove oldest entries
            if len(self._cache) >= CLEANUP_THRESHOLD:
                sorted_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][0]
                )
                for key in sorted_keys[:len(self._cache) - MAX_CACHE_SIZE + 20]:
                    del self._cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        with self._lock:
            if key in self._cache:
                timestamp, data = self._cache[key]
                ttl = self._get_ttl(key)
                if (datetime.now() - timestamp).total_seconds() < ttl:
                    logger.debug(f"Cache HIT: {key}")
                    return data
                else:
                    del self._cache[key]
                    logger.debug(f"Cache EXPIRED: {key}")
        return None
    
    def get_age(self, key: str) -> Optional[int]:
        """Get age of cached entry in seconds. Returns None if key not in cache."""
        with self._lock:
            if key in self._cache:
                timestamp, _ = self._cache[key]
                return int((datetime.now() - timestamp).total_seconds())
        return None
    
    def set(self, key: str, data: Any):
        """Set cached value with current timestamp."""
        with self._lock:
            self._maybe_cleanup()
            self._cache[key] = (datetime.now(), data)
            logger.debug(f"Cache SET: {key}")
    
    def _get_ttl(self, key: str) -> int:
        """Determine TTL based on data type."""
        if "interval=1m" in key or "interval=5m" in key or "interval=15m" in key:
            return CACHE_TTL_INTRADAY
        elif "interval=1h" in key:
            return CACHE_TTL_INTRADAY
        elif "interval=1d" in key:
            return CACHE_TTL_DAILY
        return CACHE_TTL_DAILY
    
    def clear(self):
        """Clear all cache."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            now = datetime.now()
            expired = 0
            for key, (ts, _) in self._cache.items():
                ttl = self._get_ttl(key)
                if (now - ts).total_seconds() >= ttl:
                    expired += 1
            return {
                "total_entries": len(self._cache),
                "expired_entries": expired,
                "active_entries": len(self._cache) - expired
            }


class RateLimiter:
    """Rate limiter for yfinance requests."""
    
    def __init__(self, max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE):
        self._max_requests = max_requests_per_minute
        self._requests: List[float] = []
        self._lock = threading.RLock()
    
    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire permission to make a request."""
        with self._lock:
            now = time.time()
            # Remove requests older than 1 minute
            self._requests = [ts for ts in self._requests if now - ts < 60]
            
            if len(self._requests) < self._max_requests:
                self._requests.append(now)
                return True
            
            # Wait for oldest request to expire
            wait_time = 60 - (now - self._requests[0])
            if wait_time > timeout:
                logger.warning(f"Rate limit exceeded. Need to wait {wait_time:.1f}s")
                return False
            
            time.sleep(min(wait_time, timeout))
            self._requests.append(time.time())
            return True
    
    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        with self._lock:
            now = time.time()
            self._requests = [ts for ts in self._requests if now - ts < 60]
            return max(0, self._max_requests - len(self._requests))


# Global instances
_cache = YFinanceCache()
_rate_limiter = RateLimiter()


def with_rate_limit_and_cache(ttl_override: Optional[int] = None):
    """Decorator for functions that need rate limiting and caching."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache first
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Acquire rate limit permission
            if not _rate_limiter.acquire():
                # Check if cache is not too stale (within MAX_CACHE_AGE_RATE_LIMIT)
                cache_age = _cache.get_age(cache_key) if cached_result is not None else None
                if cached_result is not None and cache_age is not None and cache_age <= MAX_CACHE_AGE_RATE_LIMIT:
                    logger.warning(f"Rate limit timeout for {func.__name__}, returning cached data (age: {cache_age}s)")
                    return cached_result
                else:
                    cache_age_str = f"{cache_age}s" if cache_age is not None else "unavailable"
                    logger.error(f"Rate limit timeout for {func.__name__}, cache too stale or unavailable (age: {cache_age_str})")
                    return None
            
            # Execute with retries
            max_retries = 3
            base_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    # Cache successful results
                    if result is not None and not (isinstance(result, pd.DataFrame) and result.empty):
                        _cache.set(cache_key, result)
                    return result
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"yfinance error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"yfinance error after {max_retries} attempts: {e}")
                        raise
        
        return wrapper
    return decorator


def is_market_open() -> bool:
    """Check if NSE market is currently open (IST timezone)."""
    import pytz
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Market hours: 9:15 AM to 3:30 PM IST, Monday-Friday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close


@with_rate_limit_and_cache()
def fetch_ohlcv_cached(ticker: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """
    Fetch OHLCV data with caching and rate limiting.
    
    Args:
        ticker: Stock ticker symbol (e.g., "RELIANCE.NS")
        period: Period to fetch (e.g., "5d", "1mo")
        interval: Data interval (e.g., "5m", "15m", "1d")
    
    Returns:
        DataFrame with OHLCV data
    """
    yf_ticker = ticker if ticker.endswith(".NS") or "^" in ticker or "=" in ticker or ticker.startswith(("CL", "BZ", "GC", "SI")) else f"{ticker}.NS"
    
    try:
        df = yf.download(
            yf_ticker, 
            period=period, 
            interval=interval, 
            auto_adjust=True, 
            progress=False,
            threads=True,  # Enable multi-threading
            session=_yfinance_session  # Use browser session to avoid rate limiting
        )
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.dropna(inplace=True)
        
        if not df.empty:
            df.columns = [str(c).capitalize() for c in df.columns]
            logger.info(f"Fetched {len(df)} rows for {ticker} via yfinance")
        
        return df
    except Exception as e:
        logger.error(f"yfinance fetch error for {ticker}: {e}")
        return pd.DataFrame()


@with_rate_limit_and_cache()
def fetch_quote_cached(ticker: str) -> Optional[Dict]:
    """
    Fetch real-time quote with caching.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with quote data
    """
    yf_ticker = ticker if ticker.endswith(".NS") or "^" in ticker or "=" in ticker or ticker.startswith(("CL", "BZ", "GC", "SI")) else f"{ticker}.NS"
    
    try:
        t = yf.Ticker(yf_ticker, session=_yfinance_session)
        info = t.info
        
        if info and 'regularMarketPrice' in info:
            return {
                "symbol": ticker,
                "price": info.get('regularMarketPrice'),
                "previous_close": info.get('regularMarketPreviousClose'),
                "open": info.get('regularMarketOpen'),
                "high": info.get('regularMarketDayHigh'),
                "low": info.get('regularMarketDayLow'),
                "volume": info.get('regularMarketVolume'),
                "change": info.get('regularMarketChange'),
                "change_pct": info.get('regularMarketChangePercent')
            }
    except Exception as e:
        logger.error(f"yfinance quote error for {ticker}: {e}")
    
    return None


@with_rate_limit_and_cache()
def fetch_history_cached(ticker: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical data with caching.
    
    Args:
        ticker: Stock ticker symbol
        period: Period to fetch
        interval: Data interval
    
    Returns:
        DataFrame with historical data
    """
    yf_ticker = ticker if ticker.endswith(".NS") or "^" in ticker or "=" in ticker or ticker.startswith(("CL", "BZ", "GC", "SI")) else f"{ticker}.NS"
    
    try:
        t = yf.Ticker(yf_ticker, session=_yfinance_session)
        hist = t.history(period=period, interval=interval)
        
        if not hist.empty:
            logger.info(f"Fetched {len(hist)} rows of history for {ticker} via yfinance")
        
        return hist
    except Exception as e:
        logger.error(f"yfinance history error for {ticker}: {e}")
        return pd.DataFrame()


@with_rate_limit_and_cache()
def fetch_fundamentals_cached(ticker: str) -> Dict:
    """
    Fetch fundamental data with caching.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with fundamental data
    """
    yf_ticker = ticker if ticker.endswith(".NS") or "^" in ticker or "=" in ticker or ticker.startswith(("CL", "BZ", "GC", "SI")) else f"{ticker}.NS"
    
    try:
        t = yf.Ticker(yf_ticker, session=_yfinance_session)
        info = t.info
        
        return {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "dividend_yield": info.get("dividendYield"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
            "eps": info.get("trailingEps"),
            "book_value": info.get("bookValue"),
            "price_to_book": info.get("priceToBook"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity")
        }
    except Exception as e:
        logger.error(f"yfinance fundamentals error for {ticker}: {e}")
        return {}


def batch_fetch_quotes(tickers: List[str], max_batch: int = 10) -> Dict[str, Dict]:
    """
    Batch fetch quotes for multiple tickers efficiently.
    
    Args:
        tickers: List of ticker symbols
        max_batch: Maximum batch size
    
    Returns:
        Dictionary mapping ticker to quote data
    """
    results = {}
    yf_tickers = [t if t.endswith(".NS") or "^" in t or "=" in t or t.startswith(("CL", "BZ", "GC", "SI")) else f"{t}.NS" for t in tickers]
    
    # Use yfinance batch download
    try:
        data = yf.download(
            yf_tickers[:max_batch], 
            period="5d", 
            interval="1d", 
            group_by='ticker',
            progress=False,
            threads=True,
            session=_yfinance_session  # Use browser session to avoid rate limiting
        )
        
        for ticker in tickers[:max_batch]:
            yf_ticker = ticker if ticker.endswith(".NS") or "^" in ticker or "=" in ticker or ticker.startswith(("CL", "BZ", "GC", "SI")) else f"{ticker}.NS"
            try:
                if yf_ticker in data.columns.get_level_values(0):
                    t_data = data[yf_ticker]
                    if not t_data.empty:
                        current = t_data['Close'].iloc[-1]
                        prev = t_data['Close'].iloc[-2] if len(t_data) > 1 else current
                        change = current - prev if prev != 0 else 0
                        
                        results[ticker] = {
                            "symbol": ticker,
                            "price": float(current) if not pd.isna(current) else None,
                            "change": float(change),
                            "change_pct": float((change / prev * 100)) if prev != 0 and not pd.isna(prev) else 0,
                            "volume": int(t_data['Volume'].iloc[-1]) if 'Volume' in t_data else 0
                        }
            except Exception as e:
                logger.warning(f"Error processing {ticker} from batch: {e}")
    
    except Exception as e:
        logger.error(f"yfinance batch fetch error: {e}")
    
    return results


def clear_cache():
    """Clear the yfinance cache."""
    _cache.clear()
    logger.info("yfinance cache cleared")


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    return _cache.get_stats()


def get_rate_limit_status() -> Dict:
    """Get rate limiter status."""
    return {
        "remaining_requests": _rate_limiter.get_remaining(),
        "max_requests_per_minute": MAX_REQUESTS_PER_MINUTE
    }
