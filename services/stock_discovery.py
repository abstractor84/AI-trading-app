import logging
import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from services.upstox_service import upstox_client, get_instrument_key
from services.yfinance_helper import (
    fetch_history_cached,
    batch_fetch_quotes,
    is_market_open,
    clear_cache,
    get_cache_stats,
    _cache as yfinance_cache
)
from services.quota_service import quota_svc

logger = logging.getLogger(__name__)

# Flag to track if we're using yfinance fallback
_using_yfinance_fallback = False

# Core NSE universe for discovery
_NIFTY100_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "BHARTIARTL",
    "INFY", "SBIN", "LICI", "ITC", "HINDUNILVR",
    "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA",
    "ADANIENT", "KOTAKBANK", "TITAN", "ONGC", "TMPV",
    "NTPC", "AXISBANK", "ADANIPORTS", "ASIANPAINT", "COALINDIA",
    "BAJAJHLDNG", "BAJAJ-AUTO", "BHEL", "JSWSTEEL", "ADANIPOWER",
    "M&M", "TRENT", "BEL", "ULTRACEMCO", "SIEMENS",
    "GRASIM", "SBILIFE", "BPCL", "NESTLEIND", "HAL",
    "ETERNAL", "PNB", "IOB", "CANBK", "INDHOTEL",
    "DLF", "EICHERMOT", "HINDALCO", "DIVISLAB", "CIPLA",
    "BRITANNIA", "TATASTEEL", "ADANIENSOL"
]
_NIFTY100_NS = [f"{s}.NS" for s in _NIFTY100_SYMBOLS]


class StockDiscoveryService:
    # Class-level cache for global indices to reduce yfinance calls
    _global_indices_cache = {"data": None, "timestamp": None}
    _CACHE_TTL_SECONDS = 900  # 15 minutes cache for global indices (they change slowly)
    
    # Map of Indian indices to Upstox instrument keys
    # CRITICAL FIX: Upstox API requires PIPE format (NSE_INDEX|Nifty 50), not colon
    # POC verified keys - case sensitive! SENSEX not Sensex, NIFTY MIDCAP not Nifty Midcap
    # Available indices in Upstox (from instrument search):
    # - NSE_INDEX: Nifty 50, Nifty Bank, Nifty Fin Service, NIFTY MIDCAP 100, NIFTY SMLCAP 100, etc.
    # - BSE_INDEX: SENSEX (CAPS), BSE 100, BSE 200, etc.
    _UPSTOX_INDICES = {
        "GIFT Nifty": "NSE_INDEX|Nifty 50",  # GIFT Nifty not available in Upstox REST API - use Nifty 50 as proxy for pre-market
        "Nifty 50": "NSE_INDEX|Nifty 50",
        "Sensex": "BSE_INDEX|SENSEX",  # MUST BE CAPS!
        "Bank Nifty": "NSE_INDEX|Nifty Bank",
        "FinNifty": "NSE_INDEX|Nifty Fin Service",
        "Nifty Midcap 100": "NSE_INDEX|NIFTY MIDCAP 100",  # MUST BE CAPS!
        "Nifty Smallcap 100": "NSE_INDEX|NIFTY SMLCAP 100",  # MUST BE CAPS!
    }
    
    # India VIX - can be fetched from Upstox
    _UPSTOX_VIX_KEY = "NSE_INDEX|India VIX"
    
    # Indices known to be unavailable in Upstox - use yfinance fallback
    # Note: Midcap 100 and Smallcap 100 ARE now available with correct keys (NIFTY MIDCAP 100, NIFTY SMLCAP 100)
    # GIFT Nifty is not available in Upstox REST API - use yfinance
    _UPSTOX_UNAVAILABLE_INDICES = {
        "GIFT Nifty": "^NIFTY",  # Yahoo ticker for GIFT Nifty (pre-market indicator)
    }
    
    def __init__(self):
        self.universe = list(_NIFTY100_NS)  # Start with hardcoded list immediately

    def fetch_global_indices(self):
        """Fetches the latest prices and formats them into categorized HTML rows.
        
        IMPROVED: Uses Upstox for all Indian indices when available.
        Uses caching to minimize API calls.
        """
        global _using_yfinance_fallback
        import time
        
        # Check cache first (15 minute TTL for global indices)
        now = time.time()
        if (self._global_indices_cache["data"] is not None and 
            self._global_indices_cache["timestamp"] is not None and
            now - self._global_indices_cache["timestamp"] < self._CACHE_TTL_SECONDS):
            logger.debug("Returning cached global indices")
            return self._global_indices_cache["data"]
        
        symbols = {
            "GLOBAL": {
                "USD/INR": "INR=X",
                "WTI Crude": "CL=F",
                "Brent Crude": "BZ=F", # Near-month futures
                "Gold": "GC=F",
                "Silver": "SI=F",
                "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW 30": "^DJI",
                "DAX": "^GDAXI", "FTSE 100": "^FTSE", "NIKKEI": "^N225"
            },
            "INDIA": {
                # These use Upstox keys - will be fetched via Upstox first
                # CRITICAL FIX: Use correct keys (case-sensitive!) from POC verification
                "GIFT Nifty": "NSE_INDEX|Nifty 50",  # Use Nifty 50 as proxy
                "Nifty 50": "NSE_INDEX|Nifty 50",
                "Sensex": "BSE_INDEX|SENSEX",  # MUST BE CAPS!
                "Bank Nifty": "NSE_INDEX|Nifty Bank",
                "FinNifty": "NSE_INDEX|Nifty Fin Service",
                "Nifty Midcap 100": "NSE_INDEX|NIFTY MIDCAP 100",  # MUST BE CAPS!
                "Nifty Smallcap 100": "NSE_INDEX|NIFTY SMLCAP 100",  # MUST BE CAPS!
            }
        }

        output = {"global": {}, "india": {}, "vix": {"value": 0, "change": 0, "change_pct": 0}}
        
        # First, try to fetch ALL Indian indices from Upstox in one batch
        from services.upstox_service import upstox_client
        upstox_indices_keys = [v for k, v in self._UPSTOX_INDICES.items()]
        
        if upstox_client.is_authenticated and upstox_indices_keys:
            try:
                keys_str = ",".join(upstox_indices_keys)
                q = upstox_client.fetch_market_quote(keys_str)
                if q and 'data' in q:
                    raw_data = q['data']
                    for name, upstox_key in self._UPSTOX_INDICES.items():
                        # SKEPTIC: Try multiple key formats that Upstox might return
                        # Format 1: NSE_INDEX|Nifty 50 (original)
                        # Format 2: NSE_INDEX:Nifty 50 (colon)
                        # Format 3: Just the name part (e.g., "Nifty 50")
                        # Format 4: Uppercase version (e.g., "NIFTY MIDCAP 100" -> "Nifty Midcap 100")
                        formats_to_try = [
                            upstox_key,
                            upstox_key.replace("|", ":"),
                            upstox_key.split("|")[-1],
                        ]
                        
                        # Also add case-normalized versions for indices like NIFTY MIDCAP 100
                        parts = upstox_key.split("|")
                        if len(parts) == 2 and parts[1] != parts[1].title():
                            # Try title case for the second part
                            formats_to_try.append(f"{parts[0]}|{parts[1].title()}")
                        
                        target_key = None
                        for fmt in formats_to_try:
                            if fmt in raw_data:
                                target_key = fmt
                                break
                        
                        # Case-insensitive search as fallback
                        if not target_key:
                            upstox_key_upper = upstox_key.upper()
                            for k in raw_data:
                                if k.upper() == upstox_key_upper:
                                    target_key = k
                                    break
                        
                        if target_key:
                            item = raw_data[target_key]
                            current = item.get('last_price', 0)
                            # Use net_change from API - it's the difference from previous close
                            net_change = item.get('net_change', 0)
                            # Calculate change_pct from net_change / previous close
                            prev_close = item.get('ohlc', {}).get('close', current)
                            change_pct = (net_change / prev_close * 100) if prev_close != 0 else 0
                            output["india"][name] = {
                                "name": name,
                                "value": round(current, 2),
                                "change": round(net_change, 2),
                                "change_pct": round(change_pct, 2)
                            }
                            logger.debug(f"Fetched {name} from Upstox: value={current}, change={net_change}, pct={change_pct:.2f}%")
            except Exception as e:
                logger.warning(f"Upstox batch fetch failed: {e}")
        
        # FALLBACK: Try yfinance for known unavailable Upstox indices
        # This handles cases where Upstox returns "not available" for Midcap/Smallcap indices
        for index_name, yf_ticker in self._UPSTOX_UNAVAILABLE_INDICES.items():
            # Skip if we already got it from Upstox
            if index_name in output["india"]:
                continue
            
            # Try yfinance fallback for this specific index
            try:
                if quota_svc.check_yfinance_quota():
                    # Check cache first
                    cache_key = f"fetch_history_cached:('{yf_ticker}',){{'interval': '1d', 'period': '5d'}}"
                    cached_data = yfinance_cache.get(cache_key)
                    
                    if cached_data is not None:
                        hist = cached_data
                        logger.debug(f"Cache hit for {index_name} ({yf_ticker})")
                    else:
                        hist = fetch_history_cached(yf_ticker, period="5d", interval="1d")
                        logger.info(f"Fetched fresh data for unavailable Upstox index: {index_name} ({yf_ticker})")
                    
                    if hist is not None and not hist.empty and len(hist) >= 1:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2] if len(hist) > 1 else 0
                        change = current - prev if prev != 0 else 0
                        change_pct = (change / prev * 100) if prev != 0 else 0
                        
                        # Only log yfinance usage when we actually get valid data
                        quota_svc.log_yfinance_usage()
                        _using_yfinance_fallback = True
                        
                        output["india"][index_name] = {
                            "name": index_name,
                            "value": round(float(current), 2) if not pd.isna(current) else 0,
                            "change": round(float(change), 2) if not pd.isna(change) else 0,
                            "change_pct": round(float(change_pct), 2) if not pd.isna(change_pct) else 0
                        }
                    else:
                        logger.warning(f"yfinance also unavailable for {index_name} ({yf_ticker})")
                else:
                    logger.warning(f"yfinance quota exhausted, cannot fetch {index_name}")
            except Exception as e:
                logger.warning(f"yfinance fallback failed for {index_name}: {e}")

        for category, ticker_map in symbols.items():
            for name, ticker in ticker_map.items():
                try:
                    data = None
                    
                    # Skip if we already got it from Upstox
                    if category == "INDIA" and name in output["india"]:
                        continue

                    # SKIP Indian indices with Upstox keys (NSE_ or BSE_ prefix) - don't try yfinance for these
                    if category == "INDIA" and (ticker.startswith("NSE_") or ticker.startswith("BSE_")):
                        logger.debug(f"Skipping yfinance for Indian index {name} - using Upstox only")
                        continue

                    # If data is still None, use yfinance fallback.
                    # PRIORITY: Upstox first, yfinance only as fallback
                    if data is None:
                        # Check if we have yfinance quota available
                        if quota_svc.check_yfinance_quota():
                            # Check cache FIRST to avoid logging for cache hits
                            # Cache key format in yfinance_helper: func_name:args:kwargs
                            cache_key = f"fetch_history_cached:('{ticker}',){{'interval': '1d', 'period': '5d'}}"
                            cached_data = yfinance_cache.get(cache_key)
                            
                            if cached_data is not None:
                                # Cache hit - use cached data without logging
                                hist = cached_data
                                logger.debug(f"Cache hit for {ticker}, using cached data")
                            else:
                                # Cache miss - fetch from yfinance (will make API call)
                                hist = fetch_history_cached(ticker, period="5d", interval="1d")
                                # Log usage only if we got valid data (done below)
                                logger.info(f"Fetched fresh data from yfinance for {name} ({ticker})")
                            
                            # Check if we got valid data
                            if hist is not None and not hist.empty and len(hist) >= 1:
                                current = hist['Close'].iloc[-1]
                                prev = hist['Close'].iloc[-2] if len(hist) > 1 else 0
                                change = current - prev if prev != 0 else 0
                                change_pct = (change / prev * 100) if prev != 0 else 0
                                
                                # Only log yfinance usage when we actually get valid data
                                quota_svc.log_yfinance_usage()
                                _using_yfinance_fallback = True
                                
                                data = {
                                    "value": round(float(current), 2) if not pd.isna(current) else 0,
                                    "change": round(float(change), 2) if not pd.isna(change) else 0,
                                    "change_pct": round(float(change_pct), 2) if not pd.isna(change_pct) else 0
                                }
                            else:
                                logger.warning(f"yfinance returned no data for {name} ({ticker})")
                        else:
                            logger.warning(f"yfinance quota exhausted, skipping {ticker}")
                    
                    if data:
                        if category == "GLOBAL":
                            output["global"][name] = data
                        else:
                            output["india"][name] = data
                except Exception as e:
                    logger.error(f"Error fetching {name} ({ticker}): {e}")

        # Manually attach VIX
        # PRIORITY: Upstox first, then yfinance fallback
        try:
            vix_data = None
            
            # Try Upstox first
            if upstox_client.is_authenticated:
                try:
                    vix_key = self._UPSTOX_VIX_KEY
                    q = upstox_client.fetch_market_quote(vix_key)
                    if q and q.get('status') == 'success' and q.get('data'):
                        # Find the VIX data - API returns with colon format in response
                        raw_data = q['data']
                        for k, v in raw_data.items():
                            if 'VIX' in k:
                                v_curr = v.get('last_price', 0)
                                net_change = v.get('net_change', 0)
                                prev_close = v.get('ohlc', {}).get('close', v_curr)
                                v_pct = (net_change / prev_close * 100) if prev_close != 0 else 0
                                vix_data = {
                                    "value": round(float(v_curr), 2),
                                    "change": round(float(net_change), 2),
                                    "change_pct": round(float(v_pct), 2)
                                }
                                logger.info(f"Fetched India VIX from Upstox: {vix_data}")
                                break
                except Exception as e:
                    logger.warning(f"Upstox VIX fetch failed: {e}")
            
            # Fallback to yfinance if Upstox didn't work
            if vix_data is None and quota_svc.check_yfinance_quota():
                try:
                    vix_cache_key = f"fetch_history_cached:('^INDIAVIX',){{'interval': '1d', 'period': '5d'}}"
                    cached_vix = yfinance_cache.get(vix_cache_key)
                    
                    if cached_vix is not None:
                        v_hist = cached_vix
                        logger.debug("Cache hit for India VIX")
                    else:
                        v_hist = fetch_history_cached("^INDIAVIX", period="5d", interval="1d")
                        quota_svc.log_yfinance_usage()
                        logger.info("Fetched fresh data for India VIX from yfinance")
                    
                    if v_hist is not None and not v_hist.empty:
                        v_curr = v_hist['Close'].iloc[-1]
                        v_prev = v_hist['Close'].iloc[-2] if len(v_hist) > 1 else v_curr
                        v_chg = v_curr - v_prev
                        v_pct = (v_chg / v_prev * 100) if v_prev != 0 else 0
                        
                        vix_data = {
                            "value": round(float(v_curr), 2),
                            "change": round(float(v_chg), 2),
                            "change_pct": round(float(v_pct), 2)
                        }
                except Exception as e:
                    logger.warning(f"yfinance VIX fallback failed: {e}")
            
            if vix_data:
                output['vix'] = vix_data
            else:
                logger.warning("VIX data unavailable - using fallback with zero values")
                output['vix'] = {"value": 0, "change": 0, "change_pct": 0}
        except Exception as e:
            logger.error(f"VIX fetch error: {e}, using fallback with zero values")
            output['vix'] = {"value": 0, "change": 0, "change_pct": 0}

        # Cache the result
        self._global_indices_cache = {"data": output, "timestamp": now}
        
        return output

    def discover_nse_universe(self):
        """Fetch NSE 500 or similar broad list."""
        try:
            # We try to use the hardcoded list first, then attempt to find more
            # In a real system, we'd fetch from NSE website or a local cache
            return list(self.universe)
        except Exception as e:
            logger.error(f"Universe discovery error: {e}")
            return list(_NIFTY100_NS)

    def _get_top_candidates(self, limit=10):
        """Scan broad universe for high-volume momentum movers.
        
        PRIORITY: Upstox first for NSE data, yfinance as fallback only.
        Uses batch fetching for efficiency.
        """
        try:
            if not self.universe: self.universe = self.discover_nse_universe()
            
            universe_subset = self.universe[:50]
            candidates = []

            # PRIORITY 1: Try Upstox first (primary data source for NSE)
            if upstox_client.is_authenticated:
                logger.info("Using Upstox for top candidates (priority 1)")
                keys = []
                key_to_ticker = {}
                for ticker in universe_subset:
                    k = get_instrument_key(ticker)
                    if k:
                        keys.append(k)
                        key_to_ticker[k] = ticker

                # Batch fetch quotes from Upstox
                if keys:
                    keys_str = ",".join(keys)
                    res = upstox_client.fetch_market_quote(keys_str)
                    if res and res.get("status") == "success":
                        data = res.get("data", {})
                        for k, quote in data.items():
                            ticker = key_to_ticker.get(k)
                            if not ticker: continue
                            
                            last_price = quote.get("last_price", 0)
                            prev_close = quote.get("close", last_price)
                            if prev_close == 0: continue
                            
                            gain = (last_price - prev_close) / prev_close
                            
                            # Upstox quote may have volume, but we rely on gain for ranking if volume isn't comparable easily
                            vol_surge = quote.get("volume", 1) / 100000  # Pseudo vol surge based on absolute volume
                            
                            if gain > 0.01: # 1% gain
                                candidates.append((ticker, gain * vol_surge))
                    else:
                        logger.warning("Upstox quote fetch returned no data, will try yfinance fallback")
            
            # FALLBACK 2: Use yfinance only if Upstox didn't return candidates
            # Check quota before making yfinance calls
            if not candidates and quota_svc.check_yfinance_quota():
                logger.info("Using yfinance fallback for top candidates (quota available)")
                # Use batch fetch for efficiency - this is a SINGLE API call for multiple tickers
                batch_results = batch_fetch_quotes(universe_subset[:20], max_batch=20)
                
                # Log yfinance usage ONCE for the entire batch (not per ticker)
                # batch_fetch_quotes makes ONE request that returns multiple tickers
                if batch_results:
                    quota_svc.log_yfinance_usage()
                
                global _using_yfinance_fallback
                _using_yfinance_fallback = True
                
                for ticker, quote_data in batch_results.items():
                    try:
                        price = quote_data.get("price")
                        change_pct = quote_data.get("change_pct", 0)
                        volume = quote_data.get("volume", 0)
                        
                        if price is None or price == 0:
                            continue
                        
                        # Heuristic: Gain > 1.5%
                        gain = change_pct / 100
                        if gain > 0.015:  # 1.5% gain
                            # Estimate vol_surge (we don't have prev volume in batch)
                            vol_surge = 1.5 if volume > 1000000 else 1.0
                            candidates.append((ticker, gain * vol_surge))
                    except Exception as e:
                        logger.debug(f"Error processing {ticker}: {e}")
                        continue
            elif not candidates:
                logger.warning("yfinance quota exhausted, cannot fetch top candidates")
            
            # Sort by combined score
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [c[0] for c in candidates[:limit]]
        except Exception as e:
            logger.error(f"Top candidates error: {e}")
            return self.universe[:limit]

    async def get_market_discovery(self):
        """Discovers stocks based on technical momentum scans."""
        return self._get_top_candidates(8)

    async def get_top_movers(self):
        """Fetch NSE top gainers/losers."""
        return self._get_top_candidates(5)
