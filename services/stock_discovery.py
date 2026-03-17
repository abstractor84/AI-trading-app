import logging
import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from services.upstox_service import upstox_client, get_instrument_key

logger = logging.getLogger(__name__)

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
    def __init__(self):
        self.universe = list(_NIFTY100_NS)  # Start with hardcoded list immediately

    def fetch_global_indices(self):
        """Fetches the latest prices and formats them into categorized HTML rows."""
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
                "GIFT Nifty": "NSE_INDEX|GIFT Nifty",
                "NIFTY 50": "^NSEI",
                "SENSEX": "^BSESN",
                "BANK NIFTY": "^NSEBANK",                "FINNIFTY": "^CNXFIN",
                "NIFTY MIDCAP 100": "^CRSLDX", 
                "NIFTY SMALLCAP 100": "^CNXSC",
                "NIFTY IT": "^CNXIT",
                "NIFTY AUTO": "^CNXAUTO", 
                "NIFTY METAL": "^CNXMETAL",
                "NIFTY PHARMA": "^CNXPHARMA",
                "NIFTY FMCG": "^CNXFMCG",
                "NIFTY REALTY": "^CNXREALTY",
                "NIFTY ENERGY": "^CNXENERGY",
                "NIFTY INFRA": "^CNXINFRA"
            }
        }

        output = {"global": {}, "india": {}, "vix": {"value": 0, "change": 0, "change_pct": 0}}
        cache = {}

        for category, ticker_map in symbols.items():
            for name, ticker in ticker_map.items():
                try:
                    data = None
                    # Use Upstox for GIFT Nifty if authenticated
                    if "NSE_INDEX" in ticker:
                        from services.upstox_service import upstox_client
                        if upstox_client.is_authenticated:
                            q = upstox_client.fetch_market_quote(ticker)
                            if q and 'data' in q and ticker in q['data']:
                                item = q['data'][ticker]
                                current = item.get('last_price', 0)
                                prev = item.get('close_price', 0)
                                change = current - prev if prev != 0 else 0
                                change_pct = (change / prev * 100) if prev != 0 else 0
                                data = {
                                    "name": name,
                                    "value": round(current, 2),
                                    "change": round(change, 2),
                                    "change_pct": round(change_pct, 2)
                                }

                    # Fallback to yfinance
                    if data is None:
                        if ticker not in cache:
                            t = yf.Ticker(ticker)
                            hist = t.history(period="5d", interval="1d")
                            cache[ticker] = hist
                        else:
                            hist = cache[ticker]
                            
                        if not hist.empty and len(hist) >= 1:
                            current = hist['Close'].iloc[-1]
                            prev = hist['Close'].iloc[-2] if len(hist) > 1 else 0
                            change = current - prev if prev != 0 else 0
                            change_pct = (change / prev * 100) if prev != 0 else 0
                            
                            data = {
                                "value": round(float(current), 2) if not pd.isna(current) else 0,
                                "change": round(float(change), 2) if not pd.isna(change) else 0,
                                "change_pct": round(float(change_pct), 2) if not pd.isna(change_pct) else 0
                            }
                    
                    if data:
                        if category == "GLOBAL":
                            output["global"][name] = data
                        else:
                            output["india"][name] = data
                except Exception as e:
                    logger.error(f"Error fetching {name} ({ticker}): {e}")

        # Manually attach VIX
        try:
            v_ticker = yf.Ticker("^INDIAVIX")
            v_hist = v_ticker.history(period="5d", interval="1d")
            if not v_hist.empty:
                v_curr = v_hist['Close'].iloc[-1]
                v_prev = v_hist['Close'].iloc[-2] if len(v_hist) > 1 else 0
                v_chg = v_curr - v_prev if v_prev != 0 else 0
                v_pct = (v_chg / v_prev * 100) if v_prev != 0 else 0
                output['vix'] = {
                    "value": round(float(v_curr), 2),
                    "change": round(float(v_chg), 2),
                    "change_pct": round(float(v_pct), 2)
                }
        except Exception as e:
            logger.error(f"VIX fetch error: {e}")

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
        """Scan broad universe for high-volume momentum movers."""
        try:
            if not self.universe: self.universe = self.discover_nse_universe()
            
            universe_subset = self.universe[:50]
            candidates = []

            if upstox_client.is_authenticated:
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
                                
            # Fallback to yfinance if Upstox fails or isn't connected
            if not candidates:
                df = yf.download(universe_subset, period="2d", interval="1d", group_by='ticker', progress=False)
                
                for ticker in universe_subset:
                    try:
                        t_data = df[ticker]
                        if len(t_data) < 2: continue
                        
                        close_prev = t_data['Close'].iloc[-2]
                        close_curr = t_data['Close'].iloc[-1]
                        vol_prev = t_data['Volume'].iloc[-2]
                        vol_curr = t_data['Volume'].iloc[-1]
                        
                        if pd.isna(close_curr) or pd.isna(vol_curr): continue
                        
                        gain = (close_curr - close_prev) / close_prev if close_prev > 0 else 0
                        vol_surge = vol_curr / vol_prev if vol_prev > 0 else 1
                        
                        # Heuristic: Gain > 1.5% and Volume > 1.5x previous
                        if gain > 0.015 and vol_surge > 1.5:
                            candidates.append((ticker, gain * vol_surge))
                    except:
                        continue
            
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
