import logging
import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Core NSE universe for discovery
_NIFTY100_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "BHARTIARTL",
    "INFY", "SBI-N", "LICI", "ITC", "HINDUNILVR",
    "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA",
    "ADANIENT", "KOTAKBANK", "TITAN", "ONGC", "TATAMOTORS",
    "NTPC", "AXISBANK", "ADANIPORTS", "ASIANPAINT", "COALINDIA",
    "BAJAJHLDNG", "BAJAJ-AUTO", "BHAL", "JSWSTEEL", "ADANIPOWR",
    "M&M", "TRENT", "BEL", "ULTRACEMCO", "SIEMENS",
    "GRASIM", "SBILIFE", "BPCL", "NESTLEIND", "HAL",
    "ZOMATO", "PNB", "IOB", "CANBK", "INDHOTEL",
    "DLF", "EICHERMOT", "HINDALCO", "DIVISLAB", "CIPLA",
    "BRITANNIA", "TATASTEEL", "ADANIENSOL", "M&M", "TRENT", "BEL",
    # Nifty Next 50
    "ABB", "ADANIPOWR", "AMBUJACEM", "BANKBARODA", "BEL",
    "BHEL", "CANBK", "CHOLAFIN", "DLF", "GAIL",
    "HAVELLS", "HEROMOTOCO", "INDHOTEL", "IOC", "IRCTC",
    "JINDALSTEL", "JSWSTEEL", "LTIM", "MARICO", "PFC",
    "PNB", "RECLTD", "SAIL", "SHREECEM", "SIEMENS",
    "SRF", "TATAPOWER", "TVSMOTOR", "UNITDSPR", "VBL",
    "YESBANK", "SHREECEM", "DLF", "HAVELLS", "PIDILITIND",
    "ICICIPRULI", "HDFCLIFE", "BAJAJFINSV", "SHRIRAMFIN", "MCDOWELL-N",
    "OFSS", "PFC", "PIDILITIND", "PNB", "RECLTD",
    "SAIL", "SHREECEM", "SIEMENS", "SRF", "TATAPOWER",
    "TORNTPHARM", "TVSMOTOR", "VBL", "VEDL", "VOLTAS",
    "YESBANK", "ZOMATO", "NYKAA", "POLICYBZR", "PAYTM",
    "IRFC", "ABCAPITAL", "PIIND", "UNITDSPR", "IDFCFIRSTB",
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
                "Brent Crude": "BZ=F", 
                "Gold": "GC=F", 
                "Silver": "SI=F",
                "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW 30": "^DJI",
                "DAX": "^GDAXI", "FTSE 100": "^FTSE", "NIKKEI": "^N225"
            },
            "INDIA": {
                "GIFT Nifty": "^NSEI", 
                "NIFTY 50": "^NSEI", 
                "SENSEX": "^BSESN", 
                "BANK NIFTY": "^NSEBANK", 
                "NIFTY IT": "^CNXIT",
                "NIFTY AUTO": "^CNXAUTO", "NIFTY METAL": "^CNXMETAL",
                "MIDCAP 100": "^CRSLDX", "SMALLCAP 100": "^CNXSC"
            }
        }

        output = {"global": {}, "india": {}, "vix": {"value": 0, "change": 0, "change_pct": 0}}

        for category, ticker_map in symbols.items():
            for name, ticker in ticker_map.items():
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="5d", interval="1d")
                    if not hist.empty and len(hist) >= 1:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        change = current - prev
                        change_pct = (change / prev * 100) if prev != 0 else 0
                        
                        data = {
                            "value": round(current, 2),
                            "change": round(change, 2),
                            "change_pct": round(change_pct, 2)
                        }
                        
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
                v_prev = v_hist['Close'].iloc[-2] if len(v_hist) > 1 else v_curr
                v_chg = v_curr - v_prev
                v_pct = (v_chg / v_prev * 100) if v_prev != 0 else 0
                output['vix'] = {
                    "value": round(v_curr, 2),
                    "change": round(v_chg, 2),
                    "change_pct": round(v_pct, 2)
                }
        except Exception as e:
            logger.error(f"VIX fetch error: {e}")

        return output

    async def get_market_discovery(self):
        """Discovers stocks based on technical momentum scans."""
        # Simple placeholder for now, actual logic in background_engine
        return []

    async def get_top_movers(self):
        """Fetch NSE top gainers/losers."""
        return []
