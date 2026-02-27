import yfinance as yf
import pandas as pd
import requests
import io
import logging

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# Hardcoded universe: Nifty 50 + Nifty Next 50 — used as instant fallback
# (Wikipedia fetch may fail or be slow — this ensures scan always works)
_NIFTY100_SYMBOLS = [
    # Nifty 50
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "SBIN", "BAJFINANCE",
    "AXISBANK", "MARUTI", "ASIANPAINT", "HCLTECH", "SUNPHARMA", "TITAN",
    "WIPRO", "ONGC", "NTPC", "POWERGRID", "ULTRACEMCO", "NESTLEIND",
    "TECHM", "BAJAJFINSV", "BPCL", "COALINDIA", "GRASIM", "TATACONSUM",
    "TATAMOTORS", "BRITANNIA", "INDUSINDBK", "APOLLOHOSP", "SBILIFE",
    "JSWSTEEL", "HINDALCO", "HDFCLIFE", "DIVISLAB", "CIPLA",
    "DRREDDY", "EICHERMOT", "UPL", "ADANIPORTS", "ADANIENT",
    "TATASTEEL", "ADANIENSOL", "M&M", "TRENT", "BEL",
    # Nifty Next 50
    "ABB", "ADANIPOWER", "AMBUJACEM", "BAJAJ-AUTO", "BANKBARODA",
    "BERGEPAINT", "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL",
    "DABUR", "DLF", "GAIL", "GODREJCP", "HAVELLS",
    "INDIGO", "IOC", "IRCTC", "JINDALSTEL", "LICI",
    "LTIM", "LUPIN", "MARICO", "MUTHOOTFIN", "OBEROIRLTY",
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
                "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW 30": "^DJI",
                "FTSE": "^FTSE", "NIKKEI": "^N225",
            },
            "INDIA": {
                "NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK",
                "FINNIFTY": "NIFTY_FIN_SERVICE.NS", "NIFTY IT": "^CNXIT",
                "NIFTY AUTO": "^CNXAUTO", "NIFTY METAL": "^CNXMETAL",
                "NIFTY PHARMA": "^CNXPHARMA", "NIFTY FMCG": "^CNXFMCG",
                "NIFTY ENERGY": "^CNXENERGY", "NIFTY MID 50": "^NSMIDCP"
            }
        }

        output = {"global": {}, "india": {}}

        for category, ticker_map in symbols.items():
            dict_group = {}
            for name, ticker in ticker_map.items():
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="5d", interval="1d")
                    if len(hist) >= 2:
                        last_close = hist['Close'].iloc[-2]
                        current = hist['Close'].iloc[-1]
                        change = current - last_close
                        change_pct = (change / last_close) * 100
                        dict_group[name] = {
                            "value": round(current, 2),
                            "change": round(change, 2),
                            "change_pct": round(change_pct, 2)
                        }
                    else:
                        dict_group[name] = {"value": 0, "change": 0, "change_pct": 0}
                except Exception as e:
                    logger.error(f"Error fetching {name}: {e}")
                    dict_group[name] = {"value": 0, "change": 0, "change_pct": 0, "error": str(e)}
            output[category.lower()] = dict_group

        # Manually attach VIX
        try:
            v_ticker = yf.Ticker("^INDIAVIX")
            v_hist = v_ticker.history(period="5d", interval="1d")
            if len(v_hist) >= 2:
                v_chg = v_hist['Close'].iloc[-1] - v_hist['Close'].iloc[-2]
                v_pct = (v_chg / v_hist['Close'].iloc[-2]) * 100
                output['vix'] = {
                    "value": round(v_hist['Close'].iloc[-1], 2),
                    "change": round(v_chg, 2),
                    "change_pct": round(v_pct, 2)
                }
        except Exception as e:
            output['vix'] = {"value": 0, "change": 0, "change_pct": 0}

        return output

    def discover_nse_universe(self):
        """
        Dynamically fetches the latest NIFTY 200 index constituents directly
        from the official NSE India CSV.
        """
        try:
            url = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
            df = pd.read_csv(url, timeout=10)
            if 'Symbol' in df.columns:
                syms = [str(s).strip().upper() for s in df['Symbol'].tolist()]
                self.universe = [f"{s}.NS" for s in syms if s and s != 'NAN']
                logger.info(f"Dynamically fetched {len(self.universe)} NIFTY 200 symbols from official NSE India CSV.")
            else:
                raise ValueError("NSE CSV format changed, 'Symbol' column missing")
                
        except Exception as e:
            logger.warning(f"NSE universe dynamic fetch failed ({e}). Falling back to internal Nifty 100 cache.")
            self.universe = list(_NIFTY100_NS)

        return self.universe

    def _get_top_candidates(self, limit=12):
        """Screen the universe based on day's volume and price movement."""
        if not self.universe:
            self.discover_nse_universe()

        try:
            tickers_str = " ".join(self.universe)
            data = yf.download(tickers_str, period="5d", interval="1d", group_by="ticker", auto_adjust=True, progress=False)

            moves = []
            for ticker in self.universe:
                if ticker in data and len(data[ticker]) >= 2:
                    df_t = data[ticker]
                    last_close = df_t['Close'].iloc[-2]
                    current = df_t['Close'].iloc[-1]
                    volume = df_t['Volume'].iloc[-1]
                    avg_volume = df_t['Volume'].mean()

                    if pd.isna(current) or last_close == 0:
                        continue

                    change_pct = ((current - last_close) / last_close) * 100
                    vol_surge = 0 if avg_volume == 0 else (volume / avg_volume)

                    moves.append({
                        "ticker": ticker,
                        "change_pct": change_pct,
                        "volume": volume,
                        "vol_surge": vol_surge
                    })

            moves.sort(key=lambda x: (abs(x['change_pct']) * x['vol_surge']), reverse=True)
            top_candidates = [m['ticker'] for m in moves[:limit]]
            
            logger.info(f"=== STOCK SCANNER PRE-FILTER ===")
            logger.info(f"Top {limit} candidates selected from {len(moves)} valid symbols based on (Change% * VolSurge).")
            for i, m in enumerate(moves[:limit]):
                logger.info(f"  {i+1}. {m['ticker']} | Chg: {m['change_pct']:.2f}% | VolSurge: {m['vol_surge']:.2f}x")
            logger.info(f"================================")
            
            return top_candidates
        except Exception as e:
            logger.error(f"Candidate screening error: {e}")
            return self.universe[:limit]
