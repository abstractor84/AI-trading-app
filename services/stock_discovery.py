import yfinance as yf
import pandas as pd
import requests
import io
import logging

logger = logging.getLogger(__name__)

class StockDiscoveryService:
    def __init__(self):
        self.universe = []
    def fetch_global_indices(self):
        """Fetches the latest prices and formats them into categorized HTML rows."""
        symbols = {
            "GLOBAL": {
                "S&P 500": "^GSPC",
                "NASDAQ": "^IXIC",
                "DOW 30": "^DJI",
                "FTSE": "^FTSE",
                "NIKKEI": "^N225",
            },
            "INDIA": {
                "NIFTY 50": "^NSEI",
                "SENSEX": "^BSESN",
                "BANK NIFTY": "^NSEBANK",
                "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
                "NIFTY IT": "^CNXIT",
                "NIFTY AUTO": "^CNXAUTO",
                "NIFTY METAL": "^CNXMETAL",
                "NIFTY PHARMA": "^CNXPHARMA",
                "NIFTY FMCG": "^CNXFMCG",
                "NIFTY ENERGY": "^CNXENERGY",
                "NIFTY MID 50": "^NSMIDCP"
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
        """Dynamically fetch Nifty 100 list from NSE file or yfinance screener logic."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            url = "https://en.wikipedia.org/wiki/NIFTY_50"
            html = requests.get(url, headers=headers).content
            df_list = pd.read_html(html)
            # Find the constituents table
            for df in df_list:
                if 'Symbol' in df.columns:
                    nifty50_symbols = df['Symbol'].tolist()
                    self.universe = [f"{sym}.NS" for sym in nifty50_symbols]
                    break
            
            url_next50 = "https://en.wikipedia.org/wiki/NIFTY_Next_50"
            html_n = requests.get(url_next50, headers=headers).content
            df_list_n = pd.read_html(html_n)
            # Find the constituents table for next 50
            for df in df_list_n:
                if 'Symbol' in df.columns:
                    nifty_next50_symbols = df['Symbol'].tolist()
                    self.universe.extend([f"{sym}.NS" for sym in nifty_next50_symbols])
                    break
            
            logger.info(f"Discovered {len(self.universe)} NSE symbols dynamically.")
            return self.universe
        except Exception as e:
            logger.error(f"Failed to discover NSE universe dynamically: {e}")
            # Fallback to a few major ones to prevent total failure
            self.universe = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
            return self.universe

    def _get_top_candidates(self, limit=12):
        """Screen the universe based on day's volume, top gainers/losers."""
        if not self.universe:
            self.discover_nse_universe()
        
        # Download 1-day data for all to find top movers
        try:
            tickers_str = " ".join(self.universe)
            # Fetch today's data (or yesterday if market closed)
            data = yf.download(tickers_str, period="5d", interval="1d", group_by="ticker", auto_adjust=True)
            
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
            
            # Sort by a combination of absolute change and volume surge
            moves.sort(key=lambda x: (abs(x['change_pct']) * x['vol_surge']), reverse=True)
            top_candidates = [m['ticker'] for m in moves[:limit]]
            logger.info(f"Top {limit} candidates selected from {len(moves)} valid symbols.")
            return top_candidates
        except Exception as e:
            logger.error(f"Candidate screening error: {e}")
            return self.universe[:limit] # return random top if fails
