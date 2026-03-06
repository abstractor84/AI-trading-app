import pandas as pd
import logging
import datetime
import time
from openchart import NSEData

logger = logging.getLogger(__name__)

class OpenChartService:
    """
    Service to fetch historical intraday and EOD data directly from NSE/NFO 
    using the openchart library. This provides high-fidelity data beyond 
    the 30-day limit typical of many brokers.
    """
    def __init__(self):
        self.nse = NSEData()

    def fetch_historical(self, ticker: str, start_date: datetime.date, end_date: datetime.date, interval: str = "5m"):
        """
        Fetch historical data for a given symbol and range.
        OpenChart Library: expects datetime objects for start/end.
        Intervals: '1m', '5m', '15m', '1d'
        """
        # Normalize ticker: OpenChart expects 'RELIANCE', not 'RELIANCE.NS'
        symbol = ticker.replace(".NS", "").upper()
        
        # Map interval names to OpenChart internal map: '1m', '5m', etc.
        interval_map = {
            "1minute": "1m",
            "5minute": "5m",
            "15minute": "15m",
            "1day": "1d",
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1d": "1d"
        }
        oc_interval = interval_map.get(interval, interval)
        
        try:
            # Map interval names to OpenChart internal map
            interval_map = {
                "1minute": "1m", "5minute": "5m", "15minute": "15m", "1day": "1d",
                "1m": "1m", "5m": "5m", "15m": "15m", "1d": "1d"
            }
            oc_interval = interval_map.get(interval, "5m")
            
            logger.info(f"OpenChart: Fetching {oc_interval} for {symbol}")
            
            # Start/End as datetime
            start_dt = datetime.datetime.combine(start_date, datetime.time(9, 15))
            end_dt = datetime.datetime.combine(end_date, datetime.time(15, 30))

            df = self.nse.historical(
                symbol=symbol,
                segment="EQ",
                start=start_dt,
                end=end_dt,
                interval=oc_interval
            )
            
            if df is None or df.empty:
                logger.warning(f"OpenChart: No data returned for {symbol}")
                return pd.DataFrame()

            # Normalize column names
            df.columns = [c.capitalize() for c in df.columns]
            
            # OpenChart returns a DataFrame with 'Date' or indexed by timestamp
            if "Date" in df.columns:
                df.set_index("Date", inplace=True)
            
            # Ensure index is Datetime and timezone is IST
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
            else:
                df.index = df.index.tz_convert("Asia/Kolkata")
                
            logger.info(f"OpenChart: Successfully fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"OpenChart: Fetch failed for {symbol}: {e}")
            return pd.DataFrame()

openchart_svc = OpenChartService()
