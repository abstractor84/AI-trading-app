import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

from services.upstox_service import UpstoxService

logger = logging.getLogger(__name__)

# Singleton Upstox instance shared across calls
_upstox_svc = UpstoxService()

class TechnicalAnalysisService:
    def __init__(self):
        pass

    def fetch_ohlcv(self, ticker: str, period="5d", interval="5m", data_provider="yfinance"):
        """Fetch 5-minute OHLCV data. Uses Upstox when available, falls back to yfinance."""
        if data_provider == "upstox" and _upstox_svc.is_authenticated:
            days = int(period.replace("d", "")) if period.endswith("d") else 5
            df_upstox = _upstox_svc.fetch_ohlcv(ticker, days=days, interval=interval.replace("m", "minute"))
            if df_upstox is not None and not df_upstox.empty and len(df_upstox) >= 10:
                logger.info(f"Using Upstox data for {ticker} ({len(df_upstox)} rows)")
                return df_upstox
            logger.warning(f"Upstox data unavailable for {ticker}. Falling back to yfinance.")

        # Default: yfinance - Using Ticker object for cleaner single-ticker data
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                # Fallback to download if history fails
                df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
            
            df.dropna(inplace=True)
            # Ensure columns are standard case if yfinance/upstox differ
            df.columns = [c.capitalize() for c in df.columns]
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()

    def compute_indicators(self, df: pd.DataFrame):
        """Compute RSI, EMA, VWAP, MACD, BB, ADX, and Volume Surge."""
        if df.empty or len(df) < 50:
            return None

        # Typical Price for calculations - ensuring we use Series even if columns are slightly mismatched
        try:
            high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
            low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
            close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
            df['TP_Internal'] = (high + low + close) / 3
        except Exception as e:
            logger.warning(f"Typical Price internal calculation issue: {e}")
            df['TP_Internal'] = df['Close']

        # Short & Medium EMA
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        
        # RSI
        df.ta.rsi(length=14, append=True)
        
        # MACD
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True)
        
        # ADX
        df.ta.adx(length=14, append=True)

        # Drop NaNs to safely access current values
        df.dropna(inplace=True)
        
        if len(df) == 0:
            return None

        # VWAP custom calculation for today only
        # We find today's start index
        today = df.index[-1].date()
        today_df = df[df.index.date == today].copy()
        
        if len(today_df) > 0:
            try:
                high_t = today_df['High'].squeeze() if isinstance(today_df['High'], pd.DataFrame) else today_df['High']
                low_t = today_df['Low'].squeeze() if isinstance(today_df['Low'], pd.DataFrame) else today_df['Low']
                close_t = today_df['Close'].squeeze() if isinstance(today_df['Close'], pd.DataFrame) else today_df['Close']
                vol_t = today_df['Volume'].squeeze() if isinstance(today_df['Volume'], pd.DataFrame) else today_df['Volume']
                
                tp = (high_t + low_t + close_t) / 3
                vp = tp * vol_t
                vwap = vp.cumsum() / vol_t.cumsum()
                current_vwap = vwap.iloc[-1]
            except Exception as e:
                logger.error(f"VWAP computation error: {e}")
                current_vwap = df['Close'].iloc[-1]
        else:
            current_vwap = df['Close'].iloc[-1] 
        
        # Volume Surge check
        avg_vol_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        if isinstance(current_vol, pd.Series): current_vol = current_vol.iloc[-1]
        if isinstance(avg_vol_20, pd.Series): avg_vol_20 = avg_vol_20.iloc[-1]
        
        vol_surge = float(current_vol / avg_vol_20) if avg_vol_20 > 0 else 1.0

        latest = df.iloc[-1]

        # Extract names from pandas_ta suffixes which can vary
        macd_col = [c for c in df.columns if 'MACD_' in c][0]
        macdh_col = [c for c in df.columns if 'MACDh_' in c][0]
        macds_col = [c for c in df.columns if 'MACDs_' in c][0]
        
        bbm_col = [c for c in df.columns if 'BBM_' in c][0]
        bbu_col = [c for c in df.columns if 'BBU_' in c][0]
        bbl_col = [c for c in df.columns if 'BBL_' in c][0]
        
        adx_col = [c for c in df.columns if 'ADX_' in c][0]
        
        return {
            "close": float(latest['Close']),
            "ema_9": float(latest['EMA_9']),
            "ema_21": float(latest['EMA_21']),
            "rsi_14": float(latest['RSI_14']),
            "macd": float(latest[macd_col]),
            "macd_hist": float(latest[macdh_col]),
            "macd_signal": float(latest[macds_col]),
            "bb_upper": float(latest[bbu_col]),
            "bb_lower": float(latest[bbl_col]),
            "bb_mid": float(latest[bbm_col]),
            "adx_14": float(latest[adx_col]),
            "vwap": float(current_vwap),
            "vol_surge": float(round(vol_surge, 2))
        }

    def analyze_stock(self, ticker: str, data_provider: str = "yfinance"):
        try:
            df = self.fetch_ohlcv(ticker, data_provider=data_provider)
            indicators = self.compute_indicators(df)
            return indicators
        except Exception as e:
            logger.error(f"Error computing TA for {ticker}: {e}")
            return None

    def fetch_fundamentals(self, ticker: str):
        """Fetch basic fundamental data via yfinance."""
        try:
            info = yf.Ticker(ticker).info
            return {
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "sector": info.get("sector", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A")
            }
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {}
