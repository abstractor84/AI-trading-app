import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

import logging
logger = logging.getLogger(__name__)

class TechnicalAnalysisService:
    def __init__(self):
        pass

    def fetch_ohlcv(self, ticker: str, period="5d", interval="5m"):
        """Fetch 5-minute OHLCV data for technical analysis."""
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        # flatten multi-index if necessary (happens in newer yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df.dropna(inplace=True)
        return df

    def compute_indicators(self, df: pd.DataFrame):
        """Compute RSI, EMA, VWAP, MACD, BB, ADX, and Volume Surge."""
        if df.empty or len(df) < 50:
            return None

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
            today_df['Typical_Price'] = (today_df['High'] + today_df['Low'] + today_df['Close']) / 3
            today_df['VP'] = today_df['Typical_Price'] * today_df['Volume']
            vwap = today_df['VP'].cumsum() / today_df['Volume'].cumsum()
            current_vwap = vwap.iloc[-1]
        else:
            current_vwap = df['Close'].iloc[-1] # Fallback if today logic fails

        # Calculate volume surge (current bar vol / average of last 20 bars)
        avg_vol_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
        current_vol = df['Volume'].iloc[-1]
        vol_surge = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

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

    def analyze_stock(self, ticker: str):
        try:
            df = self.fetch_ohlcv(ticker)
            indicators = self.compute_indicators(df)
            return indicators
        except Exception as e:
            logger.error(f"Error computing TA for {ticker}: {e}")
            return None
