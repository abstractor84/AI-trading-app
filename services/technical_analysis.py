import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

from services.upstox_service import UpstoxService
from services.advanced_indicators import classifier

logger = logging.getLogger(__name__)

# Singleton Upstox instance shared across calls
_upstox_svc = UpstoxService()

class TechnicalAnalysisService:
    def __init__(self):
        pass

    def fetch_ohlcv(self, ticker: str, period="5d", interval="5m", data_provider="upstox"):
        """Fetch 5-minute OHLCV data. Uses Upstox by default, falls back to yfinance."""
        if _upstox_svc.is_authenticated:
            days = int(period.replace("d", "")) if period.endswith("d") else 5
            
            # Upstox API only supports 1minute or 30minute. We pull 1min and resample to 5min.
            upstox_interval = "1minute" if interval == "5m" else interval.replace("m", "minute")
            if upstox_interval == "5minute": upstox_interval = "1minute" # Mandatory correction
            
            df_upstox = _upstox_svc.fetch_ohlcv(ticker, days=days, interval=upstox_interval)
            
            if df_upstox is not None and not df_upstox.empty:
                # Resample to 5-minute bars if requested 5m
                if interval == "5m" and upstox_interval == "1minute":
                    df_upstox = df_upstox.resample('5min').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                    }).dropna()
                
                if len(df_upstox) >= 10:
                    logger.info(f"Using Upstox data for {ticker} ({len(df_upstox)} rows, resampled to {interval})")
                    return df_upstox
            
            logger.warning(f"Upstox data unavailable for {ticker}. Falling back to yfinance.")
        else:
            if data_provider == "upstox":
                logger.debug(f"Upstox not authenticated, skipping for {ticker}")

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
            "vol_surge": float(round(vol_surge, 2)),
            "lorentzian": classifier.classify(df)
        }

    def analyze_stock(self, ticker: str, data_provider: str = "upstox"):
        try:
            df = self.fetch_ohlcv(ticker, data_provider=data_provider)
            indicators = self.compute_indicators(df)
            return indicators
        except Exception as e:
            logger.error(f"Error computing TA for {ticker}: {e}")
            return None

    def evaluate_math_probability(self, ta_data: dict) -> float:
        """
        Pure Math evaluator. Calculates a mathematical probability of a valid trade setup (0.0 to 1.0)
        based purely on technical indicator alignments.
        Prevents sending low probability stocks to the AI, saving API costs and time.
        """
        if not ta_data: return 0.0
        
        close = ta_data.get('close', 0)
        vwap = ta_data.get('vwap', 0)
        ema9 = ta_data.get('ema_9', 0)
        ema21 = ta_data.get('ema_21', 0)
        rsi = ta_data.get('rsi_14', 50)
        adx = ta_data.get('adx_14', 0)
        macd_hist = ta_data.get('macd_hist', 0)
        
        score = 0.0
        
        # 1. Trend Alignment (0.3 weight)
        if ema9 > ema21 and close > vwap:
            score += 0.3  # Strong bullish alignment
        elif ema9 < ema21 and close < vwap:
            score += 0.3  # Strong bearish alignment
        elif ema9 > ema21 or ema9 < ema21:
            score += 0.15 # Weak trend
            
        # 2. Momentum / ADX (0.3 weight)
        if adx > 25:
            score += 0.3  # High momentum
        elif adx > 15:
            score += 0.15 # Moderate momentum
            
        # 3. RSI Oscillators (0.2 weight)
        if 40 <= rsi <= 60:
            score += 0.1  # Choppy neutral
        elif 30 <= rsi <= 70:
            score += 0.2  # Healthy active
        else:
            score += 0.0  # Overextended (exhaustion risk)
            
        # 4. MACD Directional Bias (0.2 weight)
        if (ema9 > ema21 and macd_hist > 0) or (ema9 < ema21 and macd_hist < 0):
            score += 0.2  # MACD supports the trend
            
        return round(score, 2)

    def classify_signal(self, ta_data: dict) -> str:
        """
        Surgical TA classification. 
        Requires multi-indicator alignment to trigger a non-NEUTRAL signal.
        """
        if not ta_data: return "NEUTRAL"
        
        score = 0
        rsi = ta_data.get("rsi_14", 50)
        macd_hist = ta_data.get("macd_hist", 0)
        adx = ta_data.get("adx_14", 0)
        vol_surge = ta_data.get("vol_surge", 1)
        ema_9 = ta_data.get("ema_9", 0)
        ema_21 = ta_data.get("ema_21", 0)
        close = ta_data.get("close", 0)
        vwap = ta_data.get("vwap", 0)

        # 1. Trend Alignment (Highest Weight)
        if ema_9 > ema_21 and close > vwap: score += 2
        elif ema_9 < ema_21 and close < vwap: score -= 2

        # 2. Momentum (MACD)
        if macd_hist > 0.5: score += 1
        elif macd_hist < -0.5: score -= 1

        # 3. Strength (ADX)
        if adx > 25:
            if score > 0: score += 1
            elif score < 0: score -= 1

        # 4. Exhaustion (RSI)
        if rsi > 70: score -= 1 # Overbought
        elif rsi < 30: score += 1 # Oversold

        # 5. Volume Surge
        if vol_surge > 2.0:
            if score > 0: score += 1
            elif score < 0: score -= 1

        if score >= 4: return "STRONG BUY"
        if score >= 2: return "BUY"
        if score <= -4: return "SHORT SELL" # Standardized from STRONG SELL
        if score <= -2: return "SELL"       # Standardized from SELL
        return "NEUTRAL"

    def get_connection_status(self) -> dict:
        """Unified status for all external data & AI providers."""
        from services.quota_service import quota_svc
        upstox_profile = _upstox_svc.fetch_profile()
        return {
            "upstox": {
                "connected": _upstox_svc.is_authenticated,
                "user": upstox_profile.get("user_name") if upstox_profile else None,
                "error": None if _upstox_svc.is_authenticated else "Missing Token"
            },
            "ai": {
                "remaining": max(0, 20 - quota_svc.get_total_daily_usage()),
                "limit": 20
            }
        }

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
