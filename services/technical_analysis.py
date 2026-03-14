import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import datetime
import pytz

from services.upstox_service import UpstoxService
from services.advanced_indicators import classifier

logger = logging.getLogger(__name__)

# Singleton Upstox instance shared across calls
_upstox_svc = UpstoxService()

# Cache for connection status
_status_cache = {"last_check": 0, "data": None}

class TechnicalAnalysisService:
    def __init__(self):
        pass

    def fetch_ohlcv(self, ticker: str, period="30d", interval="5m", data_provider="upstox", fallback_enabled=True):
        """Fetch OHLCV data. Uses Upstox by default, falls back to yfinance only if allowed."""
        import os
        import numpy as np
        if os.getenv("SIMULATION", "false").lower() == "true":
            # Generate more data for backtesting (500 rows)
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            # Use a fixed reference time for deterministic testing if needed, or now IST
            now = datetime.datetime.now(ist)

            # If current time is past 15:30, cap it at 15:30 today
            if now.hour > 15 or (now.hour == 15 and now.minute > 30):
                now = now.replace(hour=15, minute=30, second=0, microsecond=0)
            elif now.hour < 9:
                # If before market, cap at 15:30 yesterday
                now = (now - datetime.timedelta(days=1)).replace(hour=15, minute=30, second=0, microsecond=0)

            dates = [now - datetime.timedelta(minutes=5*i) for i in range(500, 0, -1)]
            # Create oscillatory data to trigger both BUY and SHORT
            x = np.linspace(0, 10 * np.pi, 500)
            prices = 100 + 5 * np.sin(x)

            df = pd.DataFrame({
                "Open": prices + np.random.randn(500) * 0.1,
                "High": prices + 0.5,
                "Low": prices - 0.5,
                "Close": prices,
                "Volume": np.random.randint(1000, 5000, 500)
            }, index=dates)
            
            # SKEPTIC: Backtester requires a Signal column in simulation mode
            df['Signal'] = 0
            # Create some mock signals: BUY at troughs, SHORT at peaks
            df.loc[df['Close'] < 96, 'Signal'] = 1
            df.loc[df['Close'] > 104, 'Signal'] = -1
            
            return df
        df = None
        # SKEPTIC: GIFT Nifty is known to fail on Upstox Historical API. Skip to trigger fallback immediately.
        is_gift = "GIFT" in ticker.upper()
        
        if not is_gift and data_provider == "upstox" and _upstox_svc.is_authenticated:
            # Existing Upstox Logic
            days = 30
            if period.endswith("d"): days = int(period.replace("d", ""))
            elif period.endswith("mo"): days = int(period.replace("mo", "")) * 30
            
            if interval == "1m": unit, num_int = "minutes", "1"
            elif interval == "5m": unit, num_int = "minutes", "5"
            elif interval == "15m": unit, num_int = "minutes", "15"
            elif interval == "30m": unit, num_int = "minutes", "30"
            elif interval == "1h": unit, num_int = "hours", "1"
            elif interval == "1d": unit, num_int = "days", "1"
            else: unit, num_int = "minutes", "5"
            
            df_upstox = _upstox_svc.fetch_ohlcv(ticker, days=days, interval=num_int, unit=unit)
            
            if df_upstox is not None and not df_upstox.empty:
                # Resampling Logic
                if interval == "5m" and len(df_upstox) > 15:
                    df_upstox = df_upstox.resample('5min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
                elif interval == "15m" and len(df_upstox) > 15:
                    df_upstox = df_upstox.resample('15min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
                elif interval == "1h" and len(df_upstox) > 60:
                    df_upstox = df_upstox.resample('1h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

                df = df_upstox
                df.columns = [c.capitalize() for c in df.columns]

        if df is None:
            if not fallback_enabled and data_provider == "upstox":
                logger.error(f"Upstox data fetch failed for {ticker} and Fallback is DISABLED.")
                return pd.DataFrame() # Return empty to trigger error in UI

            # Fallback to yfinance
            logger.info(f"Using yfinance for {ticker} (Provider: {data_provider}, Fallback: {fallback_enabled})")
            try:
                # SKEPTIC: GIFT Nifty is best represented by ^NSEI proxy on yfinance for charting
                if "GIFT" in ticker.upper():
                    yf_ticker = "^NSEI"
                else:
                    yf_ticker = ticker if ticker.endswith(".NS") or "^" in ticker else f"{ticker}.NS"
                
                df = yf.download(yf_ticker, period=period, interval=interval, auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.dropna(inplace=True)
                df.columns = [str(c).capitalize() for c in df.columns]
            except Exception as e:
                logger.error(f"Data fetch failed globally for {ticker}: {e}")
                df = pd.DataFrame()
        
        return df

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
        change_pct = float(((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100) if len(df) > 1 else 0.0

        latest = df.iloc[-1]

        # Robust Column Extraction for Indicators
        def get_col(pattern, default=0):
            cols = [c for c in df.columns if pattern in c]
            return float(latest[cols[0]]) if cols else default

        return {
            "close": float(latest['Close']),
            "ema_9": get_col('EMA_9'),
            "ema_21": get_col('EMA_21'),
            "rsi_14": get_col('RSI_14', 50),
            "macd": get_col('MACD_', 0),
            "macd_hist": get_col('MACDh_', 0),
            "macd_signal": get_col('MACDs_', 0),
            "bb_upper": get_col('BBU_', 0),
            "bb_lower": get_col('BBL_', 0),
            "bb_mid": get_col('BBM_', 0),
            "adx_14": get_col('ADX_', 0),
            "vwap": float(current_vwap),
            "vol_surge": float(round(vol_surge, 2)),
            "change_pct": round(change_pct, 2),
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
        
        # 1. Trend Alignment (0.4 weight)
        is_bullish = ema9 > ema21 and close > vwap
        is_bearish = ema9 < ema21 and close < vwap
        
        if is_bullish or is_bearish:
            score += 0.4  # Strong alignment in either direction
        elif (ema9 > ema21 and close < vwap) or (ema9 < ema21 and close > vwap):
            score += 0.1  # Mismatched trend/price (weak)
            
        # 2. Momentum / ADX (0.2 weight)
        if adx > 25:
            score += 0.2  # High momentum
        elif adx > 15:
            score += 0.1 # Moderate momentum
            
        # 3. RSI Oscillators (0.2 weight) - Hard Blocks for overextension
        if is_bullish:
            if rsi > 75: score = 0 # Overbought, skip AI
            elif 40 <= rsi <= 70: score += 0.2
        elif is_bearish:
            if rsi < 25: score = 0 # Oversold, skip AI
            elif 30 <= rsi <= 60: score += 0.2
        else:
            if 40 <= rsi <= 60: score += 0.1
            
        # 4. MACD Directional Bias (0.2 weight)
        if (is_bullish and macd_hist > 0) or (is_bearish and macd_hist < 0):
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
        if ema_9 > ema_21 and close > vwap: 
            score += 3
        elif ema_9 < ema_21 and close < vwap: 
            score -= 3
        elif close < vwap and close < ema_9: # Strong bearish pressure
            score -= 2
        elif close > vwap and close > ema_9: # Strong bullish pressure
            score += 2

        # 2. Momentum (MACD)
        if macd_hist > 0.2: score += 1
        elif macd_hist < -0.2: score -= 1

        # 3. Strength (ADX)
        if adx > 25:
            if score > 0: score += 1
            elif score < 0: score -= 1

        # 4. Exhaustion (RSI) - Hard Blocks
        if rsi > 75: 
            if score > 0: score = 0 # Block Buy if overbought
            score -= 2
        elif rsi < 25: 
            if score < 0: score = 0 # Block Short if oversold
            score += 2

        # 5. Volume Surge
        if vol_surge > 1.5:
            if score > 0: score += 1
            elif score < 0: score -= 1

        if score >= 4: return "STRONG BUY"
        if score >= 2: return "BUY"
        if score <= -4: return "STRONG SHORT SELL"
        if score <= -2: return "SHORT SELL"
        return "NEUTRAL"

    def get_connection_status(self) -> dict:
        """Unified status for all external data & AI providers with caching."""
        global _status_cache
        import time
        now = time.time()
        
        # Cache for 30 seconds to avoid blocking WebSocket loop with network calls
        if _status_cache["data"] and (now - _status_cache["last_check"] < 30):
            return _status_cache["data"]

        from services.quota_service import quota_svc
        # fetch_profile call can be slow, but it's a direct API call
        upstox_profile = _upstox_svc.fetch_profile()
        
        status = {
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
        
        _status_cache["last_check"] = now
        _status_cache["data"] = status
        return status

    def get_chart_payload(self, ticker: str, interval: str = "5m"):
        """
        Unified method to prepare the complete V3 Chart Payload.
        Includes OHLC, Projections, and all ML Indicators.
        """
        import os
        import time
        from services.price_projector import price_projector
        from services.advanced_indicators import classifier, adaptive_st

        # 1. Simulation Check
        if os.getenv("SIMULATION", "false").lower() == "true":
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            now_dt = datetime.datetime.now(ist)
            
            # Cap at 15:30
            if now_dt.hour > 15 or (now_dt.hour == 15 and now_dt.minute > 30):
                now_dt = now_dt.replace(hour=15, minute=30, second=0, microsecond=0)
            elif now_dt.hour < 9:
                now_dt = (now_dt - datetime.timedelta(days=1)).replace(hour=15, minute=30, second=0, microsecond=0)
            
            now = int(now_dt.timestamp())
            step = 60 if interval == '1m' else 300 if interval == '5m' else 900 if interval == '15m' else 3600 if interval == '1h' else 86400
            mock_ohlc = [{"time": now - i * step, "open": 100+i%5, "high": 105+i%5, "low": 95+i%5, "close": 102+i%5} for i in range(100, 0, -1)]
            return {
                "instrument_key": ticker,
                "current_price": 102.0,
                "ohlc": mock_ohlc,
                "adx_series": [{"time": c['time'], "value": 25.0 + (i%10)} for i, c in enumerate(mock_ohlc)],
                "rsi_series": [{"time": c['time'], "value": 45.0 + (i%5)} for i, c in enumerate(mock_ohlc)],
                "ml_adaptive_st": {"time": [c['time'] for c in mock_ohlc], "value": [98+i%2 for i in range(100)], "trend": [1]*50 + [-1]*50, "regime": [1]*33 + [2]*33 + [3]*34},
                "ml_lorentzian": [{"time": c['time'], "signal": 1 if i % 20 == 0 else -1 if i % 25 == 0 else 0, "score": 0.5} for i, c in enumerate(mock_ohlc)],
                "ml_knn": [{"time": c['time'], "value": 100+i%3, "trend": 1 if i%10 < 5 else -1} for i, c in enumerate(mock_ohlc)],
                "vwap": 101.0,
                "ema_9": 101.5,
                "ema_21": 100.5,
                "interval": interval,
                "projection": [102.5, 103.0, 103.5],
                "proj_timestamps": [now + i * step for i in range(1, 4)],
                "upper_band": [104, 105, 106],
                "lower_band": [101, 101, 101]
            }

        # 2. Fetch Production Data
        from services.state import AppState
        state = AppState() # Temp instance to get settings, ideally passed in
        
        period = "1d" if interval == "1m" else "5d" if interval in ["5m", "15m"] else "1mo" if interval == "1h" else "6mo"
        df = self.fetch_ohlcv(ticker, period, interval, 
                             data_provider=getattr(state, 'data_provider', 'upstox'),
                             fallback_enabled=getattr(state, 'fallback_data', True))
        
        if df is None or df.empty:
            return {"error": f"No data available for {ticker}"}

        # SKEPTIC: Truncate data past 15:30 IST for intraday to prevent 'Ghost Data' reported by user
        if interval in ["1m", "5m", "15m", "1h"]:
            try:
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
                else:
                    df.index = df.index.tz_convert('Asia/Kolkata')
                
                # Filter rows where time is <= 15:30
                df = df[df.index.map(lambda x: (x.hour < 15) or (x.hour == 15 and x.minute <= 30))]
                # Convert back to UTC for consistency if needed, but int(t.timestamp()) is UTC epoch anyway
            except Exception as e:
                logger.error(f"SKEPTIC: Ghost data filtering failed: {e}")

        # 3. Compute Basic Indicators
        indicators = self.compute_indicators(df)
        
        # 4. Generate Projections
        proj_interval = 1 if interval == "1m" else 5
        proj_res = price_projector.generate_projection(df, interval_minutes=proj_interval)
        
        # 5. Compute ML Series
        from services.advanced_indicators import classifier, adaptive_st, knn_forecaster
        lz_series = classifier.classify_series(df, window=500)
        st_series = adaptive_st.calculate(df, window=500)
        knn_series = knn_forecaster.get_historical_shading(df, window=500)
        
        # 6. Extract ADX & RSI
        adx_series = []
        rsi_series = []
        import math
        adx_col = [c for c in df.columns if 'ADX_' in c]
        if adx_col:
            adx_vals = df[adx_col[0]].tail(500)
            for t, v in adx_vals.items():
                if not math.isnan(v):
                    adx_series.append({"time": int(t.timestamp()), "value": float(v)})
        
        rsi_col = [c for c in df.columns if 'RSI_' in c]
        if rsi_col:
            rsi_vals = df[rsi_col[0]].tail(500)
            for t, v in rsi_vals.items():
                if not math.isnan(v):
                    rsi_series.append({"time": int(t.timestamp()), "value": float(v)})

        # 7. Final Assembly
        current_price = indicators.get("close") if indicators else float(df['Close'].iloc[-1] if 'Close' in df.columns else df['close'].iloc[-1] if 'close' in df.columns else 0)
        
        # Safe OHLC extraction
        def safe_get(row, keys, default=0):
            for k in keys:
                if k in row: return float(row[k])
            return default

        ohlc_list = []
        for t, r in df.tail(500).iterrows():
            ohlc_list.append({
                "time": int(t.timestamp()),
                "open": safe_get(r, ['Open', 'open']),
                "high": safe_get(r, ['High', 'high']),
                "low": safe_get(r, ['Low', 'low']),
                "close": safe_get(r, ['Close', 'close'])
            })

        return {
            "instrument_key": ticker,
            "current_price": current_price,
            "ohlc": ohlc_list,
            "adx_series": adx_series,
            "rsi_series": rsi_series,
            "ml_adaptive_st": st_series,
            "ml_lorentzian": lz_series,
            "ml_knn": knn_series,
            "vwap": indicators.get("vwap") if indicators else current_price,
            "interval": interval,
            "projection": proj_res.get("projection"),
            "proj_timestamps": [int(pd.to_datetime(t).timestamp()) for t in proj_res.get("timestamps", [])],
            "upper_band": proj_res.get("upper_band"),
            "lower_band": proj_res.get("lower_band")
        }

    def fetch_fundamentals(self, ticker: str):
        """Fetch basic fundamental data via yfinance."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
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
