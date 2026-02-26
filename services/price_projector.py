"""
Intraday Price Projection Engine
=================================
Generates realistic waveform price projections for the rest of the trading day.

Uses 3 models blended together:
  1. Fourier Decomposition (40%) — cyclic patterns (morning rally, lunch dip)
  2. ARIMA(2,1,2)           (30%) — short-term trend extrapolation
  3. VWAP/EMA Gravity       (30%) — mean-reversion anchoring

Plus Monte Carlo confidence bands (±1σ) from 500 simulations.

Output: arrays of timestamps, projected prices, upper/lower confidence bands.
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Try importing ARIMA; graceful fallback if missing
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False
    logger.warning("statsmodels not installed. ARIMA projection disabled.")


class PriceProjector:
    """
    Generates intraday price projections with realistic waveform shape.
    """

    # NSE market hours in minutes from midnight
    MARKET_OPEN_MIN = 9 * 60 + 15   # 9:15 AM
    MARKET_CLOSE_MIN = 15 * 60       # 3:00 PM
    TOTAL_TRADING_MINS = MARKET_CLOSE_MIN - MARKET_OPEN_MIN  # 345 minutes

    def generate_projection(self, df: pd.DataFrame, interval_minutes: int = 1) -> dict:
        """
        Generate price projection from current time to market close.

        Args:
            df: OHLCV DataFrame with DatetimeIndex (today's intraday data)
            interval_minutes: granularity of the projection (1 or 5 min)

        Returns:
            {
                "ohlc": [...],              # Historical OHLC for chart
                "projection": [...],        # Projected close prices
                "upper_band": [...],        # Upper confidence (mean + 1σ)
                "lower_band": [...],        # Lower confidence (mean - 1σ)
                "timestamps": [...],        # ISO timestamps for projection
                "current_price": float,
                "vwap": float,
                "models_used": [str],
            }
        """
        if df is None or df.empty or len(df) < 10:
            return {"error": "Insufficient data for projection"}

        # Extract today's data only
        today = df.index[-1].date()
        today_df = df[df.index.date == today].copy()
        if len(today_df) < 10:
            today_df = df.tail(100).copy()  # Fallback: use last 100 candles

        prices = today_df['Close'].values.astype(float)
        current_price = float(prices[-1])
        n_observed = len(prices)

        # How many candles to project (remaining until 3 PM)
        now = datetime.now()
        current_min = now.hour * 60 + now.minute
        remaining_mins = max(10, self.MARKET_CLOSE_MIN - current_min)
        n_forecast = remaining_mins // interval_minutes

        if n_forecast <= 0:
            n_forecast = 30  # Minimum forecast window

        # Calculate VWAP as gravity anchor
        vwap = self._compute_vwap(today_df)

        # ─── Model 1: Fourier Decomposition (40%) ───────────────
        fourier_proj = self._fourier_projection(prices, n_forecast)

        # ─── Model 2: ARIMA (30%) ───────────────────────────────
        arima_proj = self._arima_projection(prices, n_forecast)

        # ─── Model 3: VWAP/EMA Gravity (30%) ───────────────────
        gravity_proj = self._gravity_projection(prices, n_forecast, vwap)

        # ─── Blend ──────────────────────────────────────────────
        models_used = ["Fourier"]
        weights = [0.4]

        if arima_proj is not None:
            models_used.append("ARIMA(2,1,2)")
            weights.append(0.3)
        else:
            weights[0] = 0.55  # Redistribute ARIMA weight

        models_used.append("VWAP Gravity")
        weights.append(0.3 if arima_proj is not None else 0.45)

        # Stack projections
        proj_stack = [fourier_proj]
        if arima_proj is not None:
            proj_stack.append(arima_proj)
        proj_stack.append(gravity_proj)

        # Weighted average
        blended = np.zeros(n_forecast)
        for proj, w in zip(proj_stack, weights):
            blended += proj[:n_forecast] * w

        # ─── Monte Carlo Confidence Bands ───────────────────────
        upper, lower = self._monte_carlo_bands(prices, n_forecast, blended)

        # ─── Build Timestamps ───────────────────────────────────
        last_time = today_df.index[-1]
        proj_timestamps = []
        for i in range(1, n_forecast + 1):
            t = last_time + timedelta(minutes=i * interval_minutes)
            proj_timestamps.append(t.strftime("%Y-%m-%dT%H:%M:%S"))

        # ─── Historical OHLC for chart ──────────────────────────
        ohlc_data = []
        for idx, row in today_df.iterrows():
            ohlc_data.append({
                "time": idx.strftime("%Y-%m-%dT%H:%M:%S"),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
            })

        return {
            "ohlc": ohlc_data,
            "projection": [round(float(p), 2) for p in blended],
            "upper_band": [round(float(u), 2) for u in upper],
            "lower_band": [round(float(l), 2) for l in lower],
            "timestamps": proj_timestamps,
            "current_price": round(current_price, 2),
            "vwap": round(float(vwap), 2),
            "models_used": models_used,
        }

    def _fourier_projection(self, prices: np.ndarray, n_forecast: int) -> np.ndarray:
        """
        Extract top frequency components from observed prices and extrapolate.
        This creates the realistic 'wavy' shape in the projection.
        """
        n = len(prices)

        # Detrend: remove linear trend to isolate cyclical component
        t = np.arange(n)
        slope, intercept = np.polyfit(t, prices, 1)
        detrended = prices - (slope * t + intercept)

        # FFT
        fft_vals = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(n)

        # Keep top 5 frequency components (dominant cycles)
        magnitudes = np.abs(fft_vals)
        top_k = min(5, len(magnitudes) - 1)
        top_indices = np.argsort(magnitudes[1:])[-top_k:] + 1  # Skip DC component

        # Zero out non-dominant frequencies
        filtered = np.zeros_like(fft_vals)
        filtered[0] = fft_vals[0]  # Keep DC
        for idx in top_indices:
            filtered[idx] = fft_vals[idx]

        # Reconstruct and extend
        # Create extended time array
        t_extended = np.arange(n + n_forecast)
        projection = np.zeros(n + n_forecast)

        # Add each frequency component
        for idx in top_indices:
            amp = np.abs(fft_vals[idx]) * 2 / n
            phase = np.angle(fft_vals[idx])
            freq = freqs[idx]
            projection += amp * np.cos(2 * np.pi * freq * t_extended + phase)

        # Add the linear trend back
        trend_extended = slope * t_extended + intercept
        projection += trend_extended

        # Ensure continuity: shift so projection starts at current price
        offset = prices[-1] - projection[n - 1]
        projection += offset

        # Apply exponential decay to dampen confidence as we go further out
        decay = np.exp(-np.arange(n_forecast) * 0.005)
        forecast = projection[n:]

        # Pull toward last price with increasing strength over time
        forecast = forecast * decay + prices[-1] * (1 - decay)

        return forecast

    def _arima_projection(self, prices: np.ndarray, n_forecast: int) -> np.ndarray:
        """
        ARIMA(2,1,2) forecast. Returns None if statsmodels unavailable.
        """
        if not HAS_ARIMA or len(prices) < 30:
            return None

        try:
            # Use log returns for stationarity
            model = ARIMA(prices, order=(2, 1, 2))
            fitted = model.fit(method_kwargs={"maxiter": 50})
            forecast = fitted.forecast(steps=n_forecast)
            return np.array(forecast)
        except Exception as e:
            logger.warning(f"ARIMA fit failed: {e}")
            return None

    def _gravity_projection(self, prices: np.ndarray, n_forecast: int,
                             vwap: float) -> np.ndarray:
        """
        Mean-reversion projection: current price pulled toward VWAP
        with exponential decay. Prevents projections from drifting unrealistically.
        """
        current = prices[-1]
        t = np.arange(1, n_forecast + 1)

        # Exponential decay toward VWAP
        # Faster decay if price is far from VWAP (strong gravity)
        distance_pct = abs(current - vwap) / vwap if vwap != 0 else 0
        decay_rate = 0.003 + distance_pct * 0.01  # Adaptive decay

        decay = 1 - np.exp(-decay_rate * t)
        projection = current + (vwap - current) * decay

        # Add small noise for realistic look
        volatility = np.std(np.diff(prices)) if len(prices) > 2 else 0
        noise = np.random.normal(0, volatility * 0.3, n_forecast)
        # Smooth the noise
        from scipy.ndimage import uniform_filter1d
        noise = uniform_filter1d(noise, size=5)
        projection += noise

        return projection

    def _monte_carlo_bands(self, prices: np.ndarray, n_forecast: int,
                            center_line: np.ndarray) -> tuple:
        """
        Run 500 Monte Carlo simulations to generate confidence bands.
        Uses historical volatility from the observed data.
        """
        returns = np.diff(prices) / prices[:-1]
        mu = np.mean(returns)
        sigma = np.std(returns)

        n_sims = 500
        all_paths = np.zeros((n_sims, n_forecast))

        for i in range(n_sims):
            random_returns = np.random.normal(mu, sigma, n_forecast)
            path = np.zeros(n_forecast)
            path[0] = prices[-1] * (1 + random_returns[0])
            for j in range(1, n_forecast):
                path[j] = path[j - 1] * (1 + random_returns[j])
            all_paths[i] = path

        # 1-sigma bands
        upper = np.percentile(all_paths, 84, axis=0)  # ~mean + 1σ
        lower = np.percentile(all_paths, 16, axis=0)  # ~mean - 1σ

        # Anchor bands to the center line
        mc_center = np.mean(all_paths, axis=0)
        band_width_upper = upper - mc_center
        band_width_lower = mc_center - lower

        upper_band = center_line + band_width_upper
        lower_band = center_line - band_width_lower

        return upper_band, lower_band

    def _compute_vwap(self, df: pd.DataFrame) -> float:
        """Compute VWAP from OHLCV data."""
        try:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            vp = tp * df['Volume']
            total_vol = df['Volume'].sum()
            if total_vol > 0:
                return float(vp.sum() / total_vol)
        except Exception:
            pass
        return float(df['Close'].iloc[-1])


# Module singleton
price_projector = PriceProjector()
