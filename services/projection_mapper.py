import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ProjectionService:
    def __init__(self):
        pass

    def calculate_projections(self, ticker: str, df: pd.DataFrame):
        """
        Calculate various statistical projections for the 3 PM IST close.
        Assumes df contains 1m or 5m intraday OHLCV data.
        """
        if df.empty or len(df) < 5:
            return None

        # 1. Prepare time indices for regression (minutes from start of day)
        df = df.copy()
        df.reset_index(inplace=True)
        time_index = np.arange(len(df))
        prices = df['Close'].values

        # 2. Linear Regression (Standard Trend)
        # y = mx + c
        slope, intercept = np.polyfit(time_index, prices, 1)
        
        # 3. Quadratic Fit (Curvature)
        # y = ax^2 + bx + c
        z_quad = np.polyfit(time_index, prices, 2)
        p_quad = np.poly1d(z_quad)

        # 4. Momentum Scaling (Recent 20 mins)
        recent_window = min(len(df), 20)
        recent_prices = prices[-recent_window:]
        recent_time = np.arange(recent_window)
        m_recent, c_recent = np.polyfit(recent_time, recent_prices, 1)

        # 5. Define 3 PM IST (assuming Market opens at 9:15 AM)
        # Approx 345 minutes from open (9:15 to 15:00 = 5h 45m = 345m)
        target_index = 345 # Minute index from open

        # Projection Calculations
        proj_linear = float(slope * target_index + intercept)
        proj_quad = float(p_quad(target_index))
        
        # Recent momentum projection (capped to prevent extreme outliers)
        # current price + (rate * remaining minutes)
        remaining_mins = max(0, 345 - len(df))
        proj_momentum = float(prices[-1] + (m_recent * remaining_mins))

        # 6. Pivot Calculation (H + L + C) / 3 from today's data
        day_high = df['High'].max()
        day_low = df['Low'].min()
        day_close = df['Close'].iloc[-1]
        pivot_point = float((day_high + day_low + day_close) / 3)

        # 7. VWAP Proxy (Volume Weighted Average so far)
        vwap = float((df['Close'] * df['Volume']).sum() / df['Volume'].sum()) if df['Volume'].sum() > 0 else day_close

        # 8. Ensemble Average (Weighted blend)
        # We give more weight to VWAP and Pivot as they are strong intraday anchors
        ensemble = (proj_linear * 0.15) + (proj_quad * 0.15) + (proj_momentum * 0.1) + (pivot_point * 0.3) + (vwap * 0.3)

        return {
            "linear_regression": round(proj_linear, 2),
            "quadratic_fit": round(proj_quad, 2),
            "momentum_vector": round(proj_momentum, 2),
            "pivot_point": round(pivot_point, 2),
            "vwap_anchor": round(vwap, 2),
            "ensemble_target": round(float(ensemble), 2),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

projection_svc = ProjectionService()
