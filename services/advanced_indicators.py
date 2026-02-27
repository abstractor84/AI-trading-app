import numpy as np
import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

class LorentzianClassifier:
    """
    Advanced Non-Linear Classifier based on Lorentzian Distance and k-NN.
    Inspired by modern 'AI' indicators used in quantitative trading.
    """
    def __init__(self, k=8, lookback=2000):
        self.k = k
        self.lookback = lookback

    def _lorentzian_distance(self, x1, x2):
        """Calculate Lorentzian distance between two feature vectors."""
        # dist = sum(log(1 + abs(x1_i - x2_i)))
        return np.sum(np.log(1 + np.abs(x1 - x2)))

    def prepare_features(self, df: pd.DataFrame):
        """Extract multi-dimensional feature set for the classifier."""
        if df.empty or len(df) < 50:
            return None

        features = pd.DataFrame(index=df.index)

        # Feature 1: RSI (Relative Strength Index)
        features['rsi'] = ta.rsi(df['Close'], length=14)

        # Feature 2: WT (Wave Trend - custom implementation)
        # WT1 = ema(avg(h,l,c), 10)
        # WT2 = ema(abs(avg(h,l,c) - active_avg), 10)
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = ta.ema(ap, length=10)
        d = ta.ema(np.abs(ap - esa), length=10)
        ci = (ap - esa) / (0.015 * d)
        features['wt'] = ta.ema(ci, length=21)

        # Feature 3: ADX (Trend Strength)
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx_df is not None:
            features['adx'] = adx_df['ADX_14']
        else:
            features['adx'] = 20

        # Feature 4: CCI (Commodity Channel Index)
        features['cci'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)

        # Feature 5: Normalized Volatility
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        features['vol'] = (atr / df['Close']) * 100

        features.dropna(inplace=True)
        return features

    def classify(self, df: pd.DataFrame):
        """
        Classify the current bar based on historical k-NN patterns.
        Returns: { 'signal': 'BUY'|'SELL'|'NEUTRAL', 'score': float }
        """
        features = self.prepare_features(df)
        if features is None or len(features) < self.k:
            return {"signal": "NEUTRAL", "score": 0}

        # Use limited history for speed/relevance
        data = features.tail(self.lookback).values
        current_v = data[-1]
        history = data[:-1]

        # Calculate Lorentzian distances to all historical points
        distances = []
        for i in range(len(history)):
            d = self._lorentzian_distance(current_v, history[i])
            distances.append(d)

        distances = np.array(distances)
        # Find indices of k smallest distances
        k_indices = np.argsort(distances)[:self.k]

        # Look at the price change following each neighbor (1 bar forward)
        # We need the alignment with original df to check price movement
        feature_indices = features.tail(self.lookback).index
        
        up_count = 0
        down_count = 0
        
        # We want to know if 'Close' went up or down in the NEXT bar from the neighbor's time
        for idx in k_indices:
            orig_idx = feature_indices[idx]
            # Find location of this index in the original dataframe
            loc = df.index.get_loc(orig_idx)
            if loc + 1 < len(df):
                change = df['Close'].iloc[loc + 1] - df['Close'].iloc[loc]
                if change > 0:
                    up_count += 1
                elif change < 0:
                    down_count += 1

        score = (up_count - down_count) / self.k
        
        signal = "NEUTRAL"
        if score > 0.6: # High confidence threshold
            signal = "BUY"
        elif score < -0.6:
            signal = "SELL"

        return {
            "signal": signal,
            "score": round(score, 2),
            "neighbors_up": up_count,
            "neighbors_down": down_count
        }

# Singleton instance
classifier = LorentzianClassifier()
