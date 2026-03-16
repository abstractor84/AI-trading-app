import numpy as np
import pandas as pd
import pandas_ta as ta
import logging
import time
import warnings

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class LorentzianClassifier:
    """
    Machine Learning: Lorentzian Classification
    Uses Scikit-Learn NearestNeighbors with Batch Querying.
    TV Parity: Default lookback limit is 2000 bars.
    """
    def __init__(self, k=8, lookback=2000):
        self.k = k
        self.lookback = min(lookback, 2000)
        self.scaler = StandardScaler()

    def prepare_features(self, df: pd.DataFrame, use_volatility=True):
        if df.empty or len(df) < 50: return None
        try:
            features = pd.DataFrame(index=df.index)
            rsi = ta.rsi(df['Close'], length=14)
            if rsi is None or rsi.empty:
                logger.warning("RSI calculation failed")
            features['rsi'] = rsi
            
            cci = ta.cci(df['High'], df['Low'], df['Close'], length=20)
            if cci is None or cci.empty:
                logger.warning("CCI calculation failed")
            features['cci'] = cci
            
            adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            if adx_df is None or adx_df.empty:
                logger.warning("ADX calculation failed")
            features['adx'] = adx_df['ADX_14'] if adx_df is not None else 20
            
            ap = (df['High'] + df['Low'] + df['Close']) / 3
            esa = ta.ema(ap, length=10)
            d = ta.ema(np.abs(ap - esa), length=10)
            ci = (ap - esa) / (0.015 * d)
            features['wt'] = ta.ema(ci, length=21)
            
            if use_volatility:
                atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                features['vol'] = (atr / df['Close']) * 100
            else:
                features['vol'] = 0

            features['ema50'] = ta.ema(df['Close'], length=50)
            st_df = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
            features['st_dir'] = st_df['SUPERTd_10_3'] if st_df is not None else 0
            
            features.dropna(inplace=True)
            return features if not features.empty else None
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def classify(self, df: pd.DataFrame, signal_threshold=0.5):
        """
        Classify the current bar for direct backend scanning.
        TV Parity: Returns { 'signal': 'BUY'|'SHORT SELL'|'NEUTRAL', 'score': float }
        """
        features = self.prepare_features(df)
        if features is None or len(features) < self.k:
            return {"signal": "NEUTRAL", "score": 0}

        knn_features = ['rsi', 'wt', 'adx', 'cci', 'vol']
        vals = features[knn_features].values
        indices = features.index
        
        # Scaling
        scaled_vals = self.scaler.fit_transform(vals)
        
        current_v = scaled_vals[-1].reshape(1, -1)
        history = scaled_vals[:-1]
        
        if len(history) < self.k:
            return {"signal": "NEUTRAL", "score": 0}

        # Fit model on history
        model = NearestNeighbors(n_neighbors=self.k, metric='manhattan', algorithm='auto')
        model.fit(history)
        
        distances, neighbor_indices = model.kneighbors(current_v)
        
        # Target: Price direction 4 bars later (historical direction)
        closes = df.loc[indices, 'Close'].values
        y_history = []
        for j in range(len(indices) - 1):
            if j + 4 < len(indices):
                y_history.append(1 if closes[j+4] > closes[j] else -1)
            else:
                y_history.append(0)
        y_history = np.array(y_history)

        neigh_idx = neighbor_indices[0]
        weights = np.exp(-distances[0])
        scores = y_history[neigh_idx] * weights
        score = np.sum(scores) / np.sum(weights) if np.sum(weights) > 0 else 0
        
        current_close = df['Close'].iloc[-1]
        ema50 = features['ema50'].iloc[-1]
        st_dir = features['st_dir'].iloc[-1]

        signal = "NEUTRAL"
        adx = features['adx'].iloc[-1]
        if score >= signal_threshold and current_close > ema50 and st_dir == 1 and adx > 20:
            signal = "BUY"
        elif score <= -signal_threshold and current_close < ema50 and st_dir == -1 and adx > 20:
            signal = "SHORT SELL"

        return {
            "signal": signal,
            "score": round(float(score), 2)
        }

    def classify_series(self, df: pd.DataFrame, window=200, **kwargs):
        start_time = time.time()
        params = kwargs.get('params', {})
        k = int(params.get('k', self.k))
        lookback = int(params.get('lookback', self.lookback))
        # TV Parity: Ensure lookback is respected
        lookback = min(max(lookback, 500), 4000) 
        threshold = float(params.get('threshold', 0.5))

        features = self.prepare_features(df, use_volatility=params.get('use_volatility', 'true') == 'true')
        
        # If not enough features even for basic k, return empty/neutral
        if features is None or len(features) < k:
            start_i = max(0, len(df) - window)
            return [{"time": int(df.index[i].timestamp()), "signal": 0, "score": 0} for i in range(start_i, len(df))]

        # Adaptive Lookback: Don't fail if we have fewer bars than 2000
        # This is CRITICAL for 5-day charts which only have ~375 bars
        actual_lookback = min(len(features) - k - 1, lookback)
        if actual_lookback < k: actual_lookback = k # Minimum safety

        knn_features = ['rsi', 'wt', 'adx', 'cci', 'vol']
        vals = features[knn_features].values
        indices = features.index
        
        # Scaling for distance-based ML
        scaled_vals = self.scaler.fit_transform(vals)
        
        total_len = len(features)
        # We want to produce results for the last 'window' bars
        start_idx = max(0, total_len - window)
        
        # Targets for the whole set (Lookahead 4)
        closes = df.loc[indices, 'Close'].values
        y_all = []
        for j in range(total_len):
            if j + 4 < total_len:
                y_all.append(1 if closes[j+4] > closes[j] else -1)
            else:
                y_all.append(0)
        y_all = np.array(y_all)

        ema50 = features['ema50'].values
        st_dir = features['st_dir'].values
        adx_vals = features['adx'].values
        
        results = []
        
        # SKEPTIC: Rolling Walk-Forward for Lorentzian
        # Fit every 20 bars to maintain performance
        step = 20
        for i in range(start_idx, total_len, step):
            end_batch = min(i + step, total_len)
            
            # Training pool: up to 'lookback' bars before current batch
            train_end = i - 1
            train_start = max(0, train_end - actual_lookback)
            
            if (train_end - train_start) < k:
                # Not enough training data yet, pad with neutral
                for j in range(i, end_batch):
                    results.append({"time": int(indices[j].timestamp()), "signal": 0, "score": 0})
                continue

            X_train = scaled_vals[train_start:train_end]
            y_train = y_all[train_start:train_end]
            
            # Scikit-Learn Batch Query
            model = NearestNeighbors(n_neighbors=k, metric='manhattan', algorithm='auto')
            model.fit(X_train)
            
            X_query = scaled_vals[i:end_batch]
            distances, neighbor_indices = model.kneighbors(X_query)
            
            for j in range(len(X_query)):
                curr_idx = i + j
                neigh_idx = neighbor_indices[j]
                
                # Gaussian Kernel Weighting
                weights = np.exp(-distances[j])
                scores = y_train[neigh_idx] * weights
                score = np.sum(scores) / np.sum(weights) if np.sum(weights) > 0 else 0
                
                signal = 0
                # SKEPTIC: Relaxed filters for Intraday frequency (ADX 15+ instead of 20)
                if score >= threshold and closes[curr_idx] > ema50[curr_idx] and st_dir[curr_idx] == 1 and adx_vals[curr_idx] > 15: 
                    signal = 1
                elif score <= -threshold and closes[curr_idx] < ema50[curr_idx] and st_dir[curr_idx] == -1 and adx_vals[curr_idx] > 15: 
                    signal = -1
                    
                results.append({
                    "time": int(indices[curr_idx].timestamp()),
                    "signal": int(signal),
                    "score": round(float(score), 2)
                })

        logger.debug(f"Lorentzian ML finished in {time.time() - start_time:.4f}s")
        return results


class AdaptiveSuperTrend:
    """
    Machine Learning: Adaptive SuperTrend (AlgoAlpha Style)
    -------------------------------------------------------
    Uses K-Means clustering on ATR to identify volatility regimes.
    Adapts the SuperTrend multiplier dynamically based on the current regime.
    """
    def __init__(self):
        self.atr_period = 10
        self.factor = 3.0
        self.training_len = 100
        self.percentiles = [0.25, 0.5, 0.75] # Low, Med, High

    def calculate(self, df: pd.DataFrame, window=200, params=None):
        if params:
            self.atr_period = int(params.get('atr_period', params.get('period', self.atr_period)))
            self.factor = float(params.get('factor', self.factor))
            self.training_len = int(params.get('training_len', self.training_len))
            self.percentiles = [
                float(params.get('p_low', 0.25)),
                float(params.get('p_med', 0.5)),
                float(params.get('p_high', 0.75))
            ]

        if df.empty or len(df) < self.atr_period * 2: return None
        
        # 1. Calculate ATR
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=self.atr_period)
        if atr is None or atr.isna().all(): return None
        
        atr_clean = atr.dropna()
        if len(atr_clean) < 20: return None
        
        # 2. ML Training (Regime Detection)
        atr_vals = atr_clean.values.reshape(-1, 1)
        train_window = min(len(atr_vals), 2000)
        train_data = atr_vals[-train_window:]
        
        initial_centroids = np.percentile(train_data, [p * 100 for p in self.percentiles]).reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=3, init=initial_centroids, n_init=1, random_state=42)
        clusters = kmeans.fit_predict(atr_vals)
        
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)
        rank_map = {sorted_indices[0]: 1, sorted_indices[1]: 2, sorted_indices[2]: 3}
        regimes = np.vectorize(rank_map.get)(clusters)
        
        # 3. Dynamic SuperTrend Calculation
        regime_adjustments = {1: 1.5, 2: 1.0, 3: 0.5}
        
        atr_idx = atr_clean.index
        df_valid = df.loc[atr_idx]
        hl2 = (df_valid['High'] + df_valid['Low']) / 2
        
        st_data = {"time": [], "value": [], "trend": [], "regime": []}
        curr_trend, prev_st = 1, 0
        prev_up, prev_lo = 1e10, -1e10
        
        for i, idx in enumerate(atr_idx):
            regime = regimes[i]
            m = self.factor * regime_adjustments[regime]
            
            v_atr, v_hl2, close = atr_clean.loc[idx], hl2.loc[idx], df_valid.loc[idx, 'Close']
            b_up, b_lo = v_hl2 + (m * v_atr), v_hl2 - (m * v_atr)
            
            if i == 0: c_up, c_lo, c_st = b_up, b_lo, b_lo
            else:
                c_up = b_up if (b_up < prev_up or df_valid.iloc[i-1]['Close'] > prev_up) else prev_up
                c_lo = b_lo if (b_lo > prev_lo or df_valid.iloc[i-1]['Close'] < prev_lo) else prev_lo
                if prev_st == prev_up: curr_trend = 1 if close > c_up else -1
                else: curr_trend = -1 if close < c_lo else 1
                c_st = c_lo if curr_trend == 1 else c_up
            
            st_data["time"].append(int(idx.timestamp()))
            st_data["value"].append(round(c_st, 2))
            st_data["trend"].append(curr_trend)
            st_data["regime"].append(int(regime))
            prev_st, prev_up, prev_lo = c_st, c_up, c_lo
            
        if window and len(st_data["time"]) > window:
            for k in st_data:
                st_data[k] = st_data[k][-window:]
                
        return st_data


class KNNTrendForecaster:
    """
    Machine Learning: kNN-Based Strategy (Capissimo Style)
    Uses KNeighborsClassifier for trend classification with multi-feature input.
    """
    def __init__(self, k=5, sequence_length=15):
        self.k = k
        self.seq_len = sequence_length
        self.scaler = StandardScaler()

    def _prepare_knn_features(self, df: pd.DataFrame):
        """Prepare multi-feature set for KNN trend classification."""
        try:
            features = pd.DataFrame(index=df.index)
            features['rsi'] = ta.rsi(df['Close'], length=14).fillna(50)
            
            # WaveTrend (WT)
            ap = (df['High'] + df['Low'] + df['Close']) / 3
            esa = ta.ema(ap, length=10)
            d = ta.ema(np.abs(ap - esa), length=10)
            ci = (ap - esa) / (0.015 * d)
            features['wt'] = ta.ema(ci, length=21).fillna(0)
            
            adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            features['adx'] = adx_df['ADX_14'].fillna(20) if adx_df is not None else 20
            
            features['cci'] = ta.cci(df['High'], df['Low'], df['Close'], length=20).fillna(0)
            
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).fillna(0)
            features['vol'] = (atr / df['Close']) * 100
            
            features.dropna(inplace=True)
            return features
        except Exception as e:
            logger.error(f"KNN Feature Prep Error: {e}")
            return None

    def forecast(self, df: pd.DataFrame, params=None):
        """Future price prediction using Scikit-Learn KNN Regressor"""
        from sklearn.neighbors import KNeighborsRegressor
        k = int(params.get('k', self.k)) if params else self.k
        sl = int(params.get('sequence_length', self.seq_len)) if params else self.seq_len
        horizon = 10
        
        if len(df) < sl + horizon + 50: return None
        closes = df['Close'].values[-1000:] # Last 1000 for training speed
        
        X = np.array([closes[i : i + sl] for i in range(len(closes) - sl - horizon)])
        y = np.array([closes[i + sl : i + sl + horizon] - closes[i + sl - 1] for i in range(len(closes) - sl - horizon)])
        
        if len(X) < k: return None
        
        model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        model.fit(X, y)
        current = closes[-sl:].reshape(1, -1)
        pred_deltas = model.predict(current)[0]
        
        return [closes[-1] + d for d in pred_deltas]

    def get_historical_shading(self, df: pd.DataFrame, window=500, **kwargs):
        """
        TradingView Parity: Continuous line with Scikit-Learn classification.
        SKEPTIC: Enhanced with a Rolling Walk-Forward training to ensure 
        signals adapt to recent market regimes.
        """
        params = kwargs.get('params', {})
        k = int(params.get('k', self.k))
        
        feat_df = self._prepare_knn_features(df)
        if feat_df is None or len(feat_df) < k + 50:
            return []
            
        indices = feat_df.index
        closes = df.loc[indices, 'Close'].values
        
        # Scaling
        scaled_vals = self.scaler.fit_transform(feat_df.values)
        
        total_len = len(feat_df)
        # We want to produce results for the last 'window' bars
        start_idx = max(0, total_len - window)
        
        # Target for the WHOLE dataset (Lookahead 4)
        y_all = []
        for j in range(total_len):
            if j + 4 < total_len:
                y_all.append(1 if closes[j+4] > closes[j] else -1)
            else:
                y_all.append(0)
        y_all = np.array(y_all)

        shading = []
        prev_trend = 0
        
        # SKEPTIC: To avoid O(N^2) complexity while maintaining 'rolling' behavior,
        # we'll use a sliding window of 500 bars for training.
        # For performance, we'll re-fit every 20 bars and batch predict.
        step = 20
        for i in range(start_idx, total_len, step):
            end_batch = min(i + step, total_len)
            
            # Training pool: up to 1000 bars before current batch
            train_end = i - 1
            train_start = max(0, train_end - 1000)
            
            if (train_end - train_start) < k:
                # Not enough training data yet, use what we have or skip
                for j in range(i, end_batch):
                    shading.append({
                        "time": int(indices[j].timestamp()), 
                        "value": round(float(closes[j]), 2),
                        "trend": 0, "marker": 0
                    })
                continue

            X_train = scaled_vals[train_start:train_end]
            y_train = y_all[train_start:train_end]
            
            model = KNeighborsClassifier(n_neighbors=k, weights='distance')
            model.fit(X_train, y_train)
            
            X_query = scaled_vals[i:end_batch]
            preds = model.predict(X_query)
            
            for j, pred in enumerate(preds):
                curr_idx = i + j
                # Baseline: 14-period EMA for visual reference
                b_slice = closes[max(0, curr_idx-14):curr_idx+1]
                baseline = np.mean(b_slice)
                
                # Marker logic: only on trend change (Transitions)
                # SKEPTIC: Ensure we don't spam markers. 
                # If prev_trend was 0, first non-zero is a marker.
                marker = 0
                if pred != prev_trend:
                    marker = int(pred)
                
                prev_trend = pred
                
                shading.append({
                    "time": int(indices[curr_idx].timestamp()), 
                    "value": round(float(baseline), 2),
                    "trend": int(pred),
                    "marker": marker
                })
            
        return shading

classifier = LorentzianClassifier()
adaptive_st = AdaptiveSuperTrend()
knn_forecaster = KNNTrendForecaster()
