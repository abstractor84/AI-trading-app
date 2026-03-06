
import time
import pandas as pd
import numpy as np
import logging
from services.technical_analysis import TechnicalAnalysisService
from services.advanced_indicators import classifier, adaptive_st, knn_forecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

def benchmark_ml():
    ta_svc = TechnicalAnalysisService()
    ticker = "RELIANCE.NS"
    
    print(f"--- BENCHMARKING ML INDICATORS FOR {ticker} ---")
    
    # 1. Fetch deep spliced data (same as chart)
    start_fetch = time.time()
    df = ta_svc.fetch_spliced_ohlcv(ticker, "5m")
    fetch_dur = time.time() - start_fetch
    print(f"Data Fetch ({len(df)} bars): {fetch_dur:.4f}s")
    
    if df.empty:
        print("Error: No data for benchmark.")
        return

    # 2. Benchmark Lorentzian
    start = time.time()
    lz_res = classifier.classify_series(df, window=200)
    dur = time.time() - start
    print(f"Lorentzian Classifier (200 bar window): {dur:.4f}s")

    # 3. Benchmark Adaptive SuperTrend
    start = time.time()
    st_res = adaptive_st.calculate(df)
    dur = time.time() - start
    print(f"Adaptive SuperTrend (K-Means Clustering): {dur:.4f}s")

    # 4. Benchmark KNN Strategy
    start = time.time()
    knn_res = knn_forecaster.get_historical_shading(df, window=200)
    dur = time.time() - start
    print(f"KNN Strategy (Pattern Shading): {dur:.4f}s")

    total_ml = (time.time() - start_fetch) - fetch_dur
    print(f"--- TOTAL ML PROCESSING TIME: {total_ml:.4f}s ---")

if __name__ == "__main__":
    benchmark_ml()
