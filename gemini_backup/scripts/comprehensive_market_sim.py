import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import threading

# Ensure project root is in path
sys.path.append(os.getcwd())

from services.technical_analysis import ta_svc
from services.advanced_indicators import LorentzianClassifier, KNNTrendForecaster, AdaptiveSuperTrend
from services.stock_discovery import StockDiscoveryService

# Constants
CSV_PATH = "scripts/audit_results_nse500.csv"
LOG_PATH = "scripts/sim_progress.log"

# Setup Logging
logger = logging.getLogger("NSE500_Sim")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)
sh = logging.StreamHandler() # Also keep console for command_status
sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)

# Global lock for CSV writing
csv_lock = threading.Lock()

def fetch_nifty500():
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        df = pd.read_csv(url)
        if 'Symbol' in df.columns:
            return [f"{s}.NS" for s in df['Symbol'].tolist()]
    except Exception as e:
        logger.error(f"Failed to fetch NIFTY 500: {e}")
    return []

def audit_ticker(ticker, target_date_str="2026-02-27"):
    try:
        # Fetch 5 days of 1-minute data for context
        df = ta_svc.fetch_ohlcv(ticker, period="5d", interval="1m")
        if df.empty or len(df) < 500:
            return None

        target_date = pd.to_datetime(target_date_str).date()
        df_target = df[df.index.date == target_date].copy()
        if len(df_target) < 100:
            return None

        # ML Components (Small lookback for speed/sensitivity on 1m chart)
        l_classifier = LorentzianClassifier(lookback=500, k=4)
        k_forecaster = KNNTrendForecaster(k=3, sequence_length=15, forecast_horizon=5)
        
        # Walk-forward Lorentzian logic
        l_signals = l_classifier.classify_series(df, window=len(df_target), signal_threshold=0.8)
        
        l_hits, l_total = 0, 0
        knn_errors = []
        
        for s in l_signals:
            dt = pd.to_datetime(s['time'])
            if dt.date() != target_date: continue
            sig_val = s['signal']
            if sig_val != 0:
                try:
                    idx = df.index.get_loc(dt)
                    if idx + 5 < len(df):
                        entry_p = df['Close'].iloc[idx]
                        exit_p = df['Close'].iloc[idx + 5]
                        ret = (exit_p - entry_p) / entry_p
                        if (sig_val == 1 and ret > 0) or (sig_val == -1 and ret < 0): l_hits += 1
                        l_total += 1
                except: continue

        # Sample KNN Forecasts at 30min intervals
        for i in range(0, len(df_target) - 5, 30):
            dt = df_target.index[i]
            sub_df = df[df.index <= dt].tail(500)
            forecast = k_forecaster.forecast(sub_df)
            if forecast:
                actual_path = df['Close'].iloc[df.index.get_loc(dt) + 1 : df.index.get_loc(dt) + 6].values
                if len(actual_path) == 5:
                    rmse = np.sqrt(np.mean((forecast - actual_path)**2))
                    knn_errors.append(rmse)

        res = {
            "Ticker": ticker,
            "L_Signals": l_total,
            "L_Precision": (l_hits / l_total * 100) if l_total > 0 else 0,
            "KNN_RMSE_Avg": round(np.mean(knn_errors), 4) if knn_errors else 0,
            "Volatility": round(df_target['Close'].pct_change().std() * 100, 4)
        }
        
        # Incremental CSV write
        with csv_lock:
            file_exists = os.path.isfile(CSV_PATH)
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=res.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(res)
        
        return res

    except Exception as e:
        logger.error(f"Error auditing {ticker}: {e}")
        return None

def main():
    logger.info("Initializing NSE-500 Simulation...")
    # Prime Upstox
    ta_svc.fetch_ohlcv("RELIANCE.NS", period="1d", interval="1m")
    
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AXISBANK.NS"]
    if not tickers: return

    # Clean existing CSV
    if os.path.exists(CSV_PATH): os.remove(CSV_PATH)
    
    logger.info(f"Auditing {len(tickers)} tickers. Following progress in: {LOG_PATH}")
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(audit_ticker, t): t for t in tickers}
        count = 0
        for future in as_completed(futures):
            count += 1
            if count % 25 == 0:
                logger.info(f"Progress Update: {count}/{len(tickers)} (Last {futures[future]} completed)")

    # Final summary from CSV
    if os.path.exists(CSV_PATH):
        rdf = pd.read_csv(CSV_PATH)
        avg_p = rdf[rdf['L_Signals'] > 0]['L_Precision'].mean()
        logger.info("="*50)
        logger.info("SIMULATION COMPLETE")
        logger.info(f"Total Tickers Audited: {len(rdf)}")
        logger.info(f"Mean Signal Precision: {avg_p:.2f}%")
        logger.info(f"Results File: {CSV_PATH}")
        logger.info("="*50)

if __name__ == "__main__":
    main()
