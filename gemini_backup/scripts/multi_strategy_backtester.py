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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.technical_analysis import TechnicalAnalysisService
ta_svc = TechnicalAnalysisService()
from services.advanced_indicators import (
    LorentzianClassifier, 
    KNNTrendForecaster, 
    AdaptiveSuperTrend,
    VWAPTacticalReversion,
    MomentumBreakout
)

# Constants
RESULTS_DIR = "backtest_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "backtest_log.log")

# Setup Logging
logger = logging.getLogger("MultiStrategyBacktester")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)

# Global lock for CSV writing
csv_lock = threading.Lock()

# Optimized Event Windows & Tiered Fidelity
TIERED_CONFIG = [
    {"name": "Macro_Trend_180D", "start": "2025-09-01", "end": "2026-03-01", "interval": "1h", "days_prior": 450, "tickers_count": 10},
    {"name": "Sniper_Audit_7D", "start": "2026-02-23", "end": "2026-03-01", "interval": "1m", "days_prior": 7, "tickers_count": 10}
]

# Specific Windows for deeper analysis
MACRO_WINDOWS = [
    {"name": "RBI_Rate_Cut", "start": "2025-12-05", "end": "2025-12-10", "interval": "1h", "days_prior": 400},
    {"name": "Union_Budget_2026", "start": "2026-01-30", "end": "2026-02-04", "interval": "1h", "days_prior": 400}
]

def fetch_nifty500():
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        df = pd.read_csv(url)
        if 'Symbol' in df.columns:
            return [f"{s}.NS" for s in df['Symbol'].tolist()]
    except Exception as e:
        logger.error(f"Failed to fetch NIFTY 500: {e}")
    return []

def run_backtest_on_window(ticker, window):
    window_name = window['name']
    start_date = window['start']
    end_date = window['end']
    interval = window['interval']
    days_prior = window['days_prior']
    
    csv_filename = os.path.join(RESULTS_DIR, f"{window_name}_results.csv")
    
    try:
        # Fetch data with sufficient context for lookback
        start_context = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=days_prior)).strftime("%Y-%m-%d")
        df = ta_svc.fetch_historical_range(ticker, start_date=start_context, end_date=end_date, interval=interval)
        
        if df.empty:
            return None
            
        target_df = df[(df.index >= start_date) & (df.index <= end_date)]
        if target_df.empty:
            return None

        # Initialize Strategies
        l_classifier = LorentzianClassifier(lookback=2000, k=12)
        k_forecaster = KNNTrendForecaster(k=5, sequence_length=15, forecast_horizon=5)
        vwap_strat = VWAPTacticalReversion(stdev_mult=2.0)
        mom_strat = MomentumBreakout(adx_threshold=20, vol_mult=1.3)
        
        # Pre-calculate signals for efficiency (Selective for Sniper)
        threshold = 0.9 if interval == "1m" else 0.75
        l_signals = l_classifier.classify_series(df, window=len(target_df), signal_threshold=threshold)
        vwap_bands = vwap_strat.calculate(df)
        
        # Performance Tracking
        stats = {
            "Ticker": ticker,
            "Window": window_name,
            "L_Sniper_Signals": 0, "L_Sniper_Hits": 0,
            "KNN_Forecast_Error": [],
            "VWAP_Reversion_Signals": 0, "VWAP_Reversion_Hits": 0,
            "Momentum_Breakouts": 0, "Momentum_Hits": 0
        }
        
        for i in range(len(target_df)):
            dt = target_df.index[i]
            idx_in_full = df.index.get_loc(dt)
            
            # --- Lorentzian Sniper ---
            sig_dict = next((s for s in l_signals if s['time'] == dt.isoformat()), None)
            if sig_dict and sig_dict['signal'] != 0:
                if idx_in_full + 5 < len(df):
                    ret = (df['Close'].iloc[idx_in_full + 5] - df['Close'].iloc[idx_in_full]) / df['Close'].iloc[idx_in_full]
                    if (sig_dict['signal'] == 1 and ret > 0) or (sig_dict['signal'] == -1 and ret < 0):
                        stats["L_Sniper_Hits"] += 1
                    stats["L_Sniper_Signals"] += 1
            
            # --- KNN Trend Follower (Sampling) ---
            if i % 30 == 0:
                sub_df = df[df.index <= dt].tail(500)
                forecast = k_forecaster.forecast(sub_df)
                if forecast is not None and idx_in_full + 5 < len(df):
                    actual = df['Close'].iloc[idx_in_full + 1 : idx_in_full + 6].values
                    if len(actual) == 5:
                        try:
                            error = np.sqrt(np.mean((np.array(forecast) - actual)**2))
                            stats["KNN_Forecast_Error"].append(error)
                        except Exception as e:
                            logger.debug(f"KNN Eval error for {ticker}: {e}")
            
            # --- VWAP Tactical Reversion ---
            if vwap_bands is not None and dt in vwap_bands.index:
                close = target_df['Close'].iloc[i]
                upper = vwap_bands.loc[dt, 'upper']
                lower = vwap_bands.loc[dt, 'lower']
                
                sig = 0
                if close > upper: sig = -1 # Sell (Overbought)
                elif close < lower: sig = 1 # Buy (Oversold)
                
                if sig != 0 and idx_in_full + 5 < len(df):
                    ret = (df['Close'].iloc[idx_in_full + 5] - df['Close'].iloc[idx_in_full]) / df['Close'].iloc[idx_in_full]
                    if (sig == 1 and ret > 0) or (sig == -1 and ret < 0):
                        stats["VWAP_Reversion_Hits"] += 1
                    stats["VWAP_Reversion_Signals"] += 1

            # --- Momentum Breakout ---
            sub_df_mom = df[df.index <= dt].tail(30)
            res_mom = mom_strat.evaluate(sub_df_mom)
            if res_mom != "NEUTRAL":
                if idx_in_full + 5 < len(df):
                    ret = (df['Close'].iloc[idx_in_full + 5] - df['Close'].iloc[idx_in_full]) / df['Close'].iloc[idx_in_full]
                    if (res_mom == "BULLISH_BREAKOUT" and ret > 0) or (res_mom == "BEARISH_BREAKOUT" and ret < 0):
                        stats["Momentum_Hits"] += 1
                    stats["Momentum_Breakouts"] += 1

        # Flatten Stats
        final_res = {
            "Ticker": ticker,
            "L_Precision": (stats["L_Sniper_Hits"] / stats["L_Sniper_Signals"] * 100) if stats["L_Sniper_Signals"] > 0 else 0,
            "L_Signals": stats["L_Sniper_Signals"],
            "KNN_RMSE": np.mean(stats["KNN_Forecast_Error"]) if stats["KNN_Forecast_Error"] else 0,
            "VWAP_Precision": (stats["VWAP_Reversion_Hits"] / stats["VWAP_Reversion_Signals"] * 100) if stats["VWAP_Reversion_Signals"] > 0 else 0,
            "VWAP_Signals": stats["VWAP_Reversion_Signals"],
            "MOM_Precision": (stats["Momentum_Hits"] / stats["Momentum_Breakouts"] * 100) if stats["Momentum_Breakouts"] > 0 else 0,
            "MOM_Signals": stats["Momentum_Breakouts"]
        }
        
        with csv_lock:
            file_exists = os.path.isfile(csv_filename)
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=final_res.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(final_res)
        
        return final_res

    except Exception as e:
        logger.error(f"Error backtesting {ticker} on {window_name}: {e}")
        return None

def get_processed_tickers(window_name):
    csv_filename = os.path.join(RESULTS_DIR, f"{window_name}_results.csv")
    if os.path.exists(csv_filename):
        try:
            return pd.read_csv(csv_filename)['Ticker'].tolist()
        except: return []
    return []

def main():
    logger.info("Initializing 180-Day Tiered Fidelity Backtester...")
    all_tickers = fetch_nifty500()
    if not all_tickers:
        return
        
    import random
    random.shuffle(all_tickers)
    
    # PHASE 1: Macro Trend Audit (180 Days, 1h Resolution, 10 Stocks)
    macro_conf = TIERED_CONFIG[0]
    processed_macro = get_processed_tickers(macro_conf['name'])
    tickers_macro = [t for t in all_tickers[:macro_conf['tickers_count']] if t not in processed_macro]
    
    logger.info(f"PHASE 1: Starting Macro Audit ({macro_conf['name']}) - {len(tickers_macro)} remaining")
    if tickers_macro:
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(run_backtest_on_window, t, macro_conf): t for t in tickers_macro}
            for i, _ in enumerate(as_completed(futures)):
                if (i+1) % 10 == 0:
                    logger.info(f"Macro Audit Progress: {i+1}/{len(tickers_macro)}")

    # PHASE 2: Tactical Sniper Audit (7 Days, 1m Resolution, 100 Stocks)
    sniper_conf = TIERED_CONFIG[1]
    processed_sniper = get_processed_tickers(sniper_conf['name'])
    tickers_sniper = [t for t in all_tickers[:sniper_conf['tickers_count']] if t not in processed_sniper]
    
    logger.info(f"PHASE 2: Starting Tactical Sniper Audit ({sniper_conf['name']}) - {len(tickers_sniper)} remaining")
    if tickers_sniper:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(run_backtest_on_window, t, sniper_conf): t for t in tickers_sniper}
            for i, _ in enumerate(as_completed(futures)):
                if (i+1) % 10 == 0:
                    logger.info(f"Sniper Audit Progress: {i+1}/{len(tickers_sniper)}")

    # PHASE 3: Event Stress Tests
    for window in MACRO_WINDOWS:
        logger.info(f"PHASE 3: Stress Testing Window: {window['name']}")
        with ThreadPoolExecutor(max_workers=10) as executor:
            top_tickers = all_tickers[:50]
            futures = {executor.submit(run_backtest_on_window, t, window): t for t in top_tickers}
            for _ in as_completed(futures): pass

    logger.info("="*50)
    logger.info("STRATEGIC TIERED BACKTEST COMPLETE")
    logger.info(f"Results saved in: {RESULTS_DIR}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
