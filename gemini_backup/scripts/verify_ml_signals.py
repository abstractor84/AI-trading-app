import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Ensure project root is in path
sys.path.append(os.getcwd())

from services.technical_analysis import ta_svc
from services.advanced_indicators import LorentzianClassifier, KNNTrendForecaster, AdaptiveSuperTrend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML_Audit")

def run_ml_audit(tickers=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ADANIPORTS.NS", "JSWSTEEL.NS"]):
    report = []
    
    for ticker in tickers:
        logger.info(f"Auditing ML Signals for {ticker}...")
        try:
            # 1. Fetch Spliced Data (180d Daily + 30d Intraday)
            df = ta_svc.fetch_spliced_ohlcv(ticker)
            if df.empty or len(df) < 800:
                logger.warning(f"Insufficient data for {ticker}")
                continue
                
            # 2. Run Lorentzian Classification over the last 800 bars
            # Using smaller lookback (500) to find more local repeating patterns
            classifier = LorentzianClassifier(lookback=500, k=4)
            window = 800
            signals = classifier.classify_series(df, window=window, signal_threshold=0.2)
            
            # 3. Evaluate Lorentzian Accuracy
            l_hits = 0
            l_total = 0
            
            for s in signals:
                sig_val = s['signal']
                if sig_val == 0: continue
                
                # Check 5 bars forward return
                dt = pd.to_datetime(s['time'])
                try:
                    current_idx = df.index.get_loc(dt)
                    if current_idx + 5 < len(df):
                        entry_price = df['Close'].iloc[current_idx]
                        exit_price = df['Close'].iloc[current_idx + 5]
                        
                        fwd_return = (exit_price - entry_price) / entry_price
                        
                        is_correct = (sig_val == 1 and fwd_return > 0) or (sig_val == -1 and fwd_return < 0)
                        if is_correct: l_hits += 1
                        l_total += 1
                except:
                    continue
            
            # Alternative: If 0.6 is too high, also check 0.4 in report
            # (Just for the audit visibility)
            
            # 4. Run KNN Forecast for the current moment
            knn = KNNTrendForecaster()
            forecast = knn.forecast(df)
            
            current_price = df['Close'].iloc[-1]
            knn_bias = "Bullish" if forecast and forecast[-1] > current_price else "Bearish" if forecast else "Neutral"
            
            # 5. Compile Result
            l_precision = (l_hits / l_total * 100) if l_total > 0 else 0
            
            report.append({
                "Ticker": ticker,
                "Lorentzian_Signals": l_total,
                "Lorentzian_Precision_5b": f"{l_precision:.1f}%",
                "KNN_Bias": knn_bias,
                "Data_Bars": len(df)
            })
            
        except Exception as e:
            logger.error(f"Audit failed for {ticker}: {e}")
            
    return report

if __name__ == "__main__":
    results = run_ml_audit()
    print("\n" + "="*60)
    print("           ML SIGNAL QUALITY AUDIT REPORT")
    print("="*60)
    print(f"{'Ticker':<12} | {'Sigs':<4} | {'Precision':<10} | {'KNN Bias':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['Ticker']:<12} | {r['Lorentzian_Signals']:<4} | {r['Lorentzian_Precision_5b']:<10} | {r['KNN_Bias']:<10}")
    print("="*60)
    print(f"Generated at: {datetime.now()}")
