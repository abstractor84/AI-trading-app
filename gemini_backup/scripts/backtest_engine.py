import os
import sys
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to sys.path to import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.technical_analysis import TechnicalAnalysisService
from services.advanced_indicators import classifier
from services.risk_engine import RiskEngine

ta_svc = TechnicalAnalysisService()
risk_engine = RiskEngine(capital=100000)

NIFTY_100 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", 
    "SBIN.NS", "LICI.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS", "HCLTECH.NS", "SUNPHARMA.NS",
    "BAJFINANCE.NS", "MARUTI.NS", "ADANIENT.NS", "TATAMOTORS.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "NTPC.NS", "TITAN.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "COALINDIA.NS",
    "POWERGRID.NS", "M&M.NS", "ADANIPOWER.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "SIEMENS.NS"
]

def run_backtest(ticker, iterations=5):
    print(f"\n--- Backtesting {ticker} ---")
    results = []
    
    for i in range(iterations):
        # Choose a random offset for history (up to 30 days back)
        offset = random.randint(0, 100) 
        df = ta_svc.fetch_ohlcv(ticker, period="30d", interval="5m")
        
        if df.empty or len(df) < 500:
            print(f"Skipping {ticker} iteration {i}: Insufficient data.")
            continue
            
        # Select a random window of 300 bars from the history
        start = random.randint(0, len(df) - 301)
        window_df = df.iloc[start : start + 300].copy()
        
        # Current bar is the last bar in window
        # We'll see what the classifier predicts and then check next 5-10 bars for outcome
        res = classifier.classify(window_df)
        signal = res['signal']
        score = res['score']
        
        if signal != "NEUTRAL":
            entry_price = window_df['Close'].iloc[-1]
            # Use Risk Engine for SL/TP
            # For backtest, we need ATR
            df_for_atr = window_df.tail(100)
            atr = ta_svc.compute_indicators(df_for_atr).get('atr', entry_price * 0.01) # fallback
            
            levels = risk_engine.compute_sl_target(entry_price, "BUY" if signal=="BUY" else "SHORT SELL", atr)
            sl = levels['stop_loss']
            tp = levels['target_1']
            
            # Check upcoming bars in original df
            hit_tp = False
            hit_sl = False
            duration = 0
            
            for j in range(start + 300, min(start + 350, len(df))):
                future_price = df['Close'].iloc[j]
                duration += 1
                if signal == "BUY":
                    if future_price >= tp:
                        hit_tp = True
                        break
                    if future_price <= sl:
                        hit_sl = True
                        break
                else: # SELL
                    if future_price <= tp:
                        hit_tp = True
                        break
                    if future_price >= sl:
                        hit_sl = True
                        break
            
            outcome = "EXPIRED"
            if hit_tp: outcome = "PROFIT"
            elif hit_sl: outcome = "LOSS"
            
            results.append({
                "iteration": i,
                "signal": signal,
                "score": score,
                "outcome": outcome,
                "duration": duration
            })
            print(f"Itr {i}: Signal {signal} (Score {score}) | Outcome: {outcome} in {duration} bars")

    return results

if __name__ == "__main__":
    tickers = random.sample(NIFTY_100, 5)
    all_results = []
    
    for t in tickers:
        all_results.extend(run_backtest(t))
        
    if all_results:
        summary = pd.DataFrame(all_results)
        win_rate = (len(summary[summary['outcome'] == 'PROFIT']) / len(summary[summary['outcome'] != 'EXPIRED'])) * 100 if len(summary[summary['outcome'] != 'EXPIRED']) > 0 else 0
        print("\n--- FINAL SUMMARY ---")
        print(f"Total Signals: {len(summary)}")
        print(f"Wins: {len(summary[summary['outcome'] == 'PROFIT'])}")
        print(f"Losses: {len(summary[summary['outcome'] == 'LOSS'])}")
        print(f"Win Rate (Excl. Expired): {win_rate:.1f}%")
