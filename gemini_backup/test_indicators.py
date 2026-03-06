import pandas as pd
from services.technical_analysis import TechnicalAnalysisService
import json

svc = TechnicalAnalysisService()
# Test ONGC or something
for ticker in ["RELIANCE.NS", "SBI.NS", "IRCTC.NS"]:
    df = svc.fetch_ohlcv(ticker, interval="5m")
    if df is not None and not df.empty:
        indicators = svc.compute_indicators(df)
        print(ticker)
        print("Close:", indicators.get('close'), " VWAP:", indicators.get('vwap'))
        print("EMA9:", indicators.get('ema_9'), " EMA21:", indicators.get('ema_21'))
        print("RSI:", indicators.get('rsi_14'), " ADX:", indicators.get('adx_14'), " VolSurge:", indicators.get('vol_surge'))
