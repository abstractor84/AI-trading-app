import os
import sys
import logging
# Add current directory to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
from services.technical_analysis import TechnicalAnalysisService

ta_svc = TechnicalAnalysisService()

print("Verifying Upstox V3 data fetch via TA Service...")
# Fetch 5m data (which should pull 1m V3 and resample)
df = ta_svc.fetch_ohlcv("RELIANCE.NS", period="5d", interval="5m")

if df is not None and not df.empty:
    print(f"Success! Fetched {len(df)} rows for RELIANCE.NS")
    print("Columns:", df.columns.tolist())
    print("Last Row:\n", df.tail(1))
    
    # Check if it was actually from Upstox (logs would have shown fallback otherwise)
    # We can check the line count or specific log output if we captured it.
else:
    print("Failed to fetch data via TA Service.")
