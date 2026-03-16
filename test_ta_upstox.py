import os
from services.technical_analysis import TechnicalAnalysisService
import logging
logging.basicConfig(level=logging.DEBUG)

ta_svc = TechnicalAnalysisService()
# Mock SIMULATION=false so it actually fetches
os.environ["SIMULATION"] = "false"
df = ta_svc.fetch_ohlcv("RELIANCE.NS", period="5d", interval="5m", data_provider="upstox")
if df is not None:
    print(f"Data shape: {df.shape}")
else:
    print("df is None")
