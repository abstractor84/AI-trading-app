import asyncio
import pandas as pd
import numpy as np
from services.technical_analysis import TechnicalAnalysisService
import os

async def check_knn():
    ta_svc = TechnicalAnalysisService()
    os.environ["SIMULATION"] = "true"
    
    ticker = "RELIANCE.NS"
    interval = "5m"
    params = {
        "knn": {"k": 5, "sequence_length": 15, "window": 200}
    }
    
    payload = await asyncio.to_thread(ta_svc.get_chart_payload, ticker, interval, params)
    
    if "error" in payload:
        print(f"Error: {payload['error']}")
        return

    knn = payload.get("ml_knn", [])
    print(f"KNN data points: {len(knn)}")
    if knn:
        markers = [p for p in knn if p.get("marker") != 0]
        print(f"KNN markers: {len(markers)}")
        if markers:
            print(f"First marker: {markers[0]}")
        else:
            # Check trends
            trends = [p.get("trend") for p in knn]
            print(f"Unique trends: {set(trends)}")

if __name__ == "__main__":
    asyncio.run(check_knn())
