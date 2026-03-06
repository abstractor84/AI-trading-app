import asyncio
import json
from services.technical_analysis import TechnicalAnalysisService
from services.price_projector import price_projector
from services.advanced_indicators import classifier, adaptive_st, knn_forecaster

async def test():
    ta_svc = TechnicalAnalysisService()
    df = ta_svc.fetch_spliced_ohlcv("RELIANCE.NS", "5m")
    
    result = price_projector.generate_projection(df, 1)
    lor_series = classifier.classify_series(df, window=200)
    st_bands = adaptive_st.calculate(df)
    knn_proj = knn_forecaster.forecast(df)
    knn_shading = knn_forecaster.get_historical_shading(df, window=200)
    
    result["ml_lorentzian"] = lor_series
    result["ml_adaptive_st"] = st_bands
    result["ml_knn_shading"] = knn_shading
    if knn_proj is not None:
        result["ml_knn_forecast"] = [round(float(p), 2) for p in knn_proj]
        
    try:
        json.dumps(result)
        print("JSON Serialization SUCCESS")
    except Exception as e:
        print(f"JSON Serialization FAILED: {e}")

asyncio.run(test())
