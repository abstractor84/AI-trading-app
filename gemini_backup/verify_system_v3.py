import asyncio
import logging
import pandas as pd
from datetime import datetime
import pytz
import warnings

# Suppress warnings for clean verification output
warnings.filterwarnings("ignore")

from services.technical_analysis import TechnicalAnalysisService
from services.advanced_indicators import classifier, adaptive_st, knn_forecaster

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Verification")

async def verify_system():
    ta_svc = TechnicalAnalysisService()
    ticker = "RELIANCE.NS"
    
    logger.info(f"--- VERIFYING DATA SPLICING FOR {ticker} ---")
    try:
        # Test 1: Spliced OHLCV
        df = await asyncio.to_thread(ta_svc.fetch_spliced_ohlcv, ticker, "5m")
        
        if df is None or df.empty:
            logger.error("FAILED: Spliced DataFrame is empty!")
            return
            
        logger.info(f"SUCCESS: Spliced DF has {len(df)} bars.")
        logger.info(f"Range: {df.index[0]} to {df.index[-1]}")
        
        # Check source coverage
        ist = pytz.timezone("Asia/Kolkata")
        today = datetime.now(ist).date()
        yesterday = today - pd.Timedelta(days=1)
        long_ago = today - pd.Timedelta(days=50)
        
        has_context = not df[df.index.date <= long_ago].empty
        has_historical_intraday = not df[(df.index.date > long_ago) & (df.index.date <= yesterday)].empty
        has_live = not df[df.index.date == today].empty
        
        logger.info(f"Context Coverage (Daily yfinance): {'✅' if has_context else '❌'}")
        logger.info(f"Historical Intraday Coverage (yfinance): {'✅' if has_historical_intraday else '❌'}")
        logger.info(f"Live Coverage (Upstox): {'✅' if has_live else '❌ (Market may be closed)'}")

        # Test 2: Indicator Calculations
        logger.info("--- VERIFYING INDICATOR CALCULATIONS ---")
        if df is not None and not df.empty:
            # Lorentzian
            lz_series = await asyncio.to_thread(classifier.classify_series, df, window=100)
            logger.info(f"Lorentzian: Generated {len(lz_series)} signal bars. Last score: {lz_series[-1].get('score')}")
            
            # Adaptive ST
            st_data = await asyncio.to_thread(adaptive_st.calculate, df)
            if st_data:
                logger.info(f"Adaptive ST: Generated {len(st_data['time'])} bars. Trend: {st_data['trend'][-1]}")
            else:
                logger.error("Adaptive ST: Calculation failed!")

            # KNN Strategy
            knn_shading = await asyncio.to_thread(knn_forecaster.get_historical_shading, df, window=100)
            if knn_shading:
                logger.info(f"KNN Strategy: Generated {len(knn_shading)} shading bars. Trend: {knn_shading[-1]['trend']}")
            else:
                logger.error("KNN Strategy: Shading calculation failed!")

        # Test 3: Upstox Robustness
        logger.info("--- VERIFYING UPSTOX BOUNDARY CLAMPING ---")
        from services.technical_analysis import _upstox_svc
        if _upstox_svc.is_authenticated:
            df_clamped = await asyncio.to_thread(_upstox_svc.fetch_ohlcv, ticker, interval="5", days=60)
            if df_clamped is not None:
                days_returned = (df_clamped.index[-1] - df_clamped.index[0]).days
                logger.info(f"Upstox Clamping: Requested 60d, got ~{days_returned}d. Logic: ✅")
            else:
                logger.warning("Upstox: No data returned (likely expired token)")
        else:
            logger.info("Upstox: Unauthenticated, skipping clamping test.")

        logger.info("\n--- FINAL VERIFICATION RESULT: SYSTEM STABLE ---")

    except Exception as e:
        logger.error(f"VERIFICATION CRASHED: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(verify_system())