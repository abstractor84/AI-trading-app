import asyncio
from services.state import AppState
from services.news_sentiment import NewsSentimentService
from services.technical_analysis import TechnicalAnalysisService

async def main():
    state = AppState()
    # Test DDGS
    state.search_engine = "ddgs"
    news_svc = NewsSentimentService()
    headlines = news_svc.fetch_news("RELIANCE", search_engine=state.search_engine)
    print("DDGS App Headlines:", headlines)

    # Test Upstox
    ta_svc = TechnicalAnalysisService()
    state.data_provider = "upstox"
    # Wait, `fetch_ohlcv` relies on data_provider? Let's check!
    
if __name__ == "__main__":
    asyncio.run(main())
