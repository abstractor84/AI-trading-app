import asyncio
from services.news_sentiment import NewsSentimentService
from services.upstox_service import UpstoxService

async def main():
    ns = NewsSentimentService()
    
    print("Testing DDGS...")
    try:
        ddgs_news = ns.fetch_news("RELIANCE", "ddgs")
        print("DDGS News:", ddgs_news)
    except Exception as e:
        print("DDGS Error:", e)
        
    print("\nTesting Tavily...")
    try:
        tav_news = ns.fetch_news("RELIANCE", "tavily")
        print("Tavily News:", tav_news)
    except Exception as e:
        print("Tavily Error:", e)
        
    print("\nTesting Upstox...")
    try:
        us = UpstoxService()
        data = us.fetch_1m_data("RELIANCE.NS", days=1)
        if data is not None:
             print("Upstox Data shape:", data.shape)
             print("Columns:", data.columns)
             print(data.head())
        else:
             print("Upstox Data is None")
    except Exception as e:
        print("Upstox Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
