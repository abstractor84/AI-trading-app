import asyncio
from duckduckgo_search import DDGS

async def main():
    def sync_call():
        with DDGS() as ddgs:
            return list(ddgs.news("RELIANCE", max_results=2))
    
    try:
        print(sync_call())
    except Exception as e:
        print("Crash:", type(e), e)

asyncio.run(main())
