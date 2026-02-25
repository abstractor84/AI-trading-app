import requests
from bs4 import BeautifulSoup
import logging
import re
from google import genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Simple keyword lists for local sentiment scoring (used when Gemini is unavailable)
_POSITIVE_WORDS = {
    'surge', 'jump', 'rally', 'gain', 'rise', 'up', 'record', 'high', 'profit',
    'growth', 'beat', 'strong', 'bullish', 'upgraded', 'outperform', 'buy', 'positive',
    'good', 'excellent', 'awarded', 'wins', 'launch', 'expands', 'deal', 'partnership'
}
_NEGATIVE_WORDS = {
    'fall', 'drop', 'crash', 'decline', 'down', 'loss', 'weak', 'sell', 'bear',
    'bearish', 'cut', 'downgrade', 'underperform', 'fraud', 'penalty', 'ban',
    'concern', 'risk', 'miss', 'slump', 'plunge', 'warning', 'probe', 'fine'
}

def _keyword_sentiment(headlines: list[str]) -> dict:
    """Fast local keyword-based sentiment — used when Gemini is unavailable."""
    pos = neg = 0
    for h in headlines:
        words = set(re.findall(r'\b\w+\b', h.lower()))
        pos += len(words & _POSITIVE_WORDS)
        neg += len(words & _NEGATIVE_WORDS)
    if pos > neg:
        return {"sentiment": "POSITIVE", "reason": f"Keyword scan: {pos}+ positive signals (Gemini unavailable)"}
    elif neg > pos:
        return {"sentiment": "NEGATIVE", "reason": f"Keyword scan: {neg}+ negative signals (Gemini unavailable)"}
    return {"sentiment": "NEUTRAL", "reason": "Keyword scan: balanced or neutral signals (Gemini unavailable)"}


def _rss_fetch(clean_ticker: str) -> list[str]:
    """Fetch from Google News RSS."""
    url = f"https://news.google.com/rss/search?q={clean_ticker}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=6)
        soup = BeautifulSoup(response.content, features="xml")
        return [item.title.text for item in soup.find_all("item")[:5]]
    except Exception as e:
        logger.error(f"RSS fetch failed for {clean_ticker}: {e}")
        return []


def _tavily_fetch(query: str) -> list[str]:
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return []
    try:
        from tavily import TavilyClient
        res = TavilyClient(api_key=tavily_key).search(query=query, search_depth="basic", max_results=5)
        return [r['title'] for r in res.get('results', [])]
    except Exception as e:
        logger.error(f"Tavily fetch failed: {e}")
    return []


def _ddgs_fetch(query: str) -> list[str]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # Use news() — more targeted for financial news than text()
            results = list(ddgs.news(query, max_results=5))
            if results:
                return [r['title'] for r in results]
            # fallback: broader text search without timelimit
            results = list(ddgs.text(query, max_results=5))
            return [r['title'] for r in results]
    except Exception as e:
        logger.error(f"DDGS fetch failed: {e}")
    return []


class NewsSentimentService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("GEMINI_API_KEY not found. Sentiment will use keyword fallback.")

    def fetch_news(self, ticker: str, search_engine: str = "gemini", fallback: bool = False) -> list[str]:
        """
        Fetch news headlines.
        fallback=False (default): fail fast — only use the selected engine.
        fallback=True: cascade through available engines until one succeeds.
        """
        clean_ticker = ticker.replace(".NS", "")
        query = f"{clean_ticker} NSE India stock market news today"

        if search_engine == "tavily":
            headlines = _tavily_fetch(query)
            if headlines or not fallback:
                return headlines
            logger.info(f"Tavily empty for {ticker}. Fallback enabled — trying DDGS.")
            headlines = _ddgs_fetch(query)
            if headlines or not fallback:
                return headlines
            logger.info(f"DDGS empty for {ticker}. Falling back to RSS.")
            return _rss_fetch(clean_ticker)

        elif search_engine == "ddgs":
            headlines = _ddgs_fetch(query)
            if headlines or not fallback:
                return headlines
            logger.info(f"DDGS empty for {ticker}. Fallback enabled — trying RSS.")
            return _rss_fetch(clean_ticker)

        else:  # gemini / default = RSS
            headlines = _rss_fetch(clean_ticker)
            if headlines or not fallback:
                return headlines
            logger.info(f"RSS empty for {ticker}. Fallback enabled — trying DDGS.")
            headlines = _ddgs_fetch(query)
            if headlines or not fallback:
                return headlines
            logger.info(f"DDGS empty for {ticker}. Falling back to Tavily.")
            return _tavily_fetch(query)


    def score_sentiment(self, headlines: list[str]) -> dict:
        """
        Score news sentiment.
        Primary: Gemini 3.0 Flash → Gemini 2.5 Flash
        Fallback: Local keyword scoring (when Gemini hits 429 or is unavailable)
        """
        if not headlines:
            return {"sentiment": "NEUTRAL", "reason": "No news found"}

        if not self.client:
            return _keyword_sentiment(headlines)

        prompt = f"""
        Analyze the following recent news headlines for an Indian stock and determine the overall sentiment.
        Headlines: {headlines}
        
        Respond ONLY with a JSON object in this exact format:
        {{"sentiment": "POSITIVE|NEGATIVE|NEUTRAL", "reason": "Short 1-sentence explanation"}}
        """

        for model_name in ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                err_str = str(e)
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                    logger.warning(f"Gemini rate limit hit on {model_name}, trying next model.")
                elif '404' in err_str or 'NOT_FOUND' in err_str:
                    logger.warning(f"Gemini model {model_name} not found, trying next.")
                else:
                    logger.warning(f"{model_name} failed: {e}")

        # All Gemini models exhausted — fall back to keyword analysis
        logger.warning("All Gemini models failed for sentiment. Using keyword fallback.")
        return _keyword_sentiment(headlines)

