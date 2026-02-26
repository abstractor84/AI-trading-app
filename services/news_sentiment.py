import requests
from bs4 import BeautifulSoup
import logging
import re
from google import genai
import os
import json
from services.quota_service import QuotaService
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Singleton QuotaService
quota_svc = QuotaService()



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
    total = pos + neg
    if total == 0:
        return {"score": 50, "label": "Neutral", "sentiment": "NEUTRAL",
                "reason": "No sentiment keywords found"}
    score = int((pos / total) * 100)  # 0=bearish, 100=bullish
    if score >= 60:
        label, sentiment = "Bullish", "POSITIVE"
    elif score <= 40:
        label, sentiment = "Bearish", "NEGATIVE"
    else:
        label, sentiment = "Neutral", "NEUTRAL"
    return {"score": score, "label": label, "sentiment": sentiment,
            "reason": f"Keyword scan: {pos} positive, {neg} negative signals"}


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
        self.google_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.samba_key = os.getenv("SAMBA_API_KEY")
        self.tavily_key = os.getenv("TAVILY_API_KEY")

        if self.google_key:
            self.google_client = genai.Client(api_key=self.google_key, http_options={'headers': {'X-Goog-Api-Client': 'genai-local'}}) 
        else:
            self.google_client = None

        if not self.tavily_key:
            logger.warning("TAVILY_API_KEY not found in .env. Tavily search will be skipped.")

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

    def score_sentiment(self, headlines: list[str], provider: str = "google", model_name: str = "gemini-2.5-flash") -> dict:
        """Score news sentiment using the user's selected AI provider."""
        if not headlines:
            return {"sentiment": "NEUTRAL", "reason": "No news found"}

        prompt = f"""
        Analyze the following recent news headlines for an Indian stock and determine the overall sentiment.
        Headlines: {headlines}
        
        Respond ONLY with a JSON object in this exact format:
        {{"sentiment": "POSITIVE|NEGATIVE|NEUTRAL", "reason": "Short 1-sentence explanation"}}
        """

        quota = quota_svc.check_quota(model_name)
        if not quota["can_call"]:
            logger.warning(f"Quota exceeded for {model_name}. Falling back to keyword analysis.")
            return _keyword_sentiment(headlines)

        try:
            if provider == "google":
                return self._call_google(model_name, prompt)
            elif provider == "groq":
                return self._call_groq(model_name, prompt)
            elif provider == "sambanova":
                return self._call_sambanova(model_name, prompt)
            else:
                logger.warning(f"Unknown sentiment provider {provider}. Using keyword fallback.")
                return _keyword_sentiment(headlines)
        except Exception as e:
            logger.error(f"Sentiment scoring failed via {provider}: {e}", exc_info=True)
            return _keyword_sentiment(headlines)

    def _call_google(self, model_name: str, prompt: str) -> dict:
        if not self.google_client:
            raise ValueError("GEMINI_API_KEY missing.")
        
        response = self.google_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(response_mime_type="application/json")
        )
        tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
        quota_svc.log_usage(model_name, tokens=tokens)
        return json.loads(response.text)

    def _call_groq(self, model_name: str, prompt: str) -> dict:
        if not self.groq_key:
            raise ValueError("GROQ_API_KEY missing.")
        
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=15)
        res.raise_for_status()
        jdoc = res.json()
        tokens = jdoc.get("usage", {}).get("total_tokens", 0)
        quota_svc.log_usage(model_name, tokens=tokens)
        
        content = jdoc["choices"][0]["message"]["content"]
        # Basic JSON clean if enclosed in markdown
        if content.startswith("```json"):
            content = content.replace("```json\n", "").replace("```", "")
        return json.loads(content)

    def _call_sambanova(self, model_name: str, prompt: str) -> dict:
        if not self.samba_key:
            raise ValueError("SAMBA_API_KEY missing.")
            
        headers = {
            "Authorization": f"Bearer {self.samba_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        res = requests.post("https://api.sambanova.ai/v1/chat/completions", headers=headers, json=payload, timeout=15)
        res.raise_for_status()
        jdoc = res.json()
        tokens = jdoc.get("usage", {}).get("total_tokens", 0)
        quota_svc.log_usage(model_name, tokens=tokens)
        
        content = jdoc["choices"][0]["message"]["content"]
        if content.startswith("```json"):
            content = content.replace("```json\n", "").replace("```", "")
        return json.loads(content)

