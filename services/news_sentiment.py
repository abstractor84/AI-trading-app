"""
News Sentiment Service
Fetches and analyzes news for market sentiment. Enhanced with:
- Dynamic time filtering based on open positions
- Prioritized free news sources (Google RSS → DuckDuckGo → Tavily)
- Comprehensive market-focused search keywords
- Caching to reduce API calls
"""
import requests
from bs4 import BeautifulSoup
import logging
import re
import asyncio
import threading
import time
from datetime import datetime, timedelta
from google import genai
import os
import json
from services.quota_service import QuotaService
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
_ddgs_lock = threading.Lock()

# Singleton QuotaService
quota_svc = QuotaService()

# ==============================================================================
# CACHING MECHANISM
# ==============================================================================
MAX_CACHE_SIZE = 100  # Maximum number of cache entries
CLEANUP_THRESHOLD = 80  # Trigger cleanup when cache reaches this size


class NewsCache:
    """Simple in-memory cache for news results to reduce API calls."""
    
    def __init__(self, ttl_seconds: int = 900):  # 15 minutes default TTL
        self._cache = {}
        self._ttl = ttl_seconds
    
    def _maybe_cleanup(self):
        """Remove oldest expired entries when cache exceeds size limit."""
        if len(self._cache) >= MAX_CACHE_SIZE:
            # Remove expired entries first
            current_time = time.time()
            expired_keys = [
                k for k, v in self._cache.items()
                if current_time - v['timestamp'] > self._ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            
            # If still over limit, remove oldest entries
            if len(self._cache) >= CLEANUP_THRESHOLD:
                sorted_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k]['timestamp']
                )
                for key in sorted_keys[:len(self._cache) - MAX_CACHE_SIZE + 10]:
                    del self._cache[key]
    
    def get(self, key: str) -> list | None:
        """Get cached results if not expired."""
        if key not in self._cache:
            return None
        entry = self._cache[key]
        if time.time() - entry['timestamp'] > self._ttl:
            del self._cache[key]
            return None
        logger.debug(f"Cache HIT for key: {key}")
        return entry['data']
    
    def set(self, key: str, data: list):
        """Store results in cache with timestamp."""
        # Trigger cleanup if needed before adding new entry
        self._maybe_cleanup()
        self._cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        logger.debug(f"Cache SET for key: {key}")
    
    def clear(self):
        """Clear all cached data."""
        self._cache.clear()

# Global news cache instance
news_cache = NewsCache(ttl_seconds=900)  # 15 minutes cache

# ==============================================================================
# MARKET-FOCUSED SEARCH KEYWORDS
# ==============================================================================
# Keywords for global/national factors that affect Indian markets
MARKET_KEYWORDS = [
    # War/conflicts, geopolitical
    "war", "conflict", "geopolitical tension", "crude oil", "OPEC",
    "Russia Ukraine", "Middle East", "Israel Gaza",
    
    # US Federal Reserve
    "Fed", "Federal Reserve", "US Federal Reserve", "FOMC", "interest rate Fed",
    "US inflation", "US CPI", "US jobs report",
    
    # RBI (India)
    "RBI", "Reserve Bank of India", "RBI policy", "RBI rate", "RBI governor",
    
    # Political news affecting markets
    "election", "budget", "government policy", "parliament",
    
    # Stock market news
    "NSE", "BSE", "Sensex", "Nifty", "stock market", "Indian stock market",
    "stock market today", "market rally", "market crash",
    
    # Corporate earnings
    "quarterly results", "earnings", "Q4 results", "revenue growth",
    "Infosys", "TCS", "Reliance", "Wipro", "HDFC", "ICICI", "SBI",
    
    # Tax, budget, economic policy
    "GST", "tax", "customs duty", "budget 2024", "budget 2025",
    "economic policy", "fiscal deficit",
    
    # Inflation, interest rates
    "inflation", "CPI", "WPI", "interest rates", "repo rate",
    "reverse repo", "monetary policy",
    
    # Market sentiment, crisis events
    "market sentiment", "FII", "DII", "foreign investors", "domestic investors",
    "banking crisis", "default", "bankruptcy", "loan default",
    "global market", "US market", "Wall Street", "Dow Jones", "NASDAQ",
    "Asian markets", "China market", "Japan market",
]

def _build_market_query() -> str:
    """Build a comprehensive query for market-wide news."""
    return " OR ".join(MARKET_KEYWORDS[:15])  # Limit to prevent overly long queries

# ==============================================================================
# SENTIMENT SCORING KEYWORDS
# ==============================================================================
_POSITIVE_WORDS = {
    'surge', 'jump', 'rally', 'gain', 'rise', 'up', 'record', 'high', 'profit',
    'growth', 'beat', 'strong', 'bullish', 'upgraded', 'outperform', 'buy', 'positive',
    'good', 'excellent', 'awarded', 'wins', 'launch', 'expands', 'deal', 'partnership',
    'boost', 'soar', 'skyrocket', 'breakout', 'momentum'
}
_NEGATIVE_WORDS = {
    'fall', 'drop', 'crash', 'decline', 'down', 'loss', 'weak', 'sell', 'bear',
    'bearish', 'cut', 'downgrade', 'underperform', 'fraud', 'penalty', 'ban',
    'concern', 'risk', 'miss', 'slump', 'plunge', 'warning', 'probe', 'fine',
    'sink', 'plummet', 'tumble', 'recession', 'crisis', 'panic', 'selloff'
}

def _keyword_sentiment(headlines: list) -> dict:
    """Fast local keyword-based sentiment — used when Gemini is unavailable."""
    pos = neg = 0
    titles = [h['title'] if isinstance(h, dict) else h for h in headlines]
    for h in titles:
        words = set(re.findall(r'\b\w+\b', h.lower()))
        pos += len(words & _POSITIVE_WORDS)
        neg += len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return {"score": 50, "label": "Neutral", "sentiment": "NEUTRAL",
                "reason": "No sentiment keywords found", "positive": 0, "negative": 0}
    score = int((pos / total) * 100)
    if score >= 60:
        label, sentiment = "Bullish", "POSITIVE"
    elif score <= 40:
        label, sentiment = "Bearish", "NEGATIVE"
    else:
        label, sentiment = "Neutral", "NEUTRAL"
    return {"score": score, "label": label, "sentiment": sentiment,
            "reason": f"Keyword scan: {pos} positive, {neg} negative signals",
            "positive": pos, "negative": neg}


# ==============================================================================
# RSS FETCH (PRIMARY SOURCE - FREE, NO RATE LIMITS)
# ==============================================================================
def _rss_fetch(query: str, max_results: int = 20) -> list[dict]:
    """Fetch from Google News RSS - primary free source."""
    # Use URL-encoded query
    encoded_query = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        response = requests.get(url, timeout=8)
        if response.status_code != 200:
            logger.warning(f"RSS fetch failed with status {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.find_all("item")[:max_results]
        
        results = []
        for item in items:
            try:
                title = item.title.text if item.title else "No Title"
                link = item.link.text if item.link else "#"
                # Try to extract pubDate for time filtering
                pub_date = None
                if item.pubDate:
                    try:
                        from email.utils import parsedate_to_datetime
                        pub_date = parsedate_to_datetime(item.pubDate.text)
                    except:
                        pass
                results.append({
                    "title": title,
                    "url": link,
                    "pub_date": pub_date.isoformat() if pub_date else None,
                    "source": "google_rss"
                })
            except Exception as e:
                logger.debug(f"Error parsing RSS item: {e}")
                continue
        return results
    except Exception as e:
        logger.error(f"RSS fetch failed: {e}")
        return []


# ==============================================================================
# DUCKDUCKGO FETCH (SECONDARY SOURCE - FREE)
# ==============================================================================
def _ddgs_fetch(query: str, max_results: int = 20) -> list[dict]:
    """Synchronous wrapper for DDGS news fetching with thread safety."""
    try:
        from duckduckgo_search import DDGS
        with _ddgs_lock:
            with DDGS() as ddgs:
                # Use news() — more targeted for financial news
                results = list(ddgs.news(query, max_results=max_results))
                if results:
                    return [{
                        "title": r.get('title', 'No Title'),
                        "url": r.get('link', r.get('url', '#')),
                        "date": r.get('date'),
                        "source": "duckduckgo"
                    } for r in results]
                # Fallback: broader text search
                results = list(ddgs.text(query, max_results=max_results))
                return [{
                    "title": r.get('title', 'No Title'),
                    "url": r.get('href', r.get('link', '#')),
                    "date": None,
                    "source": "duckduckgo"
                } for r in results]
    except Exception as e:
        logger.error(f"DDGS fetch failed for {query}: {e}")
    return []


# ==============================================================================
# TAVILY FETCH (FALLBACK ONLY - RATE LIMITED)
# ==============================================================================
def _tavily_fetch(query: str, max_results: int = 20) -> list[dict]:
    """Fetch from Tavily - fallback only, has rate limits."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        logger.debug("TAVILY_API_KEY not found, skipping Tavily")
        return []
    try:
        from tavily import TavilyClient
        res = TavilyClient(api_key=tavily_key).search(
            query=query, 
            search_depth="basic", 
            max_results=max_results
        )
        return [{
            "title": r['title'],
            "url": r['url'],
            "date": r.get('published_date'),
            "source": "tavily"
        } for r in res.get('results', [])]
    except Exception as e:
        logger.error(f"Tavily fetch failed: {e}")
    return []


# ==============================================================================
# TIME FILTERING
# ==============================================================================
def _filter_by_time(news_items: list[dict], hours: int) -> list[dict]:
    """Filter news items by time window (in hours)."""
    if not news_items:
        return []
    
    now = datetime.now()
    cutoff = now - timedelta(hours=hours)
    
    filtered = []
    for item in news_items:
        # Try to parse the date
        pub_date = item.get('pub_date') or item.get('date')
        if pub_date:
            try:
                if isinstance(pub_date, str):
                    # Try various date formats
                    for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            pub_date = datetime.strptime(pub_date, fmt)
                            break
                        except:
                            continue
                if pub_date and pub_date >= cutoff:
                    filtered.append(item)
            except Exception:
                # If date parsing fails, include the item
                filtered.append(item)
        else:
            # No date available - include it (better than losing news)
            filtered.append(item)
    
    return filtered


def _sort_by_date(news_items: list[dict], descending: bool = True) -> list[dict]:
    """Sort news items by date."""
    def get_date(item):
        pub_date = item.get('pub_date') or item.get('date')
        if pub_date:
            try:
                if isinstance(pub_date, str):
                    for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            return datetime.strptime(pub_date, fmt)
                        except:
                            continue
                return pub_date
            except:
                return datetime.min
        return datetime.min
    
    return sorted(news_items, key=get_date, reverse=descending)


# ==============================================================================
# MAIN SERVICE CLASS
# ==============================================================================
class NewsSentimentService:
    """
    News Sentiment Service with:
    - Dynamic time filtering based on open positions
    - Prioritized free news sources
    - Caching to reduce API calls
    - Market-focused search keywords
    """
    
    # Default time windows (in hours)
    HOURS_WITH_POSITIONS = 10  # More urgent - last 10 hours
    HOURS_NO_POSITIONS = 36    # Less urgent - last 36 hours
    
    # Background run interval (in seconds)
    BACKGROUND_INTERVAL = 900  # 15 minutes
    
    def __init__(self):
        self.google_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.samba_key = os.getenv("SAMBA_API_KEY")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        
        # Background task state
        self._background_task = None
        self._last_background_news = None
        self._background_running = False

        if self.google_key:
            self.google_client = genai.Client(
                api_key=self.google_key, 
                http_options={'headers': {'X-Goog-Api-Client': 'genai-local'}}
            )
        else:
            self.google_client = None

        if not self.tavily_key:
            logger.warning("TAVILY_API_KEY not found. Tavily will be used as fallback only.")

    async def fetch_news_async(self, query: str) -> list[dict]:
        """Asynchronous entry point for DDGS fetching."""
        return await asyncio.to_thread(_ddgs_fetch, query)

    def fetch_news(
        self, 
        ticker: str = None, 
        search_engine: str = "rss", 
        fallback: bool = True,
        hours: int = None,
        use_cache: bool = True
    ) -> list[dict]:
        """
        Fetch news headlines with prioritized sources.
        
        Priority order (new):
        1. Google RSS (free, no rate limits) - PRIMARY
        2. DuckDuckGo (free) - SECONDARY  
        3. Tavily (rate limited) - FALLBACK ONLY
        
        Args:
            ticker: Stock ticker (e.g., "RELIANCE")
            search_engine: Legacy parameter, now uses prioritized sources
            fallback: If True, cascade through sources on failure
            hours: Time window filter (10h with positions, 36h without)
            use_cache: Whether to use cached results
        
        Returns:
            List of news items with title, url, source, pub_date
        """
        # Build query
        if ticker:
            clean_ticker = ticker.replace(".NS", "").replace("^NS", "")
            query = f"{clean_ticker} NSE India stock market news today"
        else:
            query = _build_market_query()
        
        # Check cache first
        cache_key = f"news_{query}_{hours or 'default'}"
        if use_cache:
            cached = news_cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Try sources in priority order: RSS → DDGS → Tavily
        headlines = []
        
        # 1. Try Google RSS first (primary - free, no rate limits)
        headlines = _rss_fetch(query, max_results=25)
        if headlines:
            logger.info(f"RSS fetched {len(headlines)} items for query: {query[:50]}...")
        elif fallback:
            # 2. Try DuckDuckGo (secondary - free)
            logger.info(f"RSS empty. Trying DuckDuckGo for: {query[:50]}...")
            headlines = _ddgs_fetch(query, max_results=25)
            if headlines:
                logger.info(f"DDGS fetched {len(headlines)} items")
            else:
                # 3. Try Tavily only as last resort
                logger.info(f"DDGS empty. Trying Tavily as fallback...")
                headlines = _tavily_fetch(query, max_results=25)
                if headlines:
                    logger.info(f"Tavily fetched {len(headlines)} items")
        
        # Apply time filtering if hours specified
        if hours and headlines:
            headlines = _filter_by_time(headlines, hours)
            # Sort by date (newest first)
            headlines = _sort_by_date(headlines, descending=True)
        
        # Cache results
        if headlines and use_cache:
            news_cache.set(cache_key, headlines)
        
        return headlines

    def fetch_market_news(self, has_open_positions: bool = False) -> list[dict]:
        """
        Fetch market-wide news with dynamic time filtering.
        
        Args:
            has_open_positions: If True, use 10h window. If False, use 36h window.
        
        Returns:
            Filtered and sorted news list
        """
        hours = self.HOURS_WITH_POSITIONS if has_open_positions else self.HOURS_NO_POSITIONS
        return self.fetch_news(ticker=None, hours=hours, fallback=True)

    def score_sentiment(
        self, 
        headlines: list, 
        provider: str = "google", 
        model_name: str = "gemini-3.1-pro"
    ) -> dict:
        """Score news sentiment using the user's selected AI provider."""
        if not headlines:
            return {"sentiment": "NEUTRAL", "reason": "No news found"}

        # Extract titles for AI prompt
        titles = [h['title'] if isinstance(h, dict) else h for h in headlines]
        prompt = f"""
        Analyze the following recent news headlines for an Indian stock and determine the overall sentiment.
        Headlines: {titles}
        
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

    # ==============================================================================
    # BACKGROUND NEWS FETCHING
    # ==============================================================================
    async def start_background_fetch(self, state, manager, interval: int = None):
        """
        Start background news fetching service.
        Runs periodically regardless of market hours.
        
        Args:
            state: AppState instance for checking open positions
            manager: WebSocket manager for broadcasting
            interval: Fetch interval in seconds (default: 15 minutes)
        """
        if self._background_running:
            logger.warning("Background news service already running")
            return
        
        self._background_running = True
        interval = interval or self.BACKGROUND_INTERVAL
        logger.info(f"Starting background news service (interval: {interval}s)")
        
        while self._background_running:
            try:
                # Check if there are open positions
                has_positions = len(getattr(state, 'open_trades', [])) > 0
                
                # Fetch market news with appropriate time window
                news = await asyncio.to_thread(
                    self.fetch_market_news, 
                    has_open_positions=has_positions
                )
                
                if news:
                    # Score sentiment
                    sentiment = _keyword_sentiment(news[:10])  # Score top 10 headlines
                    
                    # Store in state for UI access
                    state.last_market_news = {
                        "headlines": news[:20],  # Keep top 20
                        "sentiment": sentiment,
                        "has_open_positions": has_positions,
                        "hours_filtered": self.HOURS_WITH_POSITIONS if has_positions else self.HOURS_NO_POSITIONS,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Broadcast to connected clients
                    await manager.broadcast({
                        "type": "market_news",
                        "data": state.last_market_news
                    })
                    
                    logger.info(f"Background news: {len(news)} items, sentiment: {sentiment['label']}")
                else:
                    logger.debug("Background news: No news fetched")
                    
            except Exception as e:
                logger.error(f"Background news fetch error: {e}", exc_info=True)
            
            # Sleep until next interval
            await asyncio.sleep(interval)
    
    def stop_background_fetch(self):
        """Stop the background news fetching service."""
        self._background_running = False
        logger.info("Background news service stopped")


# ==============================================================================
# LEGACY FUNCTIONS (for backward compatibility)
# ==============================================================================
def _keyword_sentiment_legacy(headlines: list) -> dict:
    """Legacy wrapper for backward compatibility."""
    return _keyword_sentiment(headlines)
