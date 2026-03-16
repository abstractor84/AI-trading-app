import logging
import asyncio
import threading
from duckduckgo_search import DDGS
from datetime import datetime

logger = logging.getLogger(__name__)
_ddgs_lock = threading.Lock()

class SentinelService:
    """
    High-Frequency Risk Guard (The Sentinel).
    Primary Job: Monitor open positions for 'Jackpot' news or critical shocks every 60s.
    """
    
    # Keywords that indicate high-impact events
    CRITICAL_KEYWORDS = {
        "tender", "order", "win", "contract", "awarded", "merger", "acquisition",
        "probe", "fraud", "scam", "penalty", "default", "bankruptcy", "crash",
        "plunge", "surge", "acquisition", "takeover", "regulatory", "ban",
        "war", "tariff", "tariffs", "sanctions", "politics", "global", "shock", "elections",
        "inflation", "gdp", "rbi", "fed", "recession", "stimulus", "interest", "rate"
    }

    def __init__(self):
        self._last_check_time = {}
        self._news_cache = {} # Ticker -> list of headlines

    async def check_for_opportunities(self, watchlist: list, state, manager=None) -> list:
        """
        Scan news for watchlist stocks and decide if an AI scan should be triggered.
        Returns a list of tickers that warrant an AI scan.
        """
        triggers = []
        for ticker in watchlist:
            try:
                # Check for breaking news
                clean_ticker = ticker.replace(".NS", "")
                query = f"{clean_ticker} stock price sensitive news"
                headlines_with_links = await self._fetch_headlines_with_links(query)
                
                # Check if news is cached (avoid redundant triggers for same news)
                cached = self._news_cache.get(ticker, [])
                new_headlines = [h for h in headlines_with_links if h.get('title', '') not in [c.get('title', '') for c in cached]]
                self._news_cache[ticker] = headlines_with_links
                
                if not new_headlines:
                    continue

                # Broadcast for sidebar (using all fetched headlines for display)
                if manager:
                    await manager.broadcast({
                        "type": "news_update",
                        "data": {
                            "ticker": ticker,
                            "headlines": [{"title": self._safe_get(h, ['title', 'text']), "url": self._safe_get(h, ['url', 'link', 'href']), "sentiment": "Neutral", "time": "Just now"} for h in headlines_with_links]
                        }
                    })

                for item in new_headlines:
                    title = item.get('title', '')
                    url = item.get('url', '')
                    found_keywords = self._check_keywords(title)
                    if found_keywords:
                        # Local Appraisal before calling AI
                        should_call = await self.should_invoke_ai(ticker, title, found_keywords, state)
                        if should_call:
                            # Broadcast for sidebar (specific to the appraised positive news)
                            if manager:
                                await manager.broadcast({
                                    "type": "news_update",
                                    "data": {
                                        "ticker": ticker,
                                        "headlines": [{"title": title, "url": url, "sentiment": "Bullish", "time": "Just now"}]
                                    }
                                })

                            triggers.append({"ticker": ticker, "reason": "Positive news shock detected and appraised."})
                            break # Trigger once per stock
            except Exception as e:
                logger.error(f"Opportunity check failed for {ticker}: {e}")
        
        return triggers

    async def should_invoke_ai(self, ticker, news_title, keywords, state) -> bool:
        """
        Decision matrix to determine if AI should be invoked for a news-based setup.
        Considers capital, inventory, and rough technical alignment.
        """
        # 1. Capital Check: If capital is too low (< 10% of start), skip scans
        if state.capital < 1000: # Assuming 1k as minimum for any meaningful trade
            return False

        # 2. Inventory Check: If already holding, Sentinel will alert via check_alerts
        # No need to trigger a *new* trade AI scan for something we already own
        if any(t['ticker'] == ticker for t in state.open_trades):
            return False

        # 3. Technical Filtering (Local/Cheap)
        from services.technical_analysis import ta_svc
        try:
            ta_data = await asyncio.to_thread(ta_svc.analyze_stock, ticker)
            if not ta_data:
                return False

            # Rough alignment:
            is_bullish = ta_data.get('ema_9') > ta_data.get('ema_21')
            rsi = ta_data.get('rsi_14', 50)
            vol_surge = ta_data.get('vol_surge', 1.0)
            change_pct = ta_data.get('change_pct', 0.0)
            
            # Sentiment weighting (simple)
            is_positive = any(k in ["order", "win", "contract", "surge", "merger"] for k in keywords)
            is_negative = any(k in ["probe", "fraud", "penalty", "default", "crash", "plunge"] for k in keywords)

            # Jackpot Detection: Positive news + Price Momentum + Volume Surge
            if is_positive:
                if (change_pct > 0.5 or vol_surge > 1.5) and rsi < 75:
                    return True
                
            # Exit/Short Detection: Critical Negative News + High Volatility
            if is_negative:
                if (change_pct < -0.5 or vol_surge > 1.5):
                    return True
                
        except Exception as e:
            logger.error(f"Local appraisal failed for {ticker}: {e}")
            
        return False
    def _safe_get(self, item, keys, default=""):
        if not isinstance(item, dict): return default
        for k in keys:
            if k in item: return item[k]
        return default

    async def check_alerts(self, open_trades, watchlist_tickers, state, manager=None):
        """
        Dual-mode News Sentinel:
        1. WITH POSITION: High-priority monitoring for open trades (Macro/Sector/Ticker).
        2. WITHOUT POSITION: Opportunity scanning for watchlist stocks.
        """
        all_news = []
        
        # ─── MODE 1: WITH POSITION (High-Priority) ───────────────────
        if open_trades:
            pos_tickers = [t['ticker'].replace('.NS', '') for t in open_trades]
            # Combined query for speed and context
            pos_query = f"(after:24h) ({' OR '.join(pos_tickers)}) (stock news OR breaking OR results OR alert)"
            macro_query = "(after:24h) India market (war OR tariff OR sanctions OR politics OR global shock OR RBI OR Fed)"
            
            for q in [pos_query, macro_query]:
                results = await self._fetch_news(q, state, manager)
                logger.info(f"Sentinel: Fetched {len(results)} items for query '{q}'")
                for item in results:
                    title = self._safe_get(item, ['title', 'text'])
                    url = self._safe_get(item, ['url', 'link', 'href'])
                    found_keywords = self._check_keywords(title)
                    
                    # Always add to the unified news feed
                    headline = {
                        "ticker": "POSITIONS" if q == pos_query else "MACRO",
                        "title": title,
                        "url": url,
                        "sentiment": "Negative" if any(k in ["crash", "plunge", "sanctions", "war", "fraud"] for k in found_keywords) else "Bullish" if any(k in ["order", "win", "win", "surge"] for k in found_keywords) else "Neutral",
                        "time": "Just now"
                    }
                    all_news.append(headline)

                    if found_keywords:
                        alert = {
                            "ticker": "CRITICAL",
                            "title": title,
                            "url": url,
                            "priority": "HIGH",
                            "sentiment": headline["sentiment"],
                            "time": "Just now"
                        }
                        if manager:
                            await manager.broadcast({"type": "sentinel_alert", "data": alert})

        # ─── MODE 2: WITHOUT POSITION (Opportunity Scan) ─────────────
        watchlist = [t.replace('.NS', '') for t in watchlist_tickers if t not in [p['ticker'] for p in open_trades]]
        if watchlist:
            watch_query = f"(after:24h) ({' OR '.join(watchlist[:5])}) stock news breaking OR surge OR order"
            results = await self._fetch_news(watch_query, state, manager)
            for item in results:
                title = self._safe_get(item, ['title', 'text'])
                url = self._safe_get(item, ['url', 'link', 'href'])
                found_keywords = self._check_keywords(title)
                
                all_news.append({
                    "ticker": "WATCHLIST",
                    "title": title,
                    "url": url,
                    "sentiment": "Bullish" if any(k in ["order", "win", "contract", "surge"] for k in found_keywords) else "Neutral",
                    "time": "Recent"
                })
        
        # Broadcast the unified feed to the Sentinel UI component
        if manager and all_news:
            await manager.broadcast({
                "type": "news_update",
                "data": {"headlines": all_news}
            })
        
        return all_news

    async def _fetch_news(self, query: str, state=None, manager=None):
        """
        Sentinel news scanning MUST always use DDGS (Free/Unlimited).
        """
        import os
        if os.getenv("SIMULATION", "false").lower() == "true":
            # SKEPTIC: Avoid DDGS in simulation to prevent segment faults and save quotas/API rate limits.
            return [
                {"title": f"SIM: Breaking news for {query[:20]}...", "url": "http://sim.news/1", "text": "Simulated news content."},
                {"title": "SIM: Market remains in steady trend", "url": "http://sim.news/2", "text": "Steady trend continues."}
            ]
        try:
            return await self._fetch_ddgs(query)
        except Exception as e:
            logger.error(f"Sentinel news fetch error (DDGS): {e}")
            if manager:
                asyncio.create_task(manager.broadcast({
                    "type": "notification",
                    "message": "🛡️ Sentinel: News feed temporarily unavailable.",
                    "level": "error"
                }))
            return []

    async def _fetch_ddgs(self, query: str):
        """Fetch news using DDGS with thread safety and safe field extraction."""
        try:
            def sync_fetch():
                with _ddgs_lock:
                    with DDGS() as ddgs:
                        results = list(ddgs.news(query, region="in-en", max_results=20))
                        return [{"title": self._safe_get(r, ['title', 'text'], 'No Title'), "url": self._safe_get(r, ['url', 'link', 'href'], '#')} for r in results]
            
            return await asyncio.to_thread(sync_fetch)
        except Exception as e:
            logger.error(f"SKEPTIC: DDGS fetch failed for {query}: {e}")
            return []

    async def _fetch_headlines(self, query: str) -> list:
        """Fetch latest news titles (legacy compatibility)."""
        res = await self._fetch_headlines_with_links(query)
        return [h.get('title', '') for h in res]

    async def _fetch_headlines_with_links(self, query: str) -> list:
        """Fetch latest news titles and links via DDGS."""
        import os
        if os.getenv("SIMULATION", "false").lower() == "true":
            return [{"title": f"SIM: News headline for {query[:10]}", "url": "http://sim.news"}]
        try:
            def get_news():
                with _ddgs_lock:
                    with DDGS() as ddgs:
                        results = list(ddgs.news(query, max_results=20))
                        return [{"title": self._safe_get(r, ['title', 'text'], 'No Title'), "url": self._safe_get(r, ['url', 'link', 'href'], '#')} for r in results]

            return await asyncio.to_thread(get_news)
        except Exception as e:
            logger.debug(f"SKEPTIC: DDGS fetch in Sentinel failed: {e}")
            return []

    def _check_keywords(self, text: str) -> set:
        """Check if any critical keywords are in the text."""
        text_lower = text.lower()
        import re
        words = set(re.findall(r'\b\w+\b', text_lower))
        return words & self.CRITICAL_KEYWORDS

sentinel_svc = SentinelService()
