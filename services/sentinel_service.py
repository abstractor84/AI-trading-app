import logging
import asyncio
import threading
from duckduckgo_search import DDGS
from datetime import datetime, timedelta

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
        
        # Keep track of seen URLs/titles to avoid duplicates
        seen_headlines = set()
        
        # ─── MODE 1: WITH POSITION (High-Priority) ───────────────────
        if open_trades:
            pos_tickers = [t['ticker'].replace('.NS', '') for t in open_trades]
            # FIXED: Use simpler query format - DDGS doesn't handle complex OR queries well
            # Just use first ticker for the search - avoid "OR OR OR" pattern
            if len(pos_tickers) == 1:
                pos_query = f"{pos_tickers[0]} stock news India"
            else:
                # For multiple tickers, just search for generic market news
                pos_query = f"India stock market news"
            macro_query = "India market news war tariff sanctions politics global shock RBI Fed"
            
            for q in [pos_query, macro_query]:
                results = await self._fetch_news(q, state, manager)
                logger.info(f"Sentinel: Fetched {len(results)} items for query '{q}'")
                for item in results:
                    title = self._safe_get(item, ['title', 'text'])
                    url = self._safe_get(item, ['url', 'link', 'href'])
                    
                    # Skip duplicates
                    headline_key = title.lower().strip()
                    if headline_key in seen_headlines:
                        continue
                    seen_headlines.add(headline_key)
                    
                    found_keywords = self._check_keywords(title)
                    
                    # Determine sentiment using keyword-based approach (no AI)
                    sentiment = self._get_sentiment_from_keywords(found_keywords)
                    
                    # Always add to the unified news feed
                    headline = {
                        "ticker": "POSITIONS" if q == pos_query else "MACRO",
                        "title": title,
                        "url": url,
                        "sentiment": sentiment,
                        "time": item.get('time', 'Just now')
                    }
                    all_news.append(headline)

                    if found_keywords:
                        alert = {
                            "ticker": "CRITICAL",
                            "title": title,
                            "url": url,
                            "priority": "HIGH",
                            "sentiment": sentiment,
                            "time": "Just now"
                        }
                        if manager:
                            await manager.broadcast({"type": "sentinel_alert", "data": alert})

        # ─── MODE 2: WITHOUT POSITION (Opportunity Scan) ─────────────
        watchlist = [t.replace('.NS', '') for t in watchlist_tickers if t not in [p['ticker'] for p in open_trades]]
        if watchlist:
            # FIXED: Removed invalid (after:24h) parameter
            watch_query = f"{' OR '.join(watchlist[:5])} stock news India breaking surge order"
            results = await self._fetch_news(watch_query, state, manager)
            for item in results:
                title = self._safe_get(item, ['title', 'text'])
                url = self._safe_get(item, ['url', 'link', 'href'])
                
                # Skip duplicates
                headline_key = title.lower().strip()
                if headline_key in seen_headlines:
                    continue
                seen_headlines.add(headline_key)
                
                found_keywords = self._check_keywords(title)
                
                # Keyword-based sentiment
                sentiment = self._get_sentiment_from_keywords(found_keywords)
                
                all_news.append({
                    "ticker": "WATCHLIST",
                    "title": title,
                    "url": url,
                    "sentiment": sentiment,
                    "time": item.get('time', 'Recent')
                })
        
        # ─── MODE 3: ALWAYS RUN (General Market News) ─────────────────
        # This ensures news is always fetched even with no positions or watchlist
        if not open_trades and not watchlist:
            # Fetch general market news - no specific ticker needed
            general_queries = [
                "India stock market news today",
                "Nifty 50 Sensex latest news",
                "RBI monetary policy news India"
            ]
            for query in general_queries:
                results = await self._fetch_news(query, state, manager)
                for item in results:
                    title = self._safe_get(item, ['title', 'text'])
                    url = self._safe_get(item, ['url', 'link', 'href'])
                    
                    # Skip duplicates
                    headline_key = title.lower().strip()
                    if headline_key in seen_headlines:
                        continue
                    seen_headlines.add(headline_key)
                    
                    found_keywords = self._check_keywords(title)
                    sentiment = self._get_sentiment_from_keywords(found_keywords)
                    
                    all_news.append({
                        "ticker": "MARKET",
                        "title": title,
                        "url": url,
                        "sentiment": sentiment,
                        "time": item.get('time', 'Just now')
                    })
        
        # Sort by time (most recent first) - handled in _fetch_ddgs
        # Also sort by sentiment priority (Bullish/Bearish first, then Neutral)
        sentiment_priority = {"Bullish": 0, "Bearish": 0, "Neutral": 1}
        all_news.sort(key=lambda x: (sentiment_priority.get(x.get('sentiment', 'Neutral'), 1), x.get('time', '')))
        
        # Broadcast the unified feed to the Sentinel UI component
        if manager and all_news:
            await manager.broadcast({
                "type": "news_update",
                "data": {"headlines": all_news}
            })
        
        return all_news

    def _get_sentiment_from_keywords(self, keywords: set) -> str:
        """
        Keyword-based sentiment analysis - NO AI needed.
        Returns "Bullish", "Bearish", or "Neutral" based on keyword matching.
        """
        positive_keywords = {"order", "win", "contract", "awarded", "merger", "acquisition", 
                           "surge", "surge", "growth", "profit", "boost", "rally", "gain",
                           "record", "high", "upgrade", "outperform", "buy"}
        negative_keywords = {"probe", "fraud", "scam", "penalty", "default", "bankruptcy", 
                          "crash", "plunge", "cut", "downgrade", "ban", "regulatory",
                          "sanctions", "war", "tariff", "recession", "crisis", "loss"}
        
        has_positive = bool(keywords & positive_keywords)
        has_negative = bool(keywords & negative_keywords)
        
        if has_positive and not has_negative:
            return "Bullish"
        elif has_negative and not has_positive:
            return "Bearish"
        elif has_positive and has_negative:
            return "Neutral"  # Mixed signals
        return "Neutral"

    async def _fetch_news(self, query: str, state=None, manager=None):
        """
        Sentinel news scanning MUST always use DDGS (Free/Unlimited).
        """
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
        """Fetch news using DDGS with thread safety and safe field extraction.
        
        FIXED: Removed invalid (after:24h) parameter. DuckDuckGo doesn't support this syntax.
        Instead, we fetch more results and filter locally by date.
        """
        import os
        if os.getenv("SIMULATION", "false").lower() == "true":
            return [
                {"title": f"SIM: Breaking news for {query[:20]}...", "url": "http://sim.news/1", "time": "Just now"},
                {"title": "SIM: Market remains in steady trend", "url": "http://sim.news/2", "time": "1 hour ago"}
            ]
        try:
            def sync_fetch():
                from datetime import datetime, timedelta
                with _ddgs_lock:
                    with DDGS() as ddgs:
                        # FIXED: Removed (after:24h) - use plain query
                        results = list(ddgs.news(query, region="in-en", max_results=30))
                        processed = []
                        for r in results:
                            title = self._safe_get(r, ['title', 'text'], 'No Title')
                            url = self._safe_get(r, ['url', 'link', 'href'], '#')
                            date_str = r.get('date', 'Just now')
                            
                            # SKEPTIC: Strict 24h filter. DDGS date is often "2024-03-16T..."
                            # or relative "2 hours ago".
                            is_old = False
                            
                            # ISO Date String Check (e.g., 2026-03-17T...)
                            now_dt = datetime.now()
                            today_prefix = now_dt.strftime("%Y-%m-%d")
                            yesterday_prefix = (now_dt - timedelta(days=1)).strftime("%Y-%m-%d")
                            
                            # Absolute old years
                            if any(yr in date_str for yr in ["2025", "2024", "2023"]):
                                if today_prefix not in date_str and yesterday_prefix not in date_str:
                                    is_old = True
                            
                            # Relative time keywords - check for recent
                            recent_keywords = ["hour", "minute", "now", "just", "h ago", "m ago", "seconds"]
                            is_recent_text = any(k in date_str.lower() for k in recent_keywords)
                            
                            if not is_recent_text:
                                # If it looks like an ISO date
                                if "T" in date_str and len(date_str) > 10:
                                    if today_prefix not in date_str and yesterday_prefix not in date_str:
                                        is_old = True
                                elif not ("day ago" in date_str.lower() or "1 day" in date_str.lower()):
                                    # If not absolute ISO and not explicitly '1 day ago', and not recent keywords
                                    # it's likely 2+ days or older.
                                    if len(date_str) > 5: # Some text date exists
                                        is_old = True
                            
                            if not is_old:
                                processed.append({"title": title, "url": url, "time": date_str})
                        
                        # Sort by time - more recent first
                        def get_sort_time(item):
                            date_str = item.get('time', '')
                            now = datetime.now()
                            
                            # Check for relative times
                            date_lower = date_str.lower()
                            if 'just' in date_lower or 'minute' in date_lower or 'second' in date_lower:
                                return now
                            if 'hour' in date_lower or 'h ago' in date_lower:
                                try:
                                    hrs = int(''.join(filter(str.isdigit, date_str)) or '1')
                                    return now - timedelta(hours=hrs)
                                except:
                                    return now - timedelta(hours=1)
                            if 'day' in date_lower:
                                try:
                                    days = int(''.join(filter(str.isdigit, date_str)) or '1')
                                    return now - timedelta(days=days)
                                except:
                                    return now - timedelta(days=1)
                            
                            # Try ISO format
                            try:
                                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            except:
                                return now - timedelta(days=2)  # Old unknown dates go last
                        
                        processed.sort(key=get_sort_time, reverse=True)
                        return processed[:15]  # Return top 15 most recent
            
            return await asyncio.to_thread(sync_fetch)
        except Exception as e:
            logger.error(f"SKEPTIC: DDGS fetch failed for {query}: {e}")
            return []

    async def _fetch_headlines(self, query: str) -> list:
        """Fetch latest news titles (legacy compatibility)."""
        res = await self._fetch_headlines_with_links(query)
        return [h.get('title', '') for h in res]

    async def _fetch_headlines_with_links(self, query: str) -> list:
        """Fetch latest news titles and links via DDGS with strict 24h filtering."""
        return await self._fetch_ddgs(query)

    def _check_keywords(self, text: str) -> set:
        """Check if any critical keywords are in the text."""
        text_lower = text.lower()
        import re
        words = set(re.findall(r'\b\w+\b', text_lower))
        return words & self.CRITICAL_KEYWORDS

sentinel_svc = SentinelService()
