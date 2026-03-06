import logging
import asyncio
from duckduckgo_search import DDGS
from datetime import datetime

logger = logging.getLogger(__name__)

class SentinelService:
    """
    High-Frequency Risk Guard (The Sentinel).
    Primary Job: Monitor open positions for 'Jackpot' news or critical shocks every 60s.
    """
    
    # Keywords that indicate high-impact events
    CRITICAL_KEYWORDS = {
        "tender", "order", "win", "contract", "awarded", "merger", "acquisition",
        "probe", "fraud", "scam", "penalty", "default", "bankruptcy", "crash",
        "plunge", "surge", "acquisition", "takeover", "regulatory", "ban"
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
                new_headlines = [h for h in headlines_with_links if h['title'] not in [c['title'] for c in cached]]
                self._news_cache[ticker] = headlines_with_links
                
                if not new_headlines:
                    continue

                # Broadcast for sidebar (using all fetched headlines for display)
                if manager:
                    await manager.broadcast({
                        "type": "news_update",
                        "data": {
                            "ticker": ticker,
                            "headlines": [{"title": h['title'], "url": h['link'], "sentiment": "Neutral", "time": "Just now"} for h in headlines_with_links]
                        }
                    })

                for item in new_headlines:
                    title = item['title']
                    url = item['link']
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
    async def check_alerts(self, open_trades, manager=None):
        """
        Scan news for all open positions and identify critical alerts.
        Optionally broadcasts to the manager if provided.
        """
        if not open_trades:
            return []

        alerts = []
        for trade in open_trades:
            ticker = trade['ticker']
            clean_ticker = ticker.replace(".NS", "")
            
            try:
                query = f"{clean_ticker} stock news breaking"
                headlines_with_links = await self._fetch_headlines_with_links(query)
                
                for item in headlines_with_links:
                    title = item['title']
                    url = item['link']
                    found_keywords = self._check_keywords(title)
                    if found_keywords:
                        alert = {
                            "type": "CRITICAL_ALERT",
                            "ticker": ticker,
                            "title": title,
                            "url": url,
                            "keywords": list(found_keywords),
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "level": "danger" if any(k in ["probe", "fraud", "penalty", "crash"] for k in found_keywords) else "success"
                        }
                        alerts.append(alert)
                        logger.warning(f"SENTINEL ALERT for {ticker}: {title}")
                        
                        if manager:
                            await manager.broadcast(alert)
                        break 
                        
            except Exception as e:
                logger.error(f"Sentinel check failed for {ticker}: {e}")

        return alerts

    async def _fetch_headlines(self, query: str) -> list:
        """Fetch latest news titles (legacy compatibility)."""
        res = await self._fetch_headlines_with_links(query)
        return [h['title'] for h in res]

    async def _fetch_headlines_with_links(self, query: str) -> list:
        """Fetch latest news titles and links via DDGS."""
        try:
            def get_news():
                with DDGS() as ddgs:
                    results = list(ddgs.news(query, max_results=5))
                    return [{"title": r['title'], "link": r['link']} for r in results]
            
            return await asyncio.to_thread(get_news)
        except Exception as e:
            logger.debug(f"DDGS fetch in Sentinel failed: {e}")
            return []

    def _check_keywords(self, text: str) -> set:
        """Check if any critical keywords are in the text."""
        text_lower = text.lower()
        import re
        words = set(re.findall(r'\b\w+\b', text_lower))
        return words & self.CRITICAL_KEYWORDS

sentinel_svc = SentinelService()
