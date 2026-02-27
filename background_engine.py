"""
Phase-Aware Background Engine (V2)
Replaces the old monolithic background_market_loop with a session-aware
scheduler that adapts behavior based on market phase.
"""
import asyncio
import logging
import json
from datetime import datetime

from services.market_phase import market_phase_svc, MarketPhase
from services.risk_engine import risk_engine
from services.projection_mapper import projection_svc
from services.technical_analysis import TechnicalAnalysisService
from services.stock_discovery import StockDiscoveryService
from services.ai_scorer import ai_advisor

logger = logging.getLogger(__name__)

ta_svc = TechnicalAnalysisService()
discovery_svc = StockDiscoveryService()


class BackgroundEngine:
    """
    Phase-aware background loop that:
    1. Refreshes market data continuously (every 30s)
    2. Updates open trade P&L, trailing SL, and risk advice
    3. Schedules AI calls based on market phase (≤7/day)
    4. Generates projections during POWER_HOUR
    5. Broadcasts full state to connected clients
    """

    def __init__(self, state, manager, ai_svc=None):
        self.state = state
        self.manager = manager
        self.ai_svc = ai_svc
        self._last_ai_call_time = None
        self._ai_call_count_today = 0
        self._last_date = None

    async def run(self):
        """Main background loop — runs continuously while the app is alive."""
        logger.info("V2 Background Engine started")

        while True:
            try:
                # Daily reset check
                today = datetime.now().date()
                if self._last_date != today:
                    self._last_date = today
                    self._ai_call_count_today = 0
                    risk_engine.reset_daily()
                    self.state.check_daily_reset()
                    logger.info("Daily reset complete")

                # Get current market context
                phase_ctx = market_phase_svc.get_phase_context()
                phase = phase_ctx["phase"]

                # ─── Step 0: Refresh Connection Status ─────────────────
                try:
                    conn_status = ta_svc.get_connection_status()
                    self.state.connection_status = conn_status # Store in state
                except Exception as e:
                    logger.warning(f"Connection status check failed: {e}")

                # ─── Step 1: Refresh Global Indices (always) ────────────
                try:
                    global_ctx = await asyncio.to_thread(
                        discovery_svc.fetch_global_indices
                    )
                    self.state.global_context = global_ctx
                except Exception as e:
                    logger.warning(f"Global indices fetch failed: {e}")

                # ─── Step 2: Update Open Trades (P&L, trailing SL, advice) ──
                if self.state.open_trades:
                    await self._update_open_trades(phase_ctx)

                # ─── Step 3: Phase-specific AI calls ────────────────────
                if phase_ctx["is_trading_hours"]:
                    await self._maybe_call_ai(phase_ctx)

                # ─── Step 4: Broadcast full state to all clients ────────
                await self._broadcast_state(phase_ctx)

                # ─── Step 5: Phase transition notifications ─────────────
                if phase_ctx["transitioned"]:
                    await self.manager.broadcast({
                        "type": "notification",
                        "message": f"📍 {phase_ctx['phase_label']}",
                        "level": "info"
                    })

            except Exception as e:
                logger.error(f"Background engine error: {e}", exc_info=True)

            # Sleep interval based on phase
            sleep_secs = self._get_sleep_interval(phase_ctx["phase"])
            await asyncio.sleep(sleep_secs)

    async def _update_open_trades(self, phase_ctx):
        """Update P&L, trailing SL, projections, and risk advice for all open trades."""
        mins_to_close = phase_ctx["mins_to_close"]

        for trade in self.state.open_trades:
            try:
                # Fetch latest 1-min data
                df = await asyncio.to_thread(
                    ta_svc.fetch_ohlcv, trade['ticker'], "1d", "1m"
                )
                if df is None or df.empty:
                    continue

                current_price = float(df['Close'].iloc[-1])
                trade['current_price'] = current_price

                # P&L calculation
                if trade['action'] == "BUY":
                    pnl = (current_price - trade['entry_price']) * trade['quantity']
                else:
                    pnl = (trade['entry_price'] - current_price) * trade['quantity']
                trade['pnl'] = round(pnl, 2)

                # ATR for dynamic SL
                atr = risk_engine.compute_atr(df)
                trade['atr'] = round(atr, 2)

                # Trailing SL
                trailing_sl = risk_engine.compute_trailing_sl(
                    trade['entry_price'], current_price, trade['action'], atr
                )
                trade['trailing_sl'] = trailing_sl

                # Risk-based position advice (pure math, no AI)
                advice = risk_engine.get_position_action(
                    trade, current_price, atr, mins_to_close
                )
                trade['risk_advice'] = advice

                # 3 PM Projections (during POWER_HOUR or MID_SESSION)
                if phase_ctx["phase"] in (
                    MarketPhase.MID_SESSION, MarketPhase.POWER_HOUR
                ):
                    projections = projection_svc.calculate_projections(
                        trade['ticker'], df
                    )
                    trade['projections'] = projections

            except Exception as e:
                logger.warning(f"Failed to update trade {trade['ticker']}: {e}")

    async def _maybe_call_ai(self, phase_ctx):
        """Call AI only when the phase schedule permits and quota allows."""
        schedule = market_phase_svc.get_ai_schedule()

        if schedule["prompt_type"] is None:
            return

        if self._ai_call_count_today >= 20:
            return

        # Check interval
        interval = schedule["call_interval_mins"]
        if interval <= 0:
            return

        now = datetime.now()
        if self._last_ai_call_time:
            elapsed = (now - self._last_ai_call_time).total_seconds() / 60
            if elapsed < interval:
                return

        # Determine prompt type based on open positions
        prompt_type = schedule["prompt_type"]
        if self.state.open_trades and prompt_type == "SCAN":
            prompt_type = "POSITION_REVIEW"

        self._last_ai_call_time = now
        self._ai_call_count_today += 1

        provider = getattr(self.state, 'ai_provider', 'google')
        model = getattr(self.state, 'ai_model', 'gemini-2.5-flash')
        search_engine = getattr(self.state, 'search_engine', 'ddgs')
        data_provider = getattr(self.state, 'data_provider', 'yfinance')

        logger.info(f"\n=====================================")
        logger.info(f"AI Call #{self._ai_call_count_today}/7: {prompt_type}")
        logger.info(f"Data Source: {data_provider} | Search: {search_engine}")
        logger.info(f"AI Engine: {provider} | Model: {model}")
        logger.info(f"=====================================")

        try:
            if prompt_type == "SCAN":
                # Build candidate list with TA data
                candidates = []
                top_stocks = await asyncio.to_thread(
                    discovery_svc._get_top_candidates, 8
                )
                for ticker in top_stocks[:8]:
                    ta_data = await asyncio.to_thread(
                        ta_svc.analyze_stock, ticker
                    )
                    if ta_data:
                        df = await asyncio.to_thread(
                            ta_svc.fetch_ohlcv, ticker, "5d", "5m"
                        )
                        atr = risk_engine.compute_atr(df) if df is not None else 0
                        
                        log_msg = f"  [TA] {ticker} | Close: {ta_data.get('close'):.2f} | VWAP: {ta_data.get('vwap'):.2f} | EMA9: {ta_data.get('ema_9'):.2f} | EMA21: {ta_data.get('ema_21'):.2f} | RSI: {ta_data.get('rsi_14'):.2f} | ADX: {ta_data.get('adx_14'):.2f} | Surge: {ta_data.get('vol_surge'):.2f}x"
                        
                        math_prob = ta_svc.evaluate_math_probability(ta_data)
                        ta_data['math_prob'] = math_prob
                        
                        if math_prob >= 0.50:
                            candidates.append({
                                "ticker": ticker, "ta_data": ta_data, "atr": atr, "math_prob": math_prob
                            })
                            logger.info(log_msg + f" -> [MATH: PASS] (Score: {math_prob})")
                        else:
                            logger.info(log_msg + f" -> [MATH: FAIL] (Score: {math_prob})")

                if not candidates:
                    logger.info("No stocks passed the mathematical setup pre-filter. Skipping AI scan to save API limits.")
                    return  # Fast-exit right back out of the loop!

                logger.info(f"Passing {len(candidates)} mathematically validated Candidates to AI Scorer...")
                raw_result = await asyncio.to_thread(
                    ai_advisor.scan_market, candidates,
                    self.state.global_context, phase_ctx, provider, model
                )
                
                # Enrich results
                result = []
                if isinstance(raw_result, list):
                    for pick in raw_result:
                        ticker = pick.get("ticker", "")
                        candidate = next((c for c in candidates if c["ticker"] == ticker), None)
                        if not candidate:
                            continue
                        
                        ta_data = candidate["ta_data"]
                        atr = candidate["atr"]
                        entry_price = ta_data.get("close", 0)
                        trade_action = pick.get("action", "BUY").upper().strip()
                        vix_value = self.state.global_context.get("vix", {}).get("value", 0)
                        
                        risk_levels = {}
                        try:
                            validation = risk_engine.validate_trade(
                                entry_price, trade_action, atr, entry_price,
                                ta_data=ta_data, vix_value=vix_value
                            )
                            if validation.get("passed"):
                                risk_levels = {
                                    "stop_loss": round(validation["levels"]["stop_loss"], 2),
                                    "target_1": round(validation["levels"]["target_1"], 2),
                                    "target_2": round(validation["levels"]["target_2"], 2),
                                    "quantity": validation["quantity"],
                                    "risk_per_share": round(validation["levels"]["risk_per_share"], 2),
                                    "max_loss": round(validation.get("max_loss_this_trade", 0), 0),
                                }
                        except Exception:
                            pass
                            
                        # Fundamentals
                        fundamentals = {}
                        try:
                            fundamentals = await asyncio.to_thread(
                                ta_svc.fetch_fundamentals, ticker
                            )
                        except Exception:
                            pass
                            
                        # News sentiment
                        sentiment = {"score": 50, "label": "Neutral", "headline_count": 0}
                        try:
                            from services.news_sentiment import NewsSentimentService, _keyword_sentiment
                            news_svc = NewsSentimentService()
                            search_engine = getattr(self.state, 'search_engine', 'ddgs')
                            headlines = await asyncio.to_thread(
                                news_svc.fetch_news, ticker, search_engine
                            )
                            if headlines:
                                sentiment["headline_count"] = len(headlines)
                                kw_result = _keyword_sentiment(headlines)
                                sentiment["score"] = kw_result.get("score", 50)
                                sentiment["label"] = kw_result.get("label", "Neutral")
                        except Exception:
                            pass
                            
                        ta_signal = ta_svc.classify_signal(ta_data)
                        
                        if trade_action == "BUY" and ("SELL" in ta_signal):
                            continue
                        if trade_action == "SHORT SELL" and ("BUY" in ta_signal):
                            continue
                            
                        final_signal = f"VERIFIED {trade_action}" if ta_signal != "NEUTRAL" else trade_action
                        
                        # Handle NaNs
                        def _s(val, r=2):
                            try:
                                v = round(val, r)
                                import math
                                if math.isnan(v): return 0
                                return v
                            except: return 0
                                
                        result.append({
                            **pick,
                            "signal": final_signal,
                            "live_price": _s(entry_price, 2),
                            "technicals": {
                                "rsi_14": _s(ta_data.get("rsi_14", 0), 1),
                                "macd_hist": _s(ta_data.get("macd_hist", 0), 2),
                                "adx_14": _s(ta_data.get("adx_14", 0), 1),
                                "vwap": _s(ta_data.get("vwap", 0), 2),
                                "vol_surge": _s(ta_data.get("vol_surge", 0), 2),
                                "bb_upper": _s(ta_data.get("bb_upper", 0), 2),
                                "bb_lower": _s(ta_data.get("bb_lower", 0), 2),
                                "ema_9": _s(ta_data.get("ema_9", 0), 2),
                                "ema_21": _s(ta_data.get("ema_21", 0), 2),
                            },
                            "lorentzian": {
                                "score": _s(candidate.get("math_prob", 0), 2),
                                "signal": final_signal
                            },
                            "risk_levels": risk_levels,
                            "fundamentals": fundamentals,
                            "sentiment": sentiment,
                            "atr": _s(atr, 2),
                        })

            elif prompt_type == "POSITION_REVIEW":
                raw_result = await asyncio.to_thread(
                    ai_advisor.review_positions, self.state.open_trades,
                    self.state.global_context, phase_ctx, provider, model
                )
                
                result = []
                if isinstance(raw_result, list):
                    for pick in raw_result:
                        ticker = pick.get("ticker", "")
                        
                        try:
                            ta_data = await asyncio.to_thread(ta_svc.analyze_stock, ticker)
                            if not ta_data:
                                result.append(pick)
                                continue
                                
                            entry_price = ta_data.get("close", 0)
                            ta_signal = ta_svc.classify_signal(ta_data)
                            math_prob = ta_svc.evaluate_math_probability(ta_data)
                            
                            def _s(val, r=2):
                                try:
                                    v = round(val, r)
                                    import math
                                    if math.isnan(v): return 0
                                    return v
                                except: return 0
                                
                            result.append({
                                **pick,
                                "live_price": _s(entry_price, 2),
                                "technicals": {
                                    "rsi_14": _s(ta_data.get("rsi_14", 0), 1),
                                    "macd_hist": _s(ta_data.get("macd_hist", 0), 2),
                                    "adx_14": _s(ta_data.get("adx_14", 0), 1),
                                    "vwap": _s(ta_data.get("vwap", 0), 2),
                                    "vol_surge": _s(ta_data.get("vol_surge", 0), 2),
                                    "bb_upper": _s(ta_data.get("bb_upper", 0), 2),
                                    "bb_lower": _s(ta_data.get("bb_lower", 0), 2),
                                    "ema_9": _s(ta_data.get("ema_9", 0), 2),
                                    "ema_21": _s(ta_data.get("ema_21", 0), 2),
                                },
                                "lorentzian": {
                                    "score": _s(math_prob, 2),
                                    "signal": ta_signal
                                },
                            })
                        except Exception as e:
                            logger.error(f"Error enriching POSITION_REVIEW for {ticker}: {e}")
                            result.append(pick)
                else:
                    result = raw_result

            elif prompt_type == "EXIT_GUIDANCE":
                result = await asyncio.to_thread(
                    ai_advisor.exit_guidance, self.state.open_trades,
                    self.state.global_context, phase_ctx, provider, model
                )
            else:
                return

            # Store and broadcast
            self.state.ai_advisor_message = {
                "type": prompt_type,
                "result": result,
                "timestamp": now.strftime("%H:%M:%S"),
                "call_number": self._ai_call_count_today,
            }

            self.state.action_timeline.append({
                "time": now.strftime("%H:%M:%S"),
                "type": f"AI_{prompt_type}",
                "message": f"AI {prompt_type.replace('_', ' ').title()} (#{self._ai_call_count_today}/7)"
            })

            await self.manager.broadcast({
                "type": "ai_advisor_update",
                "data": self.state.ai_advisor_message,
            })

        except Exception as e:
            logger.error(f"AI call failed: {e}", exc_info=True)

    async def _broadcast_state(self, phase_ctx):
        """Send full app state to all connected clients."""
        payload = {
            "type": "state_update",
            "capital": self.state.capital,
            "max_loss": self.state.max_loss_per_trade,
            "open_trades": self.state.open_trades,
            "closed_trades": self.state.closed_trades,
            "global_context": self.state.global_context,
            "market_phase": phase_ctx,
            "ai_calls_today": __import__('services.quota_service').quota_service.quota_svc.get_total_daily_usage(),
            "ai_calls_limit": 20,
            "ai_advisor": self.state.ai_advisor_message,
            "connection_status": getattr(self.state, 'connection_status', {}),
            "action_timeline": self.state.action_timeline[-20:],  # Last 20 events
            "search_engine": getattr(self.state, 'search_engine', 'ddgs'),
            "data_provider": getattr(self.state, 'data_provider', 'yfinance'),
            "search_fallback": getattr(self.state, 'search_fallback', False),
            "auto_refresh": getattr(self.state, 'auto_refresh', True),
            "ai_provider": getattr(self.state, 'ai_provider', 'google'),
            "ai_model": getattr(self.state, 'ai_model', 'gemini-2.5-flash'),
        }
        await self.manager.broadcast(payload)

    def _get_sleep_interval(self, phase: str) -> int:
        """Faster refresh during active hours, slower otherwise."""
        intervals = {
            MarketPhase.PRE_MARKET: 60,
            MarketPhase.OPENING_15: 10,    # Fast during opening range
            MarketPhase.MID_SESSION: 30,
            MarketPhase.POWER_HOUR: 15,    # Fast during power hour
            MarketPhase.POST_MARKET: 120,
            MarketPhase.CLOSED: 300,       # 5 min when closed
        }
        return intervals.get(phase, 60)
