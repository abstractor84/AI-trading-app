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

        if self._ai_call_count_today >= 7:
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

        logger.info(f"AI call #{self._ai_call_count_today}/7: {prompt_type}")

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
                        candidates.append({
                            "ticker": ticker, "ta_data": ta_data, "atr": atr
                        })

                result = await asyncio.to_thread(
                    ai_advisor.scan_market, candidates,
                    self.state.global_context, phase_ctx, provider, model
                )

            elif prompt_type == "POSITION_REVIEW":
                result = await asyncio.to_thread(
                    ai_advisor.review_positions, self.state.open_trades,
                    self.state.global_context, phase_ctx, provider, model
                )

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
            "ai_calls_today": self._ai_call_count_today,
            "ai_calls_limit": 7,
            "ai_advisor": self.state.ai_advisor_message,
            "action_timeline": self.state.action_timeline[-20:],  # Last 20 events
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
