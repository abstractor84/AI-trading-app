"""
WebSocket Handler (V2)
Handles all WebSocket connections and command routing.
Extracted from the monolithic main.py for clean separation of concerns.
Enhanced with rate limit handling and graceful degradation.
"""
import json
import asyncio
import threading
import logging
import pandas as pd
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

from services.risk_engine import risk_engine
from services.market_phase import market_phase_svc
from services.sentinel_service import sentinel_svc

logger = logging.getLogger(__name__)


# Rate limit state for WebSocket broadcast
_rate_limit_broadcast_state = {
    "last_broadcast_time": 0,
    "broadcast_count": 0,
    "rate_limited": False,
    "rate_limit_until": 0
}

# Thread lock for rate limit state (thread-safety)
_rate_limit_lock = threading.Lock()

# Rate limit constants
RATE_LIMIT_WINDOW_MS = 1000  # 1 second window
RATE_LIMIT_MAX_BROADCASTS = 10  # Max broadcasts per window
RATE_LIMIT_COOLDOWN_MS = 5000  # 5 second cooldown when rate limited


def check_broadcast_rate_limit():
    """Check if we should rate limit broadcasts. Thread-safe."""
    import time
    current_time_ms = int(time.time() * 1000)
    
    with _rate_limit_lock:
        if _rate_limit_broadcast_state["rate_limited"]:
            if current_time_ms < _rate_limit_broadcast_state["rate_limit_until"]:
                return True
            else:
                # Cooldown expired, reset
                _rate_limit_broadcast_state["rate_limited"] = False
                _rate_limit_broadcast_state["broadcast_count"] = 0
        
        # Check if we're exceeding the rate limit
        if current_time_ms - _rate_limit_broadcast_state["last_broadcast_time"] < RATE_LIMIT_WINDOW_MS:
            _rate_limit_broadcast_state["broadcast_count"] += 1
            if _rate_limit_broadcast_state["broadcast_count"] > RATE_LIMIT_MAX_BROADCASTS:
                _rate_limit_broadcast_state["rate_limited"] = True
                _rate_limit_broadcast_state["rate_limit_until"] = current_time_ms + RATE_LIMIT_COOLDOWN_MS
                logger.warning(f"Rate limiting broadcasts for {RATE_LIMIT_COOLDOWN_MS}ms")
                return True
        else:
            # Reset counter for new window
            _rate_limit_broadcast_state["broadcast_count"] = 1
            _rate_limit_broadcast_state["last_broadcast_time"] = current_time_ms
        
        return False


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message with rate limiting."""
        # Skip if rate limited (but still log important messages)
        if message.get("type") not in ["tick", "notification"] and check_broadcast_rate_limit():
            logger.debug("Skipping broadcast due to rate limiting")
            return
        
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception as e:
                dead.append(conn)
        for d in dead:
            self.disconnect(d)

    async def send_state(self, websocket: WebSocket, state):
        """Send full initial state to a newly connected client."""
        try:
            from services.market_phase import market_phase_svc
            phase_ctx = market_phase_svc.get_phase_context()
            
            from services.quota_service import quota_svc
            from services.holiday_service import holiday_svc
            
            # Import rate limit status
            try:
                from services.upstox_service import get_rate_limit_status, is_rate_limited
                rate_limit_status = get_rate_limit_status()
            except ImportError:
                rate_limit_status = {"is_rate_limited": False, "remaining_cooldown": 0}
            
            ai_prov = getattr(state, 'ai_provider', 'google')
            try:
                quota_status = quota_svc.check_quota(ai_prov)
            except Exception as e:
                quota_status = {"limit_rpd": 20, "remaining_rpd": 20}

            is_h, h_name = holiday_svc.is_holiday()
            upcoming = holiday_svc.get_upcoming_holiday()

            # Get data provider quota info
            data_provider = getattr(state, 'data_provider', 'upstox')
            yfinance_quota = quota_svc.check_quota("yfinance")
            upstox_quota = {"remaining_rpd": 5000, "limit_rpd": 5000, "used_rpd_pct": 0}  # Upstox has high limit
            
            # Determine which provider is active and fallback status
            is_using_fallback = False
            if data_provider == "yfinance" and not getattr(state, 'fallback_data', True):
                is_using_fallback = False
            elif data_provider == "yfinance":
                # Check if yfinance quota is exhausted
                is_using_fallback = not yfinance_quota.get("can_call", True)
            
            # Check if rate limited
            is_rate_limited_data = is_rate_limited()
            
            payload = {
                "type": "state_update",
                "capital": state.capital,
                "max_loss": state.max_loss_per_trade,
                "open_trades": state.open_trades,
                "closed_trades": state.closed_trades,
                "global_context": getattr(state, 'global_context', {}),
                "market_phase": phase_ctx,
                "ai_calls_today": quota_status.get("limit_rpd", 30) - quota_status.get("remaining_rpd", 30),
                "ai_calls_limit": quota_status.get("limit_rpd", 30),
                "ai_advisor": getattr(state, 'ai_advisor_message', None),
                "ai_scans_today": getattr(state, 'ai_scans_today', []),
                "connection_status": getattr(state, 'connection_status', {}),
                "action_timeline": getattr(state, 'action_timeline', [])[-20:],
                "search_engine": getattr(state, 'search_engine', 'tavily'),
                "data_provider": data_provider,
                "fallback_data": getattr(state, 'fallback_data', True),
                "fallback_search": getattr(state, 'fallback_search', True),
                "fallback_ai": getattr(state, 'fallback_ai', True),
                "ai_provider": getattr(state, 'ai_provider', 'google'),
                "ai_model": getattr(state, 'ai_model', 'gemini-3.1-pro'),
                "data_provider_quota": {
                    "upstox": upstox_quota,
                    "yfinance": yfinance_quota
                },
                "is_using_fallback": is_using_fallback,
                "rate_limit_status": {
                    "is_rate_limited": rate_limit_status.get("is_rate_limited", False),
                    "remaining_cooldown": rate_limit_status.get("remaining_cooldown", 0),
                    "consecutive_failures": rate_limit_status.get("consecutive_failures", 0)
                },
                "is_data_stale": is_rate_limited_data,  # Flag to show stale data indicator
                "last_data_update": getattr(state, 'last_data_update', None),  # Timestamp of last successful data
                "holiday_info": {
                    "is_holiday": is_h,
                    "holiday_name": h_name,
                    "upcoming": upcoming
                }
            }
            await websocket.send_json(payload)
        except Exception as e:
            logger.error(f"SKEPTIC: send_state failed: {e}")

    async def send_news(self, websocket: WebSocket, state):
        """Send sentinel news to a newly connected client immediately.
        
        This ensures new clients get news right away without waiting for background cycle.
        Similar to send_state() but for news data.
        """
        try:
            watchlist_tickers = list(getattr(state, 'dashboard_watch_stocks', set()))
            open_trades = getattr(state, 'open_trades', [])
            # Pass manager=None to avoid broadcast, just get news returned
            news = await sentinel_svc.check_alerts(open_trades, watchlist_tickers, state, manager=None)
            if news:
                await websocket.send_json({
                    "type": "news_update",
                    "data": {"headlines": news}
                })
                logger.info(f"Sentinel news sent to new client: {len(news)} items")
            else:
                logger.debug("No sentinel news available for new client")
        except Exception as e:
            logger.warning(f"Failed to send initial sentinel news: {e}")

    async def send_market_data(self, websocket: WebSocket, state):
        """Fetch global indices and send to a newly connected client immediately.
        
        This ensures new clients get market data right away without waiting for background cycle.
        Similar pattern to send_news() - fetch data and send directly to new client.
        """
        try:
            from services.stock_discovery import StockDiscoveryService
            discovery_svc = StockDiscoveryService()
            
            # Fetch global indices (Nifty 50, India indices, VIX, US indices)
            global_ctx = await asyncio.to_thread(discovery_svc.fetch_global_indices)
            
            if global_ctx:
                # Send market data to the new client
                await websocket.send_json({
                    "type": "market_data",
                    "global_context": global_ctx
                })
                logger.info(f"Market data sent to new client: keys={list(global_ctx.keys())}")
            else:
                logger.debug("No market data available for new client")
        except Exception as e:
            logger.warning(f"Failed to send initial market data: {e}")

    async def broadcast_state(self, state):
        """Broadcast updated state to all connected clients."""
        for conn in self.active_connections:
            await self.send_state(conn, state)


async def handle_websocket(websocket: WebSocket, manager: ConnectionManager, state):
    """
    Main WebSocket command handler.
    Routes incoming commands to the appropriate service.
    """
    logger.info("New WebSocket connection attempt")
    try:
        await manager.connect(websocket)
        logger.info("WebSocket connected and accepted")
        await manager.send_state(websocket, state)
        logger.info("Initial state sent successfully")
        
        # Send sentinel news immediately to new client (not just broadcast)
        # This ensures new clients get news right away without waiting for background cycle
        await manager.send_news(websocket, state)
        
        # Fetch and send market data immediately to new client
        # This ensures new clients get market data (global indices) right away
        # without waiting for background engine to run
        await manager.send_market_data(websocket, state)

        while True:
            try:
                data = await websocket.receive_text()
                logger.debug(f"Received WebSocket command: {data[:100]}")
                command = json.loads(data)
                action = command.get("action")
            except Exception as e:
                logger.warning(f"Error receiving/parsing WebSocket data: {e}")
                break

            if action == "update_settings":
                try:
                    success = state.update_settings(
                        float(command.get('capital', state.capital)),
                        float(command.get('max_loss', state.max_loss_per_trade)),
                        command.get('search_engine', state.search_engine),
                        command.get('data_provider', state.data_provider),
                        command.get('fallback_data', getattr(state, 'fallback_data', True)),
                        command.get('fallback_search', getattr(state, 'fallback_search', True)),
                        command.get('fallback_ai', getattr(state, 'fallback_ai', True)),
                        command.get('ai_provider', getattr(state, 'ai_provider', 'google')),
                        command.get('ai_model', getattr(state, 'ai_model', 'gemini-3.1-pro'))
                    )
                    
                    if success:
                        logger.info(f"SKEPTIC: Settings saved - Capital: {state.capital}, Max Loss: {state.max_loss_per_trade}, AI: {state.ai_provider}/{state.ai_model}")

                        # Sync risk engine with new settings
                        risk_engine.update_config(
                            state.capital,
                            state.max_loss_per_trade,
                            getattr(state, 'max_daily_loss', 5000.0)
                        )
                        
                        # Send success notification to client
                        await websocket.send_json({
                            "type": "notification",
                            "message": f"✅ Settings saved to database successfully!",
                            "level": "success"
                        })
                        await manager.send_state(websocket, state)
                    else:
                        # Database commit failed
                        logger.error("SKEPTIC: Failed to save settings - database commit failed")
                        await websocket.send_json({
                            "type": "notification",
                            "message": f"⚠️ Failed to save settings: Database commit failed. Please try again.",
                            "level": "error"
                        })
                except Exception as e:
                    logger.error(f"SKEPTIC: Failed to save settings: {e}")
                    await websocket.send_json({
                        "type": "notification",
                        "message": f"⚠️ Failed to save settings: {str(e)[:100]}",
                        "level": "error"
                    })

            elif action == "list_models":
                provider = command.get("provider", "google")
                from services.ai_scorer import ai_advisor
                models = await asyncio.to_thread(ai_advisor.list_available_models, provider)
                await websocket.send_json({
                    "type": "model_list",
                    "provider": provider,
                    "data": models
                })

            elif action == "get_status":
                from services.technical_analysis import TechnicalAnalysisService
                ta_svc = TechnicalAnalysisService()
                status = ta_svc.get_connection_status(force_refresh=True)
                
                # Add rate limit status
                try:
                    from services.upstox_service import get_rate_limit_status
                    rate_limit = get_rate_limit_status()
                    status["rate_limit"] = rate_limit
                except ImportError:
                    pass
                
                logger.info(f"SKEPTIC: Connection status refreshed - Upstox: {status.get('upstox', {}).get('connected')}")
                await manager.broadcast({
                    "type": "connection_status",
                    "status": status
                })

            elif action == "get_trade_history":
                from database import SessionLocal
                from models import Trade
                from datetime import datetime, timedelta
                
                days = int(command.get("days", 90))
                cutoff = datetime.now() - timedelta(days=days)
                
                with SessionLocal() as db:
                    trades = db.query(Trade).filter(
                        Trade.status == "CLOSED",
                        Trade.close_time >= cutoff
                    ).order_by(Trade.close_time.desc()).all()
                    
                    history = []
                    for t in trades:
                        history.append({
                            "id": t.id,
                            "ticker": t.ticker,
                            "action": t.action,
                            "quantity": t.quantity,
                            "entry_price": t.entry_price,
                            "exit_price": t.exit_price,
                            "pnl": t.pnl,
                            "close_time": t.close_time.strftime("%Y-%m-%d %H:%M") if t.close_time else "Unknown"
                        })
                        
                await websocket.send_json({
                    "type": "trade_history_90d",
                    "data": history
                })

            elif action == "get_ai_history":
                from database import SessionLocal
                from models import AIInteraction
                with SessionLocal() as db:
                    scans = db.query(AIInteraction).filter(
                        AIInteraction.prompt_type == "SCAN"
                    ).order_by(AIInteraction.timestamp.desc()).limit(20).all()
                    
                    history = []
                    for s in scans:
                        history.append({
                            "id": s.id,
                            "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                            "model_used": s.model_used,
                            "output_json": s.output_json
                        })
                await websocket.send_json({
                    "type": "ai_history_update",
                    "data": history
                })

            elif action == "log_trade":
                # V2: Validate through Risk Engine before logging
                from services.technical_analysis import TechnicalAnalysisService
                ta_svc = TechnicalAnalysisService()

                ticker = command['ticker']
                trade_action = command['trade_action']
                entry_price = float(command.get('entry_price', 0))

                # Fetch live data for ATR and Trend verification
                df = ta_svc.fetch_ohlcv(ticker, period="5d", interval="5m")
                atr = risk_engine.compute_atr(df) if df is not None and not df.empty else 0
                ta_data = ta_svc.analyze_stock(ticker) # Get technical indicators
                vix_value = (state.global_context or {}).get("vix", {}).get("value", 0)

                # One-click trade: auto-use live price when entry_price is 0
                current_price = float(df['Close'].iloc[-1]) if df is not None and not df.empty else 0
                if entry_price <= 0:
                    entry_price = current_price
                
                validation = risk_engine.validate_trade(
                    entry_price, trade_action, atr, current_price, 
                    ta_data=ta_data, vix_value=vix_value
                )

                if not validation["passed"]:
                    await manager.broadcast({
                        "type": "notification",
                        "message": f"⚠️ Trade blocked: {'; '.join(validation['reasons'])}",
                        "level": "error"
                    })
                else:
                    # Use risk-engine computed levels (not AI-hallucinated ones)
                    levels = validation["levels"]
                    qty = validation["quantity"]

                    phase_ctx = market_phase_svc.get_phase_context()

                    trade = state.log_trade(
                        ticker, trade_action, qty,
                        entry_price, levels["stop_loss"],
                        levels["target_1"], levels["target_2"],
                        phase=phase_ctx["phase"],
                        atr=atr,
                        risk_per_share=levels["risk_per_share"]
                    )
                    await manager.broadcast({
                        "type": "notification",
                        "message": (
                            f"✅ Trade logged: {ticker} {trade_action} × {qty} @ ₹{entry_price:.2f} | "
                            f"SL: ₹{levels['stop_loss']:.2f} | T1: ₹{levels['target_1']:.2f} | "
                            f"Max Risk: ₹{validation['max_loss_this_trade']:.0f}"
                        ),
                        "level": "success"
                    })
                    await manager.broadcast({
                        "type": "trades_update",
                        "open_trades": state.open_trades,
                        "closed_trades": state.closed_trades
                    })

            elif action == "close_trade":
                exit_price = float(command['exit_price'])
                trade_id = str(command['trade_id'])

                # Find the trade to record P&L for daily tracking
                trade_pnl = 0
                for t in state.open_trades:
                    if str(t.get('id', '')) == trade_id:
                        if t['action'] == "BUY":
                            trade_pnl = (exit_price - t['entry_price']) * t['quantity']
                        else:
                            trade_pnl = (t['entry_price'] - exit_price) * t['quantity']
                        break

                state.close_trade(trade_id, exit_price)
                risk_engine.record_closed_pnl(trade_pnl)

                await manager.broadcast({
                    "type": "notification",
                    "message": f"Trade closed. P&L: ₹{trade_pnl:.2f}",
                    "level": "success" if trade_pnl >= 0 else "error"
                })
                await manager.broadcast({
                    "type": "trades_update",
                    "open_trades": state.open_trades,
                    "closed_trades": state.closed_trades
                })

            elif action == "trigger_scan":
                # V3 Skeptic Audit: Implement 60s cooldown to prevent quota exhaustion
                import time
                import os
                last_scan = getattr(state, '_last_manual_scan_time', 0)
                if time.time() - last_scan < 60 and os.getenv("SIMULATION", "false").lower() != "true":
                    await websocket.send_json({
                        "type": "notification",
                        "message": f"Scan cooldown active. Wait {int(60 - (time.time() - last_scan))}s.",
                        "level": "warning"
                    })
                    return
                
                # Check if rate limited
                try:
                    from services.upstox_service import is_rate_limited, get_rate_limit_status
                    if is_rate_limited():
                        status = get_rate_limit_status()
                        await websocket.send_json({
                            "type": "notification",
                            "message": f"⚠️ Rate limited. Wait {status.get('remaining_cooldown', 60)}s before scanning.",
                            "level": "warning"
                        })
                        return
                except ImportError:
                    pass
                
                state._last_manual_scan_time = time.time()
                # Enriched scan: TA → AI picks → enrich each pick with full data
                await manager.broadcast({
                    "type": "notification",
                    "message": "🔍 Scanning market... (this takes ~30s)",
                    "level": "info"
                })

                if os.getenv("SIMULATION", "false").lower() == "true":
                    await asyncio.sleep(1) # Delay for UI stabilization
                    mock_picks = [{
                        "ticker": "RELIANCE.NS",
                        "action": "BUY",
                        "current_price": 2500,
                        "signal": "STRONG BUY",
                        "confidence": 85,
                        "reasoning": "Simulation mode: Mocked trade candidate for RELIANCE.",
                        "live_price": 2500.0,
                        "technicals": {
                            "rsi_14": 45.0,
                            "macd_hist": 2.5,
                            "adx_14": 30.0,
                            "vwap": 2490.0,
                            "vol_surge": 1.5,
                            "ema_9": 2510.0,
                            "ema_21": 2480.0,
                            "lz_score": 0.8
                        },
                        "risk_levels": {"stop_loss": 2480.0, "target_1": 2540.0, "target_2": 2560.0, "quantity": 10},
                        "fundamentals": {"sector": "Energy", "market_cap": "17T"},
                        "sentiment": {"score": 60, "label": "Bullish", "headline_count": 5},
                        "atr": 10.0
                    }]
                    
                    import datetime
                    scan_entry = {"type": "SCAN", "result": mock_picks, "timestamp": datetime.datetime.now().strftime("%H:%M:%S")}
                    state.ai_advisor_message = scan_entry
                    state.ai_scans_today.insert(0, scan_entry)
                    state.ai_scans_today = state.ai_scans_today[:50]

                    await manager.broadcast({"type": "scan_results", "data": mock_picks})
                    continue

                try:
                    from services.technical_analysis import TechnicalAnalysisService
                    from services.stock_discovery import StockDiscoveryService
                    from services.ai_scorer import ai_advisor
                    from services.news_sentiment import NewsSentimentService

                    ta_svc = TechnicalAnalysisService()
                    discovery_svc = StockDiscoveryService()
                    news_svc = NewsSentimentService()
                    phase_ctx = market_phase_svc.get_phase_context()

                    provider = getattr(state, 'ai_provider', 'google')
                    model = getattr(state, 'ai_model', 'gemini-3.1-pro')
                    search_engine = getattr(state, 'search_engine', 'ddgs')
                    data_provider = getattr(state, 'data_provider', 'upstox')

                    logger.info(f"\n=====================================")
                    logger.info(f"AI Call (Manual Scan): SCAN")
                    logger.info(f"Data Source: {data_provider} | Search: {search_engine}")
                    logger.info(f"AI Engine: {provider} | Model: {model}")
                    logger.info(f"=====================================")

                    # Step 1: Get top candidates with TA
                    top_stocks = await asyncio.to_thread(
                        discovery_svc._get_top_candidates, 8
                    )

                    candidates = []
                    ta_cache = {}  # Cache TA data for enrichment later
                    for ticker in top_stocks[:8]:
                        try:
                            ta_data = await asyncio.to_thread(ta_svc.analyze_stock, ticker)
                            if ta_data:
                                df = await asyncio.to_thread(ta_svc.fetch_ohlcv, ticker, "5d", "5m")
                                atr = risk_engine.compute_atr(df) if df is not None else 0
                                
                                math_prob = ta_svc.evaluate_math_probability(ta_data)
                                ta_data['math_prob'] = math_prob
                                
                                log_msg = f"  [TA] {ticker} | Close: {ta_data.get('close'):.2f} | VWAP: {ta_data.get('vwap'):.2f} | EMA9: {ta_data.get('ema_9'):.2f} | EMA21: {ta_data.get('ema_21'):.2f} | RSI: {ta_data.get('rsi_14'):.2f} | ADX: {ta_data.get('adx_14'):.2f} | Surge: {ta_data.get('vol_surge'):.2f}x"
                                
                                ta_cache[ticker] = {"ta_data": ta_data, "atr": atr, "df": df}
                                
                                if math_prob >= 0.50:
                                    candidates.append({"ticker": ticker, "ta_data": ta_data, "atr": atr, "math_prob": math_prob})
                                    logger.info(log_msg + f" -> [MATH: PASS] (Score: {math_prob})")
                                else:
                                    logger.info(log_msg + f" -> [MATH: FAIL] (Score: {math_prob})")
                        except Exception as e:
                            logger.warning(f"Skipping {ticker}: {e}")

                    if not candidates:
                        logger.info("No stocks passed the mathematical setup pre-filter. Skipping AI scan to save API limits.")
                        ai_picks = []
                    else:
                        from services.quota_service import quota_svc
                        quota_check = quota_svc.check_quota(provider) # Use provider as quota key
                        if not quota_check.get("can_call", True):
                            await manager.broadcast({
                                "type": "notification",
                                "message": f"🚫 Quota exhausted for {provider}. Switch provider/model in Settings.",
                                "level": "error"
                            })
                            return

                        # Step 2: AI scan
                        logger.info(f"Passing {len(candidates)} mathematically validated Candidates to AI Scorer...")
                        ai_picks = await asyncio.to_thread(
                            ai_advisor.scan_market, candidates,
                            state.global_context, phase_ctx, provider, model,
                            ai_fallback=getattr(state, 'fallback_ai', True)
                        )

                    # Step 3: Enrich each AI pick with full data
                    enriched_picks = []
                    if isinstance(ai_picks, list):
                        for pick in ai_picks:
                            ticker = pick.get("ticker", "")
                            cached = ta_cache.get(ticker, {})
                            ta_data = cached.get("ta_data", {})
                            atr = cached.get("atr", 0)

                            # Risk Engine levels + Anti-Loss Validation
                            entry_price = ta_data.get("close", 0)
                            trade_action = pick.get("action", "BUY").upper().strip()
                            vix_value = state.global_context.get("vix", {}).get("value", 0)
                            
                            logger.info(f"DEBUG: Processing {ticker} | Action: {trade_action} | Price: {entry_price}")
                            
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

                            # News sentiment (lightweight — keyword only, no AI call)
                            sentiment = {"score": 50, "label": "Neutral", "headline_count": 0}
                            try:
                                search_engine = getattr(state, 'search_engine', 'ddgs')
                                headlines = await asyncio.to_thread(
                                    news_svc.fetch_news, ticker, search_engine
                                )
                                if headlines:
                                    sentiment["headline_count"] = len(headlines)
                                    # Quick keyword sentiment (no AI call needed)
                                    from services.news_sentiment import _keyword_sentiment
                                    kw_result = _keyword_sentiment(headlines)
                                    sentiment["score"] = kw_result.get("score", 50)
                                    sentiment["label"] = kw_result.get("label", "Neutral")
                            except Exception:
                                pass

                            # Signal classification from TA
                            ta_signal = ta_svc.classify_signal(ta_data)

                            # CONSENSUS CHECK: AI Pick must not contradict TA Signal
                            trade_action = pick.get("action", "BUY")
                            if trade_action == "BUY" and ("SHORT SELL" in ta_signal):
                                logger.warning(f"Consensus Check: Blocking AI BUY for {ticker} because TA is {ta_signal}")
                                continue
                            if trade_action == "SHORT SELL" and ("BUY" in ta_signal):
                                logger.warning(f"Consensus Check: Blocking AI SHORT SELL for {ticker} because TA is {ta_signal}")
                                continue

                            # If consensus passed, merge them into a single definitive signal
                            # We replace 'action' with the verified version for the UI
                            final_signal = f"VERIFIED {trade_action}" if ta_signal != "NEUTRAL" else trade_action
                            
                            entry_price = float(pick.get("current_price", 0))

                            # Update live price from latest bar in cached df
                            if not cached.get("df").empty:
                                entry_price = float(cached.get("df")['Close'].iloc[-1])
                                
                            try:
                                from services.upstox_service import upstox_client, get_instrument_key
                                if upstox_client and upstox_client.validate_token():
                                    ikey = get_instrument_key(ticker)
                                    if ikey:
                                        quote = upstox_client.fetch_market_quote(ikey)
                                        if quote and "data" in quote and ikey in quote["data"]:
                                            entry_price = float(quote["data"][ikey].get("last_price", entry_price))
                            except Exception as e:
                                logger.warning(f"SKEPTIC: Failed to fetch live quote for {ticker}: {e}")
                                
                            # Fallback if Upstox quote is unavailable
                            if entry_price == 0 and not cached.get("df").empty:
                                entry_price = float(cached.get("df")['Close'].iloc[-1])

                            # Handle NaNs
                            def _s(val, r=2):
                                try:
                                    v = round(val, r)
                                    import math
                                    if math.isnan(v): return 0
                                    return v
                                except: return 0
                                
                            enriched_picks.append({
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
                                    "lz_score": _s(cached.get("math_prob", 0), 2),
                                },
                                "lorentzian": {
                                    "score": _s(cached.get("math_prob", 0), 2),
                                    "signal": final_signal
                                },
                                "risk_levels": risk_levels,
                                "fundamentals": fundamentals,
                                "sentiment": sentiment,
                                "atr": _s(atr, 2),
                            })

                    # Update state for persistent view WITH ENRICHED DATA
                    import datetime
                    scan_entry = {"type": "SCAN", "result": enriched_picks, "timestamp": datetime.datetime.now().strftime("%H:%M:%S")}
                    state.ai_advisor_message = scan_entry
                    state.ai_scans_today.insert(0, scan_entry)
                    state.ai_scans_today = state.ai_scans_today[:50]
                    
                    # Update last successful data timestamp
                    state.last_data_update = datetime.datetime.now().isoformat()

                    await manager.broadcast({
                        "type": "scan_results",
                        "data": enriched_picks
                    })

                except Exception as e:
                    logger.error(f"Scan failed: {e}", exc_info=True)
                    await manager.broadcast({
                        "type": "notification",
                        "message": f"⚠️ Scan error: {str(e)[:100]}",
                        "level": "error"
                    })

            elif action == "run_backtest":
                logger.info(f"SKEPTIC: Received backtest request for {command.get('ticker')}")
                # Run backtester with user parameters
                await manager.broadcast({
                    "type": "notification",
                    "message": "🧪 Running backtest...",
                    "level": "info"
                })

                try:
                    from services.technical_analysis import TechnicalAnalysisService
                    from services.backtester import VectorizedBacktester

                    ta_svc = TechnicalAnalysisService()
                    ticker = command.get('ticker', 'RELIANCE.NS')
                    
                    # Handle either 'days' or 'period' (e.g. '30d')
                    period = command.get('period', '30d')
                    days = int(period.replace('d', '')) if isinstance(period, str) else command.get('days', 30)
                    
                    capital = command.get('capital', 100000)
                    params = command.get('params', {})

                    period_str = f"{days}d"
                    df = await asyncio.to_thread(ta_svc.fetch_ohlcv, ticker, period_str, "5m")

                    if df is None or df.empty:
                        raise ValueError(f"No data for {ticker}")

                    bt = VectorizedBacktester(df, initial_capital=capital, is_simulation=state.simulation_mode)
                    results = await asyncio.to_thread(bt.run_strategy, params)

                    # Clean trade_log for JSON serialization
                    if results.get("trade_log"):
                        for t in results["trade_log"]:
                            t["entry_time"] = str(t.get("entry_time", ""))
                            t["exit_time"] = str(t.get("exit_time", ""))

                    await manager.broadcast({
                        "type": "backtest_results",
                        "data": results
                    })

                except Exception as e:
                    logger.error(f"Backtest failed: {e}", exc_info=True)
                    await manager.broadcast({
                        "type": "backtest_results",
                        "data": {"error": str(e)}
                    })

            elif action == "ai_optimize":
                # AI-optimized strategy parameter tuning
                await manager.broadcast({
                    "type": "notification",
                    "message": "✨ AI optimization started... (this may take ~60s)",
                    "level": "info"
                })

                try:
                    from services.technical_analysis import TechnicalAnalysisService
                    from services.strategy_tuner import StrategyTuner

                    ta_svc = TechnicalAnalysisService()
                    ticker = command.get('ticker', 'RELIANCE.NS')
                    days = command.get('days', 30)
                    params = command.get('params', {})

                    period = f"{days}d"
                    df = await asyncio.to_thread(ta_svc.fetch_ohlcv, ticker, period, "5m")

                    if df is None or df.empty:
                        raise ValueError(f"No data for {ticker}")

                    provider = getattr(state, 'ai_provider', 'google')
                    model = getattr(state, 'ai_model', 'gemini-3.1-pro')

                    tuner = StrategyTuner()
                    result = await asyncio.to_thread(
                        tuner.optimize, ticker, df, params, 3, provider, model
                    )

                    await manager.broadcast({
                        "type": "backtest_results",
                        "data": result
                    })

                except Exception as e:
                    logger.error(f"AI optimize failed: {e}", exc_info=True)
                    await manager.broadcast({
                        "type": "backtest_results",
                        "data": {"error": str(e)}
                    })

            elif action == "truncate_test_data":
                from database import SessionLocal
                from models import Trade, AIInteraction, DailySummary
                
                try:
                    with SessionLocal() as db:
                        db.query(Trade).delete()
                        db.query(AIInteraction).delete()
                        db.query(DailySummary).delete()
                        
                        # Clear in-memory state as well
                        state.open_trades = []
                        state.closed_trades = []
                        state.ai_scans_today = []
                        state.action_timeline = []
                        state.ai_advisor_message = None
                        db.commit()
                        
                    await manager.broadcast({
                        "type": "notification",
                        "message": "🧹 Test data truncated successfully. State reset.",
                        "level": "success"
                    })
                    # Send updated state to all clients to clear UI
                    await manager.broadcast_state(state)
                except Exception as e:
                    logger.error(f"Truncation failed: {e}")
                    await websocket.send_json({
                        "type": "notification",
                        "message": "⚠️ Truncation failed.",
                        "level": "error"
                    })

            elif action == "get_chart_data":
                # Check rate limit before fetching chart data
                try:
                    from services.upstox_service import is_rate_limited, get_rate_limit_status
                    if is_rate_limited():
                        status = get_rate_limit_status()
                        await websocket.send_json({
                            "type": "notification",
                            "message": f"⏳ Rate limited. Using cached data. Retry in {status.get('remaining_cooldown', 60)}s.",
                            "level": "warning"
                        })
                except ImportError:
                    pass
                
                try:
                    from services.technical_analysis import TechnicalAnalysisService
                    from services.upstox_streamer import get_streamer
                    from services.upstox_service import get_instrument_key
                    
                    ta_svc = TechnicalAnalysisService()
                    ticker = command.get('ticker', 'RELIANCE.NS')
                    interval = command.get('interval', '5m')
                    params = command.get('params', {}) # Capture ML settings

                    payload = await asyncio.to_thread(
                        ta_svc.get_chart_payload, ticker, interval, params
                    )

                    # Add rate limit info to chart payload
                    try:
                        from services.upstox_service import get_rate_limit_status
                        rate_status = get_rate_limit_status()
                        payload["rate_limit_status"] = rate_status
                        payload["is_stale_data"] = rate_status.get("is_rate_limited", False)
                    except ImportError:
                        pass
                    
                    # Update last successful data timestamp
                    import datetime
                    state.last_data_update = datetime.datetime.now().isoformat()

                    await websocket.send_json({
                        "type": "chart_data",
                        "data": payload
                    })

                    inst_key = get_instrument_key(ticker)
                    if inst_key:
                        # Tick callback broadcasts to all connected clients
                        async def tick_callback(tick):
                            await manager.broadcast({
                                "type": "tick",
                                "data": tick
                            })

                        # Unsubscribe from previous if any (simplified)
                        get_streamer(tick_callback).subscribe([inst_key])

                except Exception as e:
                    logger.error(f"Chart data error: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "chart_data",
                        "data": {"error": str(e)}
                    })

            elif action == "refresh_data":
                """Manual refresh with rate limit awareness."""
                try:
                    from services.upstox_service import is_rate_limited, get_rate_limit_status
                    rate_status = get_rate_limit_status()
                    
                    if rate_status.get("is_rate_limited"):
                        remaining = rate_status.get("remaining_cooldown", 60)
                        await websocket.send_json({
                            "type": "notification",
                            "message": f"⚠️ Rate limited. Please wait {remaining}s before refreshing.",
                            "level": "warning"
                        })
                        # Send current state anyway to show data
                        await manager.send_state(websocket, state)
                    else:
                        # Proceed with refresh
                        await manager.send_state(websocket, state)
                        await websocket.send_json({
                            "type": "notification",
                            "message": "✅ Data refreshed successfully",
                            "level": "success"
                        })
                except ImportError:
                    await manager.send_state(websocket, state)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
