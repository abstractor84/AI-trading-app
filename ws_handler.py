"""
WebSocket Handler (V2)
Handles all WebSocket connections and command routing.
Extracted from the monolithic main.py for clean separation of concerns.
"""
import json
import asyncio
import logging
import pandas as pd
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

from services.risk_engine import risk_engine
from services.market_phase import market_phase_svc

logger = logging.getLogger(__name__)





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
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception:
                dead.append(conn)
        for d in dead:
            self.disconnect(d)

    async def send_state(self, websocket: WebSocket, state):
        """Send full initial state to a newly connected client."""
        phase_ctx = market_phase_svc.get_phase_context()
        from services.quota_service import quota_svc
        
        payload = {
            "type": "state_update",
            "capital": state.capital,
            "max_loss": state.max_loss_per_trade,
            "open_trades": state.open_trades,
            "closed_trades": state.closed_trades,
            "global_context": getattr(state, 'global_context', {}),
            "market_phase": phase_ctx,
            "ai_calls_today": quota_svc.get_total_daily_usage(),
            "ai_calls_limit": 20,
            "ai_advisor": getattr(state, 'ai_advisor_message', None),
            "ai_scans_today": getattr(state, 'ai_scans_today', []),
            "connection_status": getattr(state, 'connection_status', {}),
            "action_timeline": getattr(state, 'action_timeline', [])[-20:],
            "search_engine": getattr(state, 'search_engine', 'tavily'),
            "data_provider": getattr(state, 'data_provider', 'upstox'),
            "fallback_data": getattr(state, 'fallback_data', True),
            "fallback_search": getattr(state, 'fallback_search', True),
            "fallback_ai": getattr(state, 'fallback_ai', True),
            "ai_provider": getattr(state, 'ai_provider', 'google'),
            "ai_model": getattr(state, 'ai_model', 'gemini-3.1-pro'),
        }
        await websocket.send_json(payload)


async def handle_websocket(websocket: WebSocket, manager: ConnectionManager, state):
    """
    Main WebSocket command handler.
    Routes incoming commands to the appropriate service.
    """
    await manager.connect(websocket)
    await manager.send_state(websocket, state)

    try:
        while True:
            data = await websocket.receive_text()
            command = json.loads(data)
            action = command.get("action")

            if action == "update_settings":
                state.update_settings(
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

                # Sync risk engine with new settings
                risk_engine.update_config(
                    state.capital,
                    state.max_loss_per_trade,
                    getattr(state, 'max_daily_loss', 5000.0)
                )
                await manager.send_state(websocket, state)

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
                status = ta_svc.get_connection_status()
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
                vix_value = state.global_context.get("vix", {}).get("value", 0)

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
                trade_id = command['trade_id']

                # Find the trade to record P&L for daily tracking
                trade_pnl = 0
                for t in state.open_trades:
                    if t['id'] == trade_id:
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
                last_scan = getattr(state, '_last_manual_scan_time', 0)
                if time.time() - last_scan < 60:
                    await websocket.send_json({
                        "type": "notification",
                        "message": f"Scan cooldown active. Wait {int(60 - (time.time() - last_scan))}s.",
                        "level": "warning"
                    })
                    return
                
                state._last_manual_scan_time = time.time()
                # Enriched scan: TA → AI picks → enrich each pick with full data
                await manager.broadcast({
                    "type": "notification",
                    "message": "🔍 Scanning market... (this takes ~30s)",
                    "level": "info"
                })

                import os
                if os.getenv("SIMULATION", "false").lower() == "true":
                    mock_picks = [{
                        "ticker": "RELIANCE.NS",
                        "action": "BUY",
                        "current_price": 2500,
                        "signal": "STRONG BUY",
                        "confidence": 85,
                        "reasoning": "Simulation mode: Mocked trade candidate for RELIANCE.",
                        "live_price": 2500.0,
                        "ta_data": {
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
                    data_provider = getattr(state, 'data_provider', 'yfinance')

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
                        # Step 2: AI scan
                        logger.info(f"Passing {len(candidates)} mathematically validated Candidates to AI Scorer...")
                        ai_picks = await asyncio.to_thread(
                            ai_advisor.scan_market, candidates,
                            state.global_context, phase_ctx, provider, model,
                            ai_fallback=getattr(state, 'fallback_ai', True)
                        )

                        # Update state for persistent view
                        import datetime
                        scan_entry = {"type": "SCAN", "result": ai_picks, "timestamp": datetime.datetime.now().strftime("%H:%M:%S")}
                        state.ai_advisor_message = scan_entry
                        state.ai_scans_today.insert(0, scan_entry)
                        state.ai_scans_today = state.ai_scans_today[:50]

                        await manager.broadcast({
                            "type": "scan_results",
                            "data": ai_picks
                        })
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
                                from services.upstox_service import upstox_client
                                if upstox_client and upstox_client.validate_token():
                                    ikey = upstox_client.get_instrument_key(ticker)
                                    if ikey:
                                        quote = upstox_client.fetch_market_quote(ikey)
                                        if quote and quote.get("ltp"):
                                            entry_price = float(quote.get("ltp"))
                            except Exception as e:
                                logger.warning(f"Failed to fetch live quote for {ticker}: {e}")
                                
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

            elif action == "get_chart_data":
                try:
                    from services.technical_analysis import TechnicalAnalysisService
                    from services.upstox_streamer import get_streamer
                    from services.upstox_service import get_instrument_key
                    
                    ta_svc = TechnicalAnalysisService()
                    ticker = command.get('ticker', 'RELIANCE.NS')
                    interval = command.get('interval', '5m')

                    payload = await asyncio.to_thread(
                        ta_svc.get_chart_payload, ticker, interval
                    )

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

    except WebSocketDisconnect:
        manager.disconnect(websocket)
