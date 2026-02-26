"""
WebSocket Handler (V2)
Handles all WebSocket connections and command routing.
Extracted from the monolithic main.py for clean separation of concerns.
"""
import json
import asyncio
import logging
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

from services.risk_engine import risk_engine
from services.market_phase import market_phase_svc

logger = logging.getLogger(__name__)


def _classify_signal(ta_data: dict) -> str:
    """Classify TA data into a composite signal: STRONG BUY/BUY/NEUTRAL/SELL/STRONG SELL."""
    score = 0
    rsi = ta_data.get("rsi_14", 50)
    macd_hist = ta_data.get("macd_hist", 0)
    adx = ta_data.get("adx_14", 0)
    vol_surge = ta_data.get("vol_surge", 1)
    ema_9 = ta_data.get("ema_9", 0)
    ema_21 = ta_data.get("ema_21", 0)
    close = ta_data.get("close", 0)
    vwap = ta_data.get("vwap", 0)

    # RSI contribution
    if rsi < 35: score += 2
    elif rsi < 45: score += 1
    elif rsi > 65: score -= 2
    elif rsi > 55: score -= 1

    # MACD histogram
    if macd_hist > 1: score += 2
    elif macd_hist > 0: score += 1
    elif macd_hist < -1: score -= 2
    elif macd_hist < 0: score -= 1

    # Trend strength (ADX)
    if adx > 25: score += 1 if macd_hist > 0 else -1

    # Volume confirmation
    if vol_surge > 1.5: score += 1 if macd_hist > 0 else -1

    # EMA crossover
    if ema_9 > ema_21: score += 1
    elif ema_9 < ema_21: score -= 1

    # VWAP position
    if close > vwap: score += 1
    elif close < vwap: score -= 1

    if score >= 4: return "STRONG BUY"
    if score >= 2: return "BUY"
    if score <= -4: return "STRONG SELL"
    if score <= -2: return "SELL"
    return "NEUTRAL"


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
        payload = {
            "type": "state_update",
            "capital": state.capital,
            "max_loss": state.max_loss_per_trade,
            "open_trades": state.open_trades,
            "closed_trades": state.closed_trades,
            "global_context": state.global_context,
            "market_phase": phase_ctx,
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
                    command.get('search_fallback', state.search_fallback),
                    command.get('auto_refresh', getattr(state, 'auto_refresh', True)),
                    command.get('ai_provider', getattr(state, 'ai_provider', 'google')),
                    command.get('ai_model', getattr(state, 'ai_model', 'gemini-2.5-flash'))
                )
                # Sync risk engine with new settings
                risk_engine.update_config(
                    state.capital,
                    state.max_loss_per_trade,
                    getattr(state, 'max_daily_loss', 5000.0)
                )
                await manager.send_state(websocket, state)

            elif action == "log_trade":
                # V2: Validate through Risk Engine before logging
                from services.technical_analysis import TechnicalAnalysisService
                ta_svc = TechnicalAnalysisService()

                ticker = command['ticker']
                trade_action = command['trade_action']
                entry_price = float(command.get('entry_price', 0))

                # Fetch live data for ATR
                df = ta_svc.fetch_ohlcv(ticker, period="5d", interval="5m")
                atr = risk_engine.compute_atr(df) if df is not None and not df.empty else 0

                # One-click trade: auto-use live price when entry_price is 0
                current_price = float(df['Close'].iloc[-1]) if df is not None and not df.empty else 0
                if entry_price <= 0:
                    entry_price = current_price
                validation = risk_engine.validate_trade(entry_price, trade_action, atr, current_price)

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
                # Enriched scan: TA → AI picks → enrich each pick with full data
                await manager.broadcast({
                    "type": "notification",
                    "message": "🔍 Scanning market... (this takes ~30s)",
                    "level": "info"
                })

                try:
                    from services.technical_analysis import TechnicalAnalysisService
                    from services.stock_discovery import StockDiscoveryService
                    from services.ai_scorer import ai_advisor
                    from services.news_sentiment import NewsSentimentService

                    ta_svc = TechnicalAnalysisService()
                    discovery_svc = StockDiscoveryService()
                    news_svc = NewsSentimentService()
                    phase_ctx = market_phase_svc.get_phase_context()

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
                                candidates.append({"ticker": ticker, "ta_data": ta_data, "atr": atr})
                                ta_cache[ticker] = {"ta_data": ta_data, "atr": atr, "df": df}
                        except Exception as e:
                            logger.warning(f"Skipping {ticker}: {e}")

                    # Step 2: AI scan
                    provider = getattr(state, 'ai_provider', 'google')
                    model = getattr(state, 'ai_model', 'gemini-2.5-flash')

                    ai_picks = await asyncio.to_thread(
                        ai_advisor.scan_market, candidates,
                        state.global_context, phase_ctx, provider, model
                    )

                    # Step 3: Enrich each AI pick with full data
                    enriched_picks = []
                    if isinstance(ai_picks, list):
                        for pick in ai_picks:
                            ticker = pick.get("ticker", "")
                            cached = ta_cache.get(ticker, {})
                            ta_data = cached.get("ta_data", {})
                            atr = cached.get("atr", 0)

                            # Risk Engine levels
                            entry_price = ta_data.get("close", 0)
                            trade_action = pick.get("action", "BUY")
                            risk_levels = {}
                            try:
                                validation = risk_engine.validate_trade(
                                    entry_price, trade_action, atr, entry_price
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
                            signal = _classify_signal(ta_data)

                            enriched_picks.append({
                                **pick,
                                "live_price": round(entry_price, 2),
                                "technicals": {
                                    "rsi_14": round(ta_data.get("rsi_14", 0), 1),
                                    "macd_hist": round(ta_data.get("macd_hist", 0), 2),
                                    "adx_14": round(ta_data.get("adx_14", 0), 1),
                                    "vwap": round(ta_data.get("vwap", 0), 2),
                                    "vol_surge": round(ta_data.get("vol_surge", 0), 1),
                                    "bb_upper": round(ta_data.get("bb_upper", 0), 2),
                                    "bb_lower": round(ta_data.get("bb_lower", 0), 2),
                                    "ema_9": round(ta_data.get("ema_9", 0), 2),
                                    "ema_21": round(ta_data.get("ema_21", 0), 2),
                                },
                                "risk_levels": risk_levels,
                                "fundamentals": fundamentals,
                                "sentiment": sentiment,
                                "signal": signal,
                                "atr": round(atr, 2),
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
                    days = command.get('days', 30)
                    capital = command.get('capital', 100000)
                    params = command.get('params', {})

                    period = f"{days}d"
                    df = await asyncio.to_thread(ta_svc.fetch_ohlcv, ticker, period, "5m")

                    if df is None or df.empty:
                        raise ValueError(f"No data for {ticker}")

                    bt = VectorizedBacktester(df, initial_capital=capital)
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
                    model = getattr(state, 'ai_model', 'gemini-2.5-flash')

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
                # Fetch OHLCV + generate projection
                try:
                    from services.technical_analysis import TechnicalAnalysisService
                    from services.price_projector import price_projector

                    ta_svc = TechnicalAnalysisService()
                    ticker = command.get('ticker', '')

                    df = await asyncio.to_thread(
                        ta_svc.fetch_ohlcv, ticker, "1d", "1m"
                    )

                    if df is None or df.empty:
                        # Fallback to 5-min data
                        df = await asyncio.to_thread(
                            ta_svc.fetch_ohlcv, ticker, "5d", "5m"
                        )

                    result = await asyncio.to_thread(
                        price_projector.generate_projection, df, 1
                    )

                    await websocket.send_json({
                        "type": "chart_data",
                        "data": result
                    })

                except Exception as e:
                    logger.error(f"Chart data error: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "chart_data",
                        "data": {"error": str(e)}
                    })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
