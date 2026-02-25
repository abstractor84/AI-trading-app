import asyncio
import json
import logging
from datetime import datetime, time as dtime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from services.stock_discovery import StockDiscoveryService
from services.technical_analysis import TechnicalAnalysisService
from services.news_sentiment import NewsSentimentService
from services.ai_scorer import AIScorerService
from services.state import AppState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

state = AppState()
discovery_svc = StockDiscoveryService()
ta_svc = TechnicalAnalysisService()
news_svc = NewsSentimentService()
ai_svc = AIScorerService()

# Ensure static dir exists
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
        # Send initial state
        await self.send_state(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        for dc in dead_connections:
            self.disconnect(dc)

    async def send_state(self, websocket: WebSocket):
        payload = {
            "type": "state_update",
            "capital": state.capital,
            "max_loss": state.max_loss_per_trade,
            "open_trades": state.open_trades,
            "closed_trades": state.closed_trades,
            "ai_signals": state.ai_signals,
            "global_context": state.global_context
        }
        await websocket.send_json(payload)

manager = ConnectionManager()

def get_market_session():
    now = datetime.now().time()
    if now < dtime(9, 15): return "Pre-Market"
    elif now < dtime(11, 0): return "Morning (Aggressive)"
    elif now < dtime(13, 0): return "Midday (Low Volatility)"
    elif now < dtime(15, 0): return "Afternoon (Tighten SL)"
    elif now < dtime(15, 15): return "EXIT ZONE"
    else: return "Post-Market"

async def perform_ai_screening():
    logger.info("Starting AI Screening...")
    await manager.broadcast({"type": "notification", "message": "Starting NSE Market Scan...", "level": "info"})
    
    # 1. Fetch globals
    state.global_context = discovery_svc.fetch_global_indices()
    
    # 2. Discover & rank
    top_candidates = discovery_svc._get_top_candidates(limit=12)
    state.selected_candidates = top_candidates
    await manager.broadcast({"type": "notification", "message": f"Found {len(top_candidates)} promising candidates. Generating signals...", "level": "info"})
    
    # 3. Analyze all
    signals = []
    for ticker in top_candidates:
        ta_data = ta_svc.analyze_stock(ticker)
        if not ta_data:
            continue
            
        headlines = news_svc.fetch_news(ticker)
        sentiment = news_svc.score_sentiment(headlines)
        
        # Merge global context into a simpler dict for the prompt to save tokens
        simple_global = {k: v.get('value') for k, v in state.global_context.items() if isinstance(v, dict)}
        
        # Time-aware restriction
        session = get_market_session()
        if session in ["EXIT ZONE", "Post-Market"]:
            rec = {
                "action": "AVOID", "confidence_score": 0, "ai_composite_score": 0,
                "explanation": f"Market session is {session}. No new trades recommended."
            }
        else:
            rec = ai_svc.generate_recommendation(ticker, ta_data, sentiment, simple_global, state.capital, state.max_loss_per_trade)
        
        signal = {
            "ticker": ticker,
            "ta_data": ta_data,
            "headlines": headlines,
            "sentiment": sentiment,
            "ai_recommendation": rec,
            "last_updated": datetime.now().strftime("%H:%M:%S")
        }
        signals.append(signal)

    # Sort by AI score
    signals.sort(key=lambda x: x["ai_recommendation"].get("ai_composite_score", 0), reverse=True)
    state.ai_signals = signals
    
    await manager.broadcast({
        "type": "state_update",
        "capital": state.capital,
        "max_loss": state.max_loss_per_trade,
        "open_trades": state.open_trades,
        "closed_trades": state.closed_trades,
        "ai_signals": state.ai_signals,
        "global_context": state.global_context
    })
    await manager.broadcast({"type": "notification", "message": "AI Screening completed successfully.", "level": "success"})

async def background_market_loop():
    while True:
        try:
            state.check_daily_reset()
            # If no signals yet, we don't automatically screen because it takes time and uses API limits
            # We just update global markets and open trades prices.
            now = datetime.now()
            
            # Fetch globals every minute
            state.global_context = discovery_svc.fetch_global_indices()
            
            # Update open trade P&L
            # For simplicity, we can fetch real-time close for open trades via yfinance fast
            if state.open_trades:
                open_tickers = [t['ticker'] for t in state.open_trades]
                if open_tickers:
                    prices = discovery_svc.fetch_global_indices() # We need a quick price fetcher
                    # Actually, we can use yf.download
                    df = getattr(ta_svc, 'fetch_ohlcv')(open_tickers[0], period="1d", interval="1m") if len(open_tickers)==1 else None
                    if len(open_tickers) == 1 and df is not None and not df.empty:
                        current_price = float(df['Close'].iloc[-1])
                        trade = state.open_trades[0]
                        trade['current_price'] = current_price
                        pnl = (current_price - trade['entry_price']) * trade['quantity'] if trade['action'] == "BUY" else (trade['entry_price'] - current_price) * trade['quantity']
                        trade['pnl'] = round(pnl, 2)
                        
            # If the market is open and we have ai_signals, maybe refresh the TA of those 12 every minute?
            # Doing full Gemini might be rate limited. Thus only updating TA and keeping AI same.
            
            msg = {
                "type": "heartbeat",
                "time": now.strftime("%H:%M:%S"),
                "session": get_market_session(),
                "global_context": state.global_context, # Now a dict of HTML strings {"global": x, "india": y}
                "open_trades": state.open_trades
            }
            await manager.broadcast(msg)
            
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in background loop: {e}")
            await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting background loop")
    task = asyncio.create_task(background_market_loop())
    yield
    # Shutdown
    task.cancel()

app = FastAPI(title="Gemini NSE Trader", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api/chart/{ticker}")
async def get_chart_data(ticker: str):
    """Serve OHLCV and technical indicator array data for TradingView Lightweight charts."""
    df = getattr(ta_svc, 'fetch_ohlcv')(f"{ticker}.NS", period="5d", interval="5m")
    if df is None or df.empty:
        return {"error": "No data available"}
    
    # Compute full series indicators
    import pandas_ta as ta
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    
    # Calculate daily VWAP for the whole series
    df['Date'] = df.index.date
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']
    df['Cumulative_VP'] = df.groupby('Date')['VP'].cumsum()
    df['Cumulative_Vol'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_VP'] / df['Cumulative_Vol']
    
    chart_data = []
    ema9_data = []
    ema21_data = []
    vwap_data = []
    
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    for index, row in df.iterrows():
        # Yahoo finance sometimes returns naive datetimes. Force it to be aware, then convert to UTC timestamp.
        if index.tzinfo is None:
            index = index.tz_localize('Asia/Kolkata')
        unix_time = int(index.timestamp())
        
        if chart_data and chart_data[-1]['time'] == unix_time:
            continue
            
        chart_data.append({
            "time": unix_time,
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close'])
        })
        
        # Add indicator points if valid
        if not pd.isna(row.get('EMA_9')):
            ema9_data.append({"time": unix_time, "value": float(row['EMA_9'])})
        if not pd.isna(row.get('EMA_21')):
            ema21_data.append({"time": unix_time, "value": float(row['EMA_21'])})
        if not pd.isna(row['VWAP']):
            vwap_data.append({"time": unix_time, "value": float(row['VWAP'])})

    return {
        "data": chart_data,
        "ema9": ema9_data,
        "ema21": ema21_data,
        "vwap": vwap_data
    }

@app.post("/api/backtest/{ticker}")
async def run_backtest_tuner(ticker: str):
    """Run the AI Strategy Tuner for a given ticker."""
    # Fetch 60 days of 5-min data for backtest scope
    df = getattr(ta_svc, 'fetch_ohlcv')(f"{ticker}.NS", period="60d", interval="5m")
    if df is None or df.empty:
        return {"error": "No historical data available for backtesting"}
        
    from services.strategy_tuner import StrategyTuner
    
    initial_params = {
        "ema_fast": 9,
        "ema_slow": 21,
        "rsi_len": 14,
        "rsi_buy_threshold": 40,
        "rsi_short_threshold": 60,
        "sl_pct": 0.01,
        "tp_pct": 0.02
    }
    
    try:
        tuner = StrategyTuner()
        results = tuner.optimize(ticker, df, initial_params, iterations=2) # Keep iterations low for API speed
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"Backtest Endpoint Error: {e}")
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            command = json.loads(data)
            action = command.get("action")
            
            if action == "trigger_scan":
                asyncio.create_task(perform_ai_screening())
            elif action == "update_settings":
                state.update_settings(float(command.get('capital')), float(command.get('max_loss')))
                await manager.send_state(websocket)
            elif action == "log_trade":
                trade = state.log_trade(
                    command['ticker'], command['trade_action'], int(command['qty']),
                    float(command['entry_price']), float(command['sl']),
                    float(command['t1']), float(command['t2'])
                )
                await manager.broadcast({"type": "notification", "message": f"Trade logged for {trade['ticker']}", "level": "success"})
                await manager.broadcast({"type": "trades_update", "open_trades": state.open_trades})
            elif action == "close_trade":
                state.close_trade(command['trade_id'], float(command['exit_price']))
                await manager.broadcast({"type": "notification", "message": "Trade closed.", "level": "info"})
                await manager.broadcast({"type": "trades_update", "open_trades": state.open_trades, "closed_trades": state.closed_trades})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
