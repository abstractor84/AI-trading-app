"""
V2 AI Trading Companion — Main Application
============================================
Slim entrypoint that wires together all V2 modules.
Previous 875-line monolith decomposed into:
  - background_engine.py   → Phase-aware market loop
  - ws_handler.py           → WebSocket connections & commands
  - services/state.py       → App state & DB persistence
  - services/market_phase.py → Market session state machine
  - services/risk_engine.py  → Deterministic risk math
"""
import os
import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Query, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

# Load environment before any service imports
_DOT_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_DOT_ENV_PATH)

from services.state import AppState
from ws_handler import ConnectionManager, handle_websocket
from background_engine import BackgroundEngine
from database import SessionLocal
from models import Watchlist, WatchlistStock, DashboardStock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Singletons ─────────────────────────────────────────────────────
state = AppState()
manager = ConnectionManager()
bg_engine = BackgroundEngine(state, manager)

# Ensure static directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)


# ─── App Lifecycle ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("V2 Background Engine starting")
    task = asyncio.create_task(bg_engine.run())
    yield
    task.cancel()

app = FastAPI(title="AI Trading Companion V2", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── Upstox OAuth Helpers ──────────────────────────────────────────
async def _perform_upstox_exchange(code: str):
    """Internal helper to exchange code for token and save to .env."""
    import requests as req
    api_key = os.getenv("UPSTOX_API_KEY")
    api_secret = os.getenv("UPSTOX_API_SECRET")
    redirect_uri = "http://localhost:8000" # Simplified redirect

    try:
        resp = req.post("https://api.upstox.com/v2/login/authorization/token", data={
            "code": code,
            "client_id": api_key,
            "client_secret": api_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        })
        token_data = resp.json()
        access_token = token_data.get("access_token")

        if not access_token:
            return HTMLResponse(f"<h2>❌ Token exchange failed</h2><pre>{resp.text}</pre>", status_code=400)

        # Write token to .env
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        lines = []
        found = False
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("UPSTOX_ACCESS_TOKEN"):
                lines[i] = f'UPSTOX_ACCESS_TOKEN="{access_token}"\n'
                found = True
        if not found:
            lines.append(f'UPSTOX_ACCESS_TOKEN="{access_token}"\n')
        with open(env_path, "w") as f:
            f.writelines(lines)
        
        # IMPORTANT: Update current process environment so services see it immediately
        os.environ["UPSTOX_ACCESS_TOKEN"] = access_token

        # Reload tokens across services
        from services.technical_analysis import _upstox_svc
        _upstox_svc.reload_token()

        logger.info("Upstox token refreshed successfully")
        return HTMLResponse(
            "<h2>✅ Upstox connected!</h2><p>Token saved. This window will close.</p>"
            "<script>setTimeout(()=>window.close(),2000)</script>"
        )
    except Exception as e:
        logger.error(f"Upstox OAuth error: {e}")
        return HTMLResponse(f"<h2>❌ Error</h2><pre>{e}</pre>", status_code=500)


@app.get("/upstox/callback")
async def upstox_callback(code: str = Query(None)):
    """Legacy endpoint for /upstox/callback redirection."""
    if not code:
        return HTMLResponse("<h2>❌ No auth code received</h2>", status_code=400)
    return await _perform_upstox_exchange(code)


@app.get("/upstox/connect")
async def upstox_connect():
    """Redirect to Upstox login page using the registered localhost:8000 URI."""
    api_key = os.getenv("UPSTOX_API_KEY")
    redirect_uri = "http://localhost:8000"
    url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={api_key}&redirect_uri={redirect_uri}"
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=url)


# ─── Pages ──────────────────────────────────────────────────────────
@app.get("/")
async def root(code: str = Query(None)):
    """Main dashboard or handle Upstox OAuth redirect at root."""
    if code:
        return await _perform_upstox_exchange(code)
    return FileResponse("static/index.html")


# ─── WebSocket ──────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket, manager, state)


# ─── REST API: Watchlists ──────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/api/watchlists")
async def get_watchlists(db: Session = Depends(get_db)):
    return db.query(Watchlist).all()


@app.post("/api/watchlists")
async def create_watchlist(name: str = Query(...), db: Session = Depends(get_db)):
    wl = Watchlist(name=name)
    db.add(wl)
    db.commit()
    db.refresh(wl)
    return wl


@app.delete("/api/watchlists/{wl_id}")
async def delete_watchlist(wl_id: int, db: Session = Depends(get_db)):
    db.query(WatchlistStock).filter(WatchlistStock.watchlist_id == wl_id).delete()
    db.query(Watchlist).filter(Watchlist.id == wl_id).delete()
    db.commit()
    return {"status": "deleted"}


@app.get("/api/watchlists/{wl_id}/stocks")
async def get_watchlist_stocks(wl_id: int, db: Session = Depends(get_db)):
    return db.query(WatchlistStock).filter(WatchlistStock.watchlist_id == wl_id).all()


@app.post("/api/watchlists/{wl_id}/stocks")
async def add_watchlist_stock(wl_id: int, ticker: str = Query(...),
                               db: Session = Depends(get_db)):
    stock = WatchlistStock(watchlist_id=wl_id, ticker=ticker)
    db.add(stock)
    db.commit()
    return stock


@app.delete("/api/watchlists/{wl_id}/stocks/{ticker}")
async def remove_watchlist_stock(wl_id: int, ticker: str,
                                  db: Session = Depends(get_db)):
    db.query(WatchlistStock).filter(
        WatchlistStock.watchlist_id == wl_id,
        WatchlistStock.ticker == ticker
    ).delete()
    db.commit()
    return {"status": "removed"}


# ─── REST API: Dashboard Stocks ────────────────────────────────────
@app.get("/api/dashboard/stocks")
async def get_dashboard_stocks():
    return list(state.dashboard_watch_stocks)


@app.post("/api/dashboard/stocks")
async def add_dashboard_stock(ticker: str = Query(...)):
    state.add_dashboard_stock(ticker)
    return {"status": "added", "ticker": ticker}


@app.delete("/api/dashboard/stocks/{ticker}")
async def remove_dashboard_stock(ticker: str):
    state.remove_dashboard_stock(ticker)
    return {"status": "removed", "ticker": ticker}


# ─── Entrypoint ─────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
