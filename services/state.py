import json
import logging
import uuid
from datetime import datetime

from database import SessionLocal
from models import Trade, StockDetail

logger = logging.getLogger(__name__)

class AppState:
    def __init__(self):
        self.capital = 100000.0  # Initial capital
        self.max_loss_per_trade = 1000.0 # Standard risk per trade
        
        # Load from DB on init
        self._load_from_db()

        self.search_engine = 'gemini'
        self.data_provider = 'yfinance'
        self.search_fallback = False  # If True, cascade to other engines on failure
        
        self.ai_signals = []
        self.global_context = {}
        self.dashboard_watch_stocks = set()
        
        self.last_reset_date = datetime.now().date()
        
    def _load_from_db(self):
        with SessionLocal() as db:
            open_db_trades = db.query(Trade).filter(Trade.status == "OPEN").all()
            self.open_trades = [self._to_dict(t) for t in open_db_trades]
            
            # Load today's closed trades only to save memory
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            closed_db_trades = db.query(Trade).filter(Trade.status == "CLOSED", Trade.timestamp >= today_start).all()
            self.closed_trades = [self._to_dict(t) for t in closed_db_trades]

    def _to_dict(self, trade: Trade):
        return {
            "id": trade.id,
            "ticker": trade.ticker,
            "action": trade.action,
            "quantity": trade.quantity,
            "entry_price": trade.entry_price,
            "stop_loss": trade.stop_loss,
            "target_1": trade.target_1,
            "target_2": trade.target_2,
            "exit_price": trade.exit_price,
            "pnl": trade.pnl,
            "status": trade.status,
            "timestamp": trade.timestamp.isoformat() if trade.timestamp else None,
            "close_time": trade.close_time.isoformat() if trade.close_time else None
        }
        
    def check_daily_reset(self):
        """Reset state if it's a new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            logger.info("Midnight passed, resetting daily trade state...")
            # We don't delete from DB, just clear active memory cache of closed trades
            self.open_trades.clear()
            self.closed_trades.clear()
            self.ai_signals.clear()
            self._load_from_db() # Refetch any open ones that carried over somehow
            self.last_reset_date = current_date
            
    def update_settings(self, capital: float, max_loss: float, search_engine: str = 'gemini', data_provider: str = 'yfinance', search_fallback: bool = False):
        self.capital = capital
        self.max_loss_per_trade = max_loss
        self.search_engine = search_engine
        self.data_provider = data_provider
        self.search_fallback = search_fallback
        
    def log_trade(self, ticker: str, action: str, qty: int, entry_price: float, sl: float, t1: float, t2: float):
        trade_id = str(uuid.uuid4())
        new_trade = Trade(
            id=trade_id, ticker=ticker, action=action, quantity=qty,
            entry_price=entry_price, stop_loss=sl, target_1=t1, target_2=t2,
            status="OPEN"
        )
        with SessionLocal() as db:
            db.add(new_trade)
            db.commit()
            db.refresh(new_trade)
            
        trade_dict = self._to_dict(new_trade)
        self.open_trades.append(trade_dict)
        return trade_dict

    def close_trade(self, trade_id: str, exit_price: float):
        with SessionLocal() as db:
            db_trade = db.query(Trade).filter(Trade.id == trade_id).first()
            if db_trade and db_trade.status == "OPEN":
                db_trade.status = "CLOSED"
                db_trade.exit_price = exit_price
                db_trade.close_time = datetime.utcnow()
                
                if db_trade.action == "BUY":
                    db_trade.pnl = (exit_price - db_trade.entry_price) * db_trade.quantity
                else:
                    db_trade.pnl = (db_trade.entry_price - exit_price) * db_trade.quantity
                
                db.commit()
                db.refresh(db_trade)
                
                # Update memory
                tdict = self._to_dict(db_trade)
                # Remove from open, add to closed
                self.open_trades = [t for t in self.open_trades if t['id'] != trade_id]
                self.closed_trades.append(tdict)
                return tdict
        return None

    def add_dashboard_stock(self, ticker: str):
        self.dashboard_watch_stocks.add(ticker)

    def remove_dashboard_stock(self, ticker: str):
        self.dashboard_watch_stocks.discard(ticker)
