"""
Application State Manager (V2)
Manages in-memory state and database persistence for trades, settings,
and session data. Enhanced for V2 with phase-aware trade fields.
"""
import json
import logging
import uuid
import os
from datetime import datetime, timedelta

from database import SessionLocal
from models import Trade, AppSettings, DashboardStock

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self):
        # Signals (legacy compat — will be replaced by AI advisor in Phase C/D)
        self.standard_ai_signals = []
        self.magic_ai_signals = []
        self.global_context = {}
        self.dashboard_watch_stocks = set()
        self.last_reset_date = datetime.now().date()

        # Trades
        self.open_trades = []
        self.closed_trades = []

        # Settings
        self.capital = 100000.0
        self.max_loss_per_trade = 1000.0
        self.max_daily_loss = 5000.0
        self.search_engine = "tavily"
        self.data_provider = "upstox"
        self.fallback_data = True
        self.fallback_search = True
        self.fallback_ai = True
        self.ai_provider = "google"
        self.ai_model = "gemini-3.1-pro"

        # AI Advisor state (V2)
        self.ai_advisor_message = None  # Current AI guidance
        self.ai_scans_today = []        # Store last 50 recommendations for historical analysis
        self.action_timeline = []  # Chronological log of events
        self.simulation_mode = os.getenv("SIMULATION", "false").lower() == "true"

        self._load_from_db()

    def _load_from_db(self):
        """Load persistent state from database on startup."""
        with SessionLocal() as db:
            # Load settings
            settings = db.query(AppSettings).filter(AppSettings.id == 1).first()
            if settings:
                self.capital = settings.capital
                self.max_loss_per_trade = settings.max_loss_per_trade
                self.max_daily_loss = getattr(settings, 'max_daily_loss', 5000.0)
                self.search_engine = settings.search_engine
                self.data_provider = settings.data_provider
                self.fallback_data = getattr(settings, 'fallback_data', True)
                self.fallback_search = getattr(settings, 'fallback_search', True)
                self.fallback_ai = getattr(settings, 'fallback_ai', True)
                self.ai_provider = settings.ai_provider
                self.ai_model = settings.ai_model

            # Load active and today's closed trades
            import pytz
            ist = pytz.timezone("Asia/Kolkata")
            now_ist = datetime.now(ist)
            # Today's start in IST, converted to naive UTC for DB comparison
            today_start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
            today_start_utc = today_start_ist.astimezone(pytz.UTC).replace(tzinfo=None)

            trades = db.query(Trade).filter(
                (Trade.status == "OPEN") | (Trade.close_time >= today_start_utc)
            ).all()

            for t in trades:
                td = self._to_dict(t)
                if t.status == "OPEN":
                    self.open_trades.append(td)
                else:
                    self.closed_trades.append(td)

            # Load dashboard stocks
            ds_list = db.query(DashboardStock).all()
            self.dashboard_watch_stocks = {ds.ticker for ds in ds_list}

        logger.info(
            f"State loaded: {len(self.open_trades)} open, "
            f"{len(self.closed_trades)} closed trades"
        )

    def _to_dict(self, trade: Trade) -> dict:
        """Convert Trade ORM object to dictionary, including V2 fields."""
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
            "timestamp": trade.timestamp.isoformat() + "Z" if trade.timestamp else None,
            "close_time": trade.close_time.isoformat() + "Z" if trade.close_time else None,
            # V2 fields
            "phase_entered": getattr(trade, 'phase_entered', None),
            "trailing_sl": getattr(trade, 'trailing_sl', None),
            "atr_at_entry": getattr(trade, 'atr_at_entry', None),
            "risk_per_share": getattr(trade, 'risk_per_share', None),
            "max_loss": getattr(trade, 'max_loss', None),
            "ai_reasoning": getattr(trade, 'ai_reasoning', None),
        }

    def check_daily_reset(self):
        """Reset state if it's a new day in IST. Saves summary before wiping."""
        import pytz
        from models import DailySummary
        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)
        current_date = now_ist.date()

        if current_date > self.last_reset_date:
            logger.info(f"Midnight (IST) passed: {current_date} > {self.last_reset_date}. Saving summary...")
            date_str = self.last_reset_date.strftime("%Y-%m-%d")

            # Save Summary
            total_pnl = sum(t.get('pnl', 0) for t in self.closed_trades)
            wins = sum(1 for t in self.closed_trades if t.get('pnl', 0) > 0)
            losses = sum(1 for t in self.closed_trades if t.get('pnl', 0) < 0)

            with SessionLocal() as db:
                try:
                    summary = db.query(DailySummary).filter(DailySummary.date == date_str).first()
                    if not summary:
                        summary = DailySummary(date=date_str)
                        db.add(summary)

                    summary.total_trades = len(self.closed_trades)
                    summary.wins = wins
                    summary.losses = losses
                    summary.total_pnl = total_pnl
                    db.commit()
                except Exception as e:
                    logger.error(f"Daily summary save failed: {e}")
                    db.rollback()

            self.open_trades.clear()
            self.closed_trades.clear()
            self.standard_ai_signals.clear()
            self.magic_ai_signals.clear()
            self.ai_advisor_message = None
            self.action_timeline.clear()
            self._load_from_db()
            self.last_reset_date = current_date
    def update_settings(self, capital: float, max_loss: float,
                        search_engine: str = 'tavily', data_provider: str = 'upstox',
                        fallback_data: bool = True, fallback_search: bool = True,
                        fallback_ai: bool = True,
                        ai_provider: str = 'google', ai_model: str = 'gemini-3.1-pro'):
        self.capital = capital
        self.max_loss_per_trade = max_loss
        self.search_engine = search_engine
        self.data_provider = data_provider
        self.fallback_data = fallback_data
        self.fallback_search = fallback_search
        self.fallback_ai = fallback_ai
        self.ai_provider = ai_provider
        self.ai_model = ai_model

        with SessionLocal() as db:
            settings = db.query(AppSettings).filter(AppSettings.id == 1).first()
            if not settings:
                settings = AppSettings(id=1)
                db.add(settings)

            settings.capital = capital
            settings.max_loss_per_trade = max_loss
            settings.search_engine = search_engine
            settings.data_provider = data_provider
            settings.fallback_data = fallback_data
            settings.fallback_search = fallback_search
            settings.fallback_ai = fallback_ai
            settings.ai_provider = ai_provider
            settings.ai_model = ai_model
            db.commit()

    def log_trade(self, ticker: str, action: str, qty: int, entry_price: float,
                  sl: float, t1: float, t2: float,
                  phase: str = None, atr: float = None,
                  risk_per_share: float = None) -> dict:
        """
        Log a new trade with V2 risk metadata.
        SL/Target/Qty come from the Risk Engine, not raw AI output.
        """
        trade_id = str(uuid.uuid4())
        new_trade = Trade(
            id=trade_id, ticker=ticker, action=action, quantity=qty,
            entry_price=entry_price, stop_loss=sl, target_1=t1, target_2=t2,
            status="OPEN",
            phase_entered=phase,
            atr_at_entry=atr,
            risk_per_share=risk_per_share,
            max_loss=round(qty * risk_per_share, 2) if risk_per_share else None,
            trailing_sl=sl,  # Initial trailing SL = SL
        )
        with SessionLocal() as db:
            db.add(new_trade)
            db.commit()
            db.refresh(new_trade)

        trade_dict = self._to_dict(new_trade)
        self.open_trades.append(trade_dict)

        # Add to action timeline
        self.action_timeline.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "TRADE_OPEN",
            "message": f"{action} {ticker} × {qty} @ ₹{entry_price:.2f}"
        })

        return trade_dict

    def close_trade(self, trade_id: str, exit_price: float):
        """Close an open trade and move it to closed list."""
        tid_str = str(trade_id)
        logger.info(f"SKEPTIC: Finalizing trade exit for ID: {tid_str} @ ₹{exit_price}")
        
        with SessionLocal() as db:
            trade = db.query(Trade).filter(Trade.id == tid_str).first()
            if trade:
                if trade.action == "BUY":
                    pnl = (exit_price - trade.entry_price) * trade.quantity
                else:
                    pnl = (trade.entry_price - exit_price) * trade.quantity

                trade.exit_price = exit_price
                trade.pnl = round(pnl, 2)
                trade.status = "CLOSED"
                trade.close_time = datetime.utcnow()
                db.commit()
                db.refresh(trade)

                # Update in-memory state: Filter MUST be string-to-string to prevent integer mismatches
                before = len(self.open_trades)
                self.open_trades = [t for t in self.open_trades if str(t.get('id')) != tid_str]
                self.closed_trades.append(self._to_dict(trade))
                logger.info(f"SKEPTIC: Memory sync complete. (Open: {before} -> {len(self.open_trades)})")

                # Add to action timeline
                self.action_timeline.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "TRADE_CLOSE",
                    "message": (
                        f"SUCCESS: Closed {trade.ticker} @ ₹{exit_price:.2f} | "
                        f"P&L: ₹{trade.pnl:.2f}"
                    )
                })
            else:
                logger.error(f"SKEPTIC: ID {tid_str} not found in DB. Logic sync failure.")

    def add_dashboard_stock(self, ticker: str):
        self.dashboard_watch_stocks.add(ticker)
        with SessionLocal() as db:
            existing = db.query(DashboardStock).filter(DashboardStock.ticker == ticker).first()
            if not existing:
                db.add(DashboardStock(ticker=ticker))
                db.commit()

    def update_ltp(self, ticker: str, price: float):
        """Update LTP for an open trade in memory."""
        for trade in self.open_trades:
            if trade['ticker'] == ticker:
                trade['current_price'] = price
                # Update P&L
                if trade['action'] == "BUY":
                    pnl = (price - trade['entry_price']) * trade['quantity']
                else:
                    pnl = (trade['entry_price'] - price) * trade['quantity']
                trade['pnl'] = round(pnl, 2)
                break

    def remove_dashboard_stock(self, ticker: str):
        self.dashboard_watch_stocks.discard(ticker)
        with SessionLocal() as db:
            db.query(DashboardStock).filter(DashboardStock.ticker == ticker).delete()
            db.commit()
