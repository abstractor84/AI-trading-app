"""
V2 Database Models
Redesigned schema for the AI Trading Companion with time-bound trades,
market snapshots, and AI interaction audit trail.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from datetime import datetime
from database import Base, engine


class Trade(Base):
    """
    Open and closed trades with full lifecycle tracking.
    V2: Added time-validity, trailing SL, partial exits, and AI guidance history.
    """
    __tablename__ = "trades"

    id = Column(String, primary_key=True, index=True)
    ticker = Column(String, index=True)
    action = Column(String)            # BUY or SHORT SELL
    quantity = Column(Integer)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    target_1 = Column(Float)
    target_2 = Column(Float)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    status = Column(String, default="OPEN")     # OPEN, CLOSED
    timestamp = Column(DateTime, default=datetime.utcnow)
    close_time = Column(DateTime, nullable=True)

    # V2 Fields
    phase_entered = Column(String, nullable=True)    # Market phase when logged
    ai_reasoning = Column(Text, nullable=True)       # Why AI recommended this
    valid_until = Column(DateTime, nullable=True)     # Recommendation expiry
    trailing_sl = Column(Float, nullable=True)        # Dynamic trailing SL
    atr_at_entry = Column(Float, nullable=True)       # ATR when trade was opened
    risk_per_share = Column(Float, nullable=True)     # Risk per share (entry - SL)
    max_loss = Column(Float, nullable=True)           # Max loss for this trade
    partial_exits_json = Column(Text, nullable=True)  # JSON: [{price, qty, time}]
    ai_updates_json = Column(Text, nullable=True)     # JSON: history of AI guidance


class MarketSnapshot(Base):
    """
    Periodic market state snapshots for AI context and historical analysis.
    Captured at phase transitions and on AI call schedule.
    """
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    market_phase = Column(String)

    # Index levels
    nifty_price = Column(Float, nullable=True)
    nifty_change_pct = Column(Float, nullable=True)
    banknifty_price = Column(Float, nullable=True)
    banknifty_change_pct = Column(Float, nullable=True)
    vix = Column(Float, nullable=True)

    # Breadth
    advances = Column(Integer, nullable=True)
    declines = Column(Integer, nullable=True)

    # Sector performance (JSON)
    sector_data_json = Column(Text, nullable=True)


class AIInteraction(Base):
    """
    Audit log for every AI API call made by the system.
    Tracks quota usage, prompt types, and whether the user acted on the advice.
    """
    __tablename__ = "ai_interactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    prompt_type = Column(String)         # SCAN, POSITION_REVIEW, EXIT_GUIDANCE
    model_used = Column(String)
    tokens_used = Column(Integer, default=0)
    input_summary = Column(Text)         # Abbreviated prompt context
    output_json = Column(Text)           # Full AI response
    was_acted_upon = Column(Boolean, default=False)
    trade_date = Column(String, index=True)  # YYYY-MM-DD for daily grouping


class Watchlist(Base):
    __tablename__ = "watchlists"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class WatchlistStock(Base):
    __tablename__ = "watchlist_stocks"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    watchlist_id = Column(Integer, index=True)
    ticker = Column(String, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)


class AppSettings(Base):
    """
    V2: Added daily loss limit and removed stale signal caching.
    """
    __tablename__ = "app_settings"
    id = Column(Integer, primary_key=True, index=True)
    capital = Column(Float, default=100000.0)
    max_loss_per_trade = Column(Float, default=1000.0)
    max_daily_loss = Column(Float, default=5000.0)     # V2: Daily loss cap
    search_engine = Column(String, default="ddgs")
    data_provider = Column(String, default="yfinance")
    search_fallback = Column(Boolean, default=False)
    auto_refresh = Column(Boolean, default=True)
    ai_provider = Column(String, default="google")
    ai_model = Column(String, default="gemini-2.5-flash")
    updated_at = Column(DateTime, default=datetime.utcnow)


class DashboardStock(Base):
    __tablename__ = "dashboard_stocks"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticker = Column(String, unique=True, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)


class ApiUsage(Base):
    """Retained for backward compatibility with quota_service."""
    __tablename__ = "api_usage"
    model_name = Column(String, primary_key=True, index=True)
    minute_requests = Column(Integer, default=0)
    minute_tokens = Column(Integer, default=0)
    day_requests = Column(Integer, default=0)
    last_request_at = Column(DateTime, default=datetime.utcnow)
    limit_rpm = Column(Integer, default=5)
    limit_tpm = Column(Integer, default=250000)
    limit_rpd = Column(Integer, default=20)


# Create all tables (additive — won't drop existing)
Base.metadata.create_all(bind=engine)
