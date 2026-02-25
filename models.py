from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base, engine

class Trade(Base):
    __tablename__ = "trades"

    id = Column(String, primary_key=True, index=True)
    ticker = Column(String, index=True)
    action = Column(String)  # BUY or SHORT SELL
    quantity = Column(Integer)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    target_1 = Column(Float)
    target_2 = Column(Float)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    status = Column(String)  # OPEN or CLOSED
    timestamp = Column(DateTime, default=datetime.utcnow)
    close_time = Column(DateTime, nullable=True)

class StockDetail(Base):
    __tablename__ = "stock_details"
    
    ticker = Column(String, primary_key=True, index=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    ta_data_json = Column(String)  # Storing as stringified JSON
    sentiment_json = Column(String)
    ai_recommendation_json = Column(String)

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

# Create tables
Base.metadata.create_all(bind=engine)
