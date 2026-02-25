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

# Create tables
Base.metadata.create_all(bind=engine)
