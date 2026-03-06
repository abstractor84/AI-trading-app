import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Force an absolute path so we never create rogue database files in random directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'trading_data.db')}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def cleanup_old_records(db, days=90):
    """
    User Feedback: Keep data persisted for at least 90 days, then clean up.
    Deletes records older than 'days' from trades, ai_interactions, and summaries.
    """
    import logging
    logger = logging.getLogger(__name__)
    cutoff_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    
    try:
        # Avoid circular imports
        from models import Trade, AIInteraction, DailySummary, MarketSnapshot
        
        # Cleanup Trades
        deleted_trades = db.query(Trade).filter(Trade.timestamp < cutoff_date).delete()
        # Cleanup AI Interactions
        deleted_ai = db.query(AIInteraction).filter(AIInteraction.timestamp < cutoff_date).delete()
        # Cleanup Summaries
        cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")
        deleted_sums = db.query(DailySummary).filter(DailySummary.date < cutoff_date_str).delete()
        # Cleanup Snapshots
        deleted_snaps = db.query(MarketSnapshot).filter(MarketSnapshot.timestamp < cutoff_date).delete()
        
        db.commit()
        logger.info(f"DB Cleanup: Removed {deleted_trades} Trades, {deleted_ai} AI logs, {deleted_sums} Summaries, {deleted_snaps} Snapshots older than {days} days.")
    except Exception as e:
        logger.error(f"Failed to cleanup old records: {e}")
        db.rollback()
