import os
from datetime import datetime, timedelta
from models import Trade, AIInteraction, DailySummary, MarketSnapshot
from database import cleanup_old_records, DATABASE_URL

def test_database_url_is_absolute():
    # Verify the database URL is an absolute path to prevent rogue DB files
    assert "sqlite:////" in DATABASE_URL or "sqlite:///C:" in DATABASE_URL or ("Trading/gemini_nse_trader" in DATABASE_URL)
    assert not DATABASE_URL.endswith("./trading_data.db")

def test_cleanup_old_records(db_session):
    # Create old records
    old_date = datetime.utcnow() - timedelta(days=95)
    recent_date = datetime.utcnow() - timedelta(days=10)

    t_old = Trade(id="1", ticker="TCS", action="BUY", timestamp=old_date)
    t_recent = Trade(id="2", ticker="RIL", action="BUY", timestamp=recent_date)
    
    ai_old = AIInteraction(prompt_type="SCAN", timestamp=old_date, input_summary="", output_json="")
    ai_recent = AIInteraction(prompt_type="SCAN", timestamp=recent_date, input_summary="", output_json="")
    
    sum_old = DailySummary(date=(old_date).strftime("%Y-%m-%d"))
    sum_recent = DailySummary(date=(recent_date).strftime("%Y-%m-%d"))
    
    snap_old = MarketSnapshot(timestamp=old_date)
    snap_recent = MarketSnapshot(timestamp=recent_date)

    db_session.add_all([t_old, t_recent, ai_old, ai_recent, sum_old, sum_recent, snap_old, snap_recent])
    db_session.commit()

    # Verify counts before
    assert db_session.query(Trade).count() == 2
    assert db_session.query(AIInteraction).count() == 2
    assert db_session.query(DailySummary).count() == 2
    assert db_session.query(MarketSnapshot).count() == 2

    # Run cleanup
    cleanup_old_records(db_session, days=90)

    # Verify old records are deleted, recent are kept
    assert db_session.query(Trade).count() == 1
    assert db_session.query(Trade).first().id == "2"

    assert db_session.query(AIInteraction).count() == 1
    assert db_session.query(DailySummary).count() == 1
    assert db_session.query(MarketSnapshot).count() == 1
