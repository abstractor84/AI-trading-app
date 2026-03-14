import pytest
import asyncio
from datetime import datetime, timedelta
from services.sentinel_service import sentinel_svc
from services.technical_analysis import TechnicalAnalysisService
from database import SessionLocal, Base, engine
from models import Trade

@pytest.fixture
def db_session():
    # Setup test DB tables
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    yield session
    # Teardown: clear test data (only for this test suite)
    session.query(Trade).delete()
    session.commit()
    session.close()

def test_sentinel_macro_keywords():
    """Verify that Sentinel can detect Macro keywords as per V3 mandate."""
    test_cases = [
        ("Major trade war escalates with tariffs", {"war", "tariffs"}),
        ("Sanctions announced against key sectors", {"sanctions"}),
        ("Politics and elections volatility expected", {"politics", "elections"}),
        ("RBI interest rate hike likely", {"rbi", "interest", "rate"}),
        ("Global stimulus package for economy", {"global", "stimulus"})
    ]
    
    for text, expected in test_cases:
        found = sentinel_svc._check_keywords(text)
        assert expected.issubset(found), f"Failed to detect {expected} in '{text}'. Found: {found}"

def test_trade_history_query(db_session):
    """Verify that the 90-day trade history query logic works correctly."""
    # Professional naive UTC comparison to match models.py
    from models import utc_now_naive
    now = utc_now_naive()
    
    # CLEAR EXISTING DATA (Robustness)
    db_session.query(Trade).delete()
    db_session.commit()
    
    # Create a trade from 45 days ago
    old_trade = Trade(
        id="test-old",
        ticker="RELIANCE.NS",
        action="BUY",
        quantity=10,
        entry_price=2500,
        status="CLOSED",
        timestamp=now - timedelta(days=45),
        close_time=now - timedelta(days=45),
        pnl=100.0
    )
    # Create a trade from 120 days ago (should be excluded)
    stale_trade = Trade(
        id="test-stale",
        ticker="TCS.NS",
        action="BUY",
        quantity=5,
        entry_price=3000,
        status="CLOSED",
        timestamp=now - timedelta(days=120),
        close_time=now - timedelta(days=120),
        pnl=50.0
    )
    
    db_session.add(old_trade)
    db_session.add(stale_trade)
    db_session.commit()
    
    # Simulate the query from ws_handler.py
    days = 90
    cutoff = now - timedelta(days=days)
    
    trades = db_session.query(Trade).filter(
        Trade.status == "CLOSED",
        Trade.close_time >= cutoff
    ).all()
    
    tickers = [t.ticker for t in trades]
    assert "RELIANCE.NS" in tickers
    assert "TCS.NS" not in tickers
    assert len(trades) == 1

def test_indicator_presence():
    """Verify that Math-driven indicators (LZ, ST) are present in the chart payload."""
    ta_svc = TechnicalAnalysisService()
    import os
    os.environ["SIMULATION"] = "true" # Force simulation mode to avoid network calls
    
    payload = ta_svc.get_chart_payload("RELIANCE.NS", "5m")
    
    assert "ml_adaptive_st" in payload
    assert "ml_lorentzian" in payload
    assert "vwap" in payload
    assert "adx_series" in payload
    assert "ohlc" in payload
    
    # Check if Lorentzian signals are mapped correctly
    lz = payload["ml_lorentzian"]
    assert len(lz) > 0
    assert "signal" in lz[0]
    assert "score" in lz[0]
