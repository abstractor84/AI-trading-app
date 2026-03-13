import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from services.state import AppState
from models import Trade, AppSettings, DashboardStock, DailySummary

@pytest.fixture
def mock_db(db_session):
    with patch("services.state.SessionLocal", return_value=db_session):
        yield db_session

def test_app_state_initialization(mock_db):
    state = AppState()
    assert state.capital == 100000.0
    assert state.max_loss_per_trade == 1000.0

def test_log_and_close_trade(mock_db):
    state = AppState()
    
    # Log trade
    trade = state.log_trade("TCS", "BUY", 10, 3000.0, 2900.0, 3100.0, 3200.0, 
                             phase="MORNING", atr=50.0, risk_per_share=100.0)
    
    assert len(state.open_trades) == 1
    assert state.open_trades[0]["ticker"] == "TCS"
    assert state.open_trades[0]["status"] == "OPEN"
    
    trade_id = trade["id"]
    
    # Close trade
    state.close_trade(trade_id, 3150.0)
    
    assert len(state.open_trades) == 0
    assert len(state.closed_trades) == 1
    assert state.closed_trades[0]["exit_price"] == 3150.0
    assert state.closed_trades[0]["pnl"] == 1500.0 # (3150-3000)*10
    assert state.closed_trades[0]["status"] == "CLOSED"

def test_update_settings(mock_db):
    state = AppState()
    state.update_settings(200000.0, 2000.0, search_engine="tavily", ai_model="gpt-4")
    
    assert state.capital == 200000.0
    assert state.ai_model == "gpt-4"
    
    # Verify DB persistence
    settings = mock_db.query(AppSettings).filter_by(id=1).first()
    assert settings.capital == 200000.0
    assert settings.ai_model == "gpt-4"

def test_daily_reset(mock_db):
    state = AppState()
    state.log_trade("TCS", "BUY", 10, 3000.0, 2900.0, 3100.0, 3200.0)
    state.close_trade(state.open_trades[0]["id"], 3100.0)
    
    # Simulate day change
    state.last_reset_date = datetime.now().date() - timedelta(days=1)
    
    with patch.object(state, '_load_from_db') as mock_load:
        target_date = state.last_reset_date
        state.check_daily_reset()
        
        # Verify summary was saved
        summary = mock_db.query(DailySummary).filter_by(date=target_date).first()
        assert summary is not None
        assert summary.total_trades == 1
        assert summary.total_pnl == 1000.0
        
        # Verify in-memory cleared
        assert len(state.open_trades) == 0
        assert len(state.closed_trades) == 0
        assert mock_load.called

def test_dashboard_stocks(mock_db):
    state = AppState()
    state.add_dashboard_stock("RELIANCE")
    assert "RELIANCE" in state.dashboard_watch_stocks
    
    # Verify DB
    ds = mock_db.query(DashboardStock).filter_by(ticker="RELIANCE").first()
    assert ds is not None
    
    state.remove_dashboard_stock("RELIANCE")
    assert "RELIANCE" not in state.dashboard_watch_stocks
    assert mock_db.query(DashboardStock).count() == 0
