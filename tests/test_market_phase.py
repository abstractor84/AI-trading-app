import pytest
from datetime import datetime, time as dtime
from unittest.mock import patch, MagicMock
from services.market_phase import MarketPhaseService, MarketPhase

@pytest.fixture
def svc():
    return MarketPhaseService()

def test_get_current_phase_all(svc):
    # Midnight -> CLOSED
    with patch("services.market_phase.datetime") as mock_dt:
        mock_dt.now.return_value.time.return_value = dtime(0, 0)
        assert svc.get_current_phase() == MarketPhase.CLOSED
        
    # 8:30 IST -> PRE_MARKET_SETUP
    with patch("services.market_phase.datetime") as mock_dt:
        mock_dt.now.return_value.time.return_value = dtime(8, 30)
        assert svc.get_current_phase() == MarketPhase.PRE_MARKET_SETUP

    # 15:00 IST -> POWER_HOUR
    with patch("services.market_phase.datetime") as mock_dt:
        mock_dt.now.return_value.time.return_value = dtime(15, 0)
        assert svc.get_current_phase() == MarketPhase.POWER_HOUR

def test_get_phase_context_transition(svc):
    # Start in CLOSED
    with patch("services.market_phase.datetime") as mock_dt:
        mock_dt.now.return_value.time.return_value = dtime(0, 0)
        mock_dt.now.return_value.replace.return_value = datetime(2024, 1, 1, 0, 0)
        ctx1 = svc.get_phase_context()
        assert not ctx1["transitioned"]
        
        # Transition to PRE_MARKET
        mock_dt.now.return_value.time.return_value = dtime(8, 30)
        ctx2 = svc.get_phase_context()
        assert ctx2["transitioned"]
        assert ctx2["phase"] == MarketPhase.PRE_MARKET_SETUP.value

def test_get_phase_context_time_left(svc):
    # 9:20 IST -> OPENING_VOLATILITY (ends at 9:45)
    class MockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 1, 1, 9, 20, tzinfo=tz)
    
    with patch("services.market_phase.datetime", MockDateTime):
        ctx = svc.get_phase_context()
        assert ctx["phase"] == MarketPhase.OPENING_VOLATILITY.value
        # 9:45 - 9:20 = 25 mins
        assert ctx["mins_left_in_phase"] == 25
        # 15:30 - 9:20 = 6h 10m = 370 mins
        assert ctx["mins_to_close"] == 370

def test_get_ai_schedule_all(svc):

    with patch("services.market_phase.datetime") as mock_dt:
        # Pre-Market
        mock_dt.now.return_value.time.return_value = dtime(8, 30)
        sched = svc.get_ai_schedule()
        assert sched["prompt_type"] == "SCAN"
        
        # Closed
        mock_dt.now.return_value.time.return_value = dtime(20, 0)
        sched_closed = svc.get_ai_schedule()
        assert sched_closed["prompt_type"] is None
