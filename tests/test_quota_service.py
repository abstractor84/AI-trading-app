from services.quota_service import QuotaService
from models import ApiUsage
import pytest
from unittest.mock import patch
from datetime import datetime, timedelta, UTC

@pytest.fixture
def quota_svc(db_session):
    svc = QuotaService()
    # Mock the internal db session getter since quota_service initializes its own inside check/log 
    # but we can replace the class level SessionLocal or mock it just for the test.
    return svc

@patch("services.quota_service.SessionLocal")
def test_quota_service_defaults(mock_session, quota_svc, db_session):
    mock_session.return_value = db_session
    
    # Check default initialization for a provider
    status = quota_svc.check_quota("groq")
    assert status["limit_rpd"] == 20
    assert status["remaining_rpd"] == 20
    assert status["can_call"] is True
    assert status["low_quota"] is False

@patch("services.quota_service.SessionLocal")
def test_quota_service_logging_and_limits(mock_session, quota_svc, db_session):
    mock_session.return_value = db_session
    
    # Log 18 requests, bypassing the 15/min RPM limit
    for i in range(18):
        quota_svc.log_usage("google", tokens=10)
        # Manually reset minute requests to bypass RPM
        usage = db_session.query(ApiUsage).filter_by(model_name="google").first()
        usage.minute_requests = 0
        db_session.commit()
        
    status = quota_svc.check_quota("google")
    assert status["remaining_rpd"] == 2
    assert status["low_quota"] is True
    assert status["can_call"] is True
    
    # Log 2 more to hit limit
    quota_svc.log_usage("google", tokens=10)
    quota_svc.log_usage("google", tokens=10)
    
    status = quota_svc.check_quota("google")
    assert status["remaining_rpd"] == 0
    assert status["low_quota"] is True
    assert status["can_call"] is False # Max 20 reached

@patch("services.quota_service.SessionLocal")
def test_quota_service_rollover(mock_session, quota_svc, db_session):
    mock_session.return_value = db_session
    
    # max out
    for _ in range(20):
        quota_svc.log_usage("sambanova", tokens=1)
        
    status = quota_svc.check_quota("sambanova")
    assert status["can_call"] is False
    
    # Artificially age the record by 1 day
    usage = db_session.query(ApiUsage).filter_by(model_name="sambanova").first()
    usage.last_request_at = datetime.now(UTC) - timedelta(days=1)
    db_session.commit()
    
    # Since a day passed, should allow calls again
    status = quota_svc.check_quota("sambanova")
    assert status["can_call"] is True
    assert status["remaining_rpd"] == 20
    assert status["low_quota"] is False

