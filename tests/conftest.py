import os
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Base


@pytest.fixture(scope="function")
def db_session():
    # Use in-memory SQLite for testing DB models and services
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()


# =============================================================================
# Sandbox Integration Test Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sandbox_credentials():
    """
    Fixture providing Upstox sandbox credentials from environment variables.
    Returns None if credentials are not available (for CI/CD scenarios).
    """
    credentials = {
        "access_token": os.getenv("UPSTOX_SANDBOX_ACCESS_TOKEN"),
        "api_key": os.getenv("UPSTOX_SANDBOX_API_KEY"),
        "api_secret": os.getenv("UPSTOX_SANDBOX_API_SECRET"),
    }
    
    # Check if sandbox credentials are available
    if not credentials["access_token"]:
        pytest.skip("Sandbox access token not available. Set UPSTOX_SANDBOX_ACCESS_TOKEN env var.")
    
    return credentials


@pytest.fixture(scope="session")
def is_sandbox_available(sandbox_credentials):
    """Check if sandbox environment is available."""
    return sandbox_credentials is not None


@pytest.fixture
def sandbox_service(sandbox_credentials):
    """
    Creates an UpstoxService instance configured for sandbox environment.
    Only available if sandbox credentials are provided.
    """
    if not sandbox_credentials:
        pytest.skip("Sandbox credentials not available")
    
    from services.upstox_service import UpstoxService
    
    # Create service with sandbox token
    service = UpstoxService()
    service.access_token = sandbox_credentials["access_token"]
    service._is_authenticated = True
    
    return service


@pytest.fixture
def sandbox_streamer():
    """
    Creates an UpstoxLiveStream instance for sandbox testing.
    Only available if sandbox credentials are provided.
    """
    if not os.getenv("UPSTOX_SANDBOX_ACCESS_TOKEN"):
        pytest.skip("Sandbox access token not available")
    
    from services.upstox_streamer import UpstoxLiveStream
    
    # Create a simple callback for testing
    async def test_callback(tick):
        pass
    
    streamer = UpstoxLiveStream(test_callback)
    return streamer


# Test markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires sandbox credentials)"
    )
    config.addinivalue_line(
        "markers", "sandbox: mark test as requiring sandbox environment"
    )
    config.addinivalue_line(
        "markers", "websocket: mark test as WebSocket test"
    )


# Skip integration tests by default unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Automatically skip integration tests unless --run-integration flag is passed."""
    if config.getoption("--run-integration", default=False):
        return
    
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
