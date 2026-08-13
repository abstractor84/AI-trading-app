"""
Integration tests for Upstox API using sandbox environment.

These tests require sandbox credentials to be set in environment variables:
- UPSTOX_SANDBOX_ACCESS_TOKEN: Sandbox access token
- UPSTOX_SANDBOX_API_KEY: Sandbox API key (optional)
- UPSTOX_SANDBOX_API_SECRET: Sandbox API secret (optional)

Run with: pytest tests/integration/test_upstox_sandbox.py --run-integration -v
"""

import os
import pytest
import asyncio
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Import services under test
from services.upstox_service import UpstoxService
from services.upstox_streamer import UpstoxLiveStream, StreamingMode

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# =============================================================================
# Test: Instrument Search API (Sandbox)
# =============================================================================

@pytest.mark.integration
@pytest.mark.sandbox
class TestInstrumentSearchSandbox:
    """Test instrument search API against sandbox environment."""
    
    def test_search_instruments_basic(self, sandbox_service):
        """Test basic instrument search functionality."""
        if sandbox_service is None:
            pytest.skip("Sandbox service not available")
        
        # Search for a known stock
        results = sandbox_service.search_instruments(query="RELIANCE")
        
        # Verify results
        assert results is not None, "Search should return results"
        assert isinstance(results, list), "Results should be a list"
        assert len(results) > 0, "Should find at least one result for RELIANCE"
        
        # Check result structure
        first_result = results[0]
        assert "instrument_key" in first_result, "Result should have instrument_key"
        assert "trading_symbol" in first_result, "Result should have trading_symbol"
        
        logger.info(f"Found {len(results)} results for RELIANCE")
    
    def test_search_instruments_nse(self, sandbox_service):
        """Test instrument search with exchange filter."""
        if sandbox_service is None:
            pytest.skip("Sandbox service not available")
        
        # Search with exchange filter
        results = sandbox_service.search_instruments(
            query="NIFTY",
            exchanges=["NSE"]
        )
        
        assert results is not None, "Search should return results"
        assert isinstance(results, list), "Results should be a list"
        
        logger.info(f"Found {len(results)} NSE results for NIFTY")
    
    def test_search_instruments_multiple_exchanges(self, sandbox_service):
        """Test instrument search across multiple exchanges."""
        if sandbox_service is None:
            pytest.skip("Sandbox service not available")
        
        # Search across multiple exchanges
        results = sandbox_service.search_instruments(
            query="SBIN",
            exchanges=["NSE", "BSE"]
        )
        
        assert results is not None, "Search should return results"
        assert isinstance(results, list), "Results should be a list"
        
        logger.info(f"Found {len(results)} results for SBIN across NSE and BSE")


# =============================================================================
# Test: Market Quote API (Sandbox)
# =============================================================================

@pytest.mark.integration
@pytest.mark.sandbox
class TestMarketQuoteSandbox:
    """Test market quote API against sandbox environment."""
    
    def test_fetch_market_quote_single(self, sandbox_service):
        """Test fetching single instrument quote."""
        if sandbox_service is None:
            pytest.skip("Sandbox service not available")
        
        # First search for an instrument key
        search_results = sandbox_service.search_instruments(query="TCS")
        assert search_results and len(search_results) > 0
        
        instrument_key = search_results[0]["instrument_key"]
        
        # Fetch quote
        quote = sandbox_service.fetch_market_quote(instrument_key)
        
        assert quote is not None, "Quote should be returned"
        assert quote.get("status") == "success", "Quote status should be success"
        assert "data" in quote, "Quote should have data"
        
        logger.info(f"Fetched quote for {instrument_key}")
    
    def test_fetch_market_quote_multiple(self, sandbox_service):
        """Test fetching multiple instrument quotes."""
        if sandbox_service is None:
            pytest.skip("Sandbox service not available")
        
        # Search for multiple instruments
        search_results = sandbox_service.search_instruments(
            query="",
            exchanges=["NSE"],
            instrument_types=["index"],
            records=5
        )
        
        if not search_results or len(search_results) < 2:
            pytest.skip("Need at least 2 instruments for this test")
        
        # Get first two instrument keys
        keys = [r["instrument_key"] for r in search_results[:2]]
        combined_key = ",".join(keys)
        
        # Fetch quotes
        quote = sandbox_service.fetch_market_quote(combined_key)
        
        assert quote is not None, "Quote should be returned"
        assert quote.get("status") == "success", "Quote status should be success"
        
        logger.info(f"Fetched quotes for {len(quote.get('data', {}))} instruments")


# =============================================================================
# Test: WebSocket Connection (Sandbox)
# =============================================================================

@pytest.mark.integration
@pytest.mark.websocket
class TestWebSocketSandbox:
    """Test WebSocket connections against sandbox environment."""
    
    @pytest.mark.asyncio
    async def test_streamer_initialization(self, sandbox_streamer):
        """Test that streamer initializes correctly."""
        if sandbox_streamer is None:
            pytest.skip("Sandbox streamer not available")
        
        # Check initial state
        assert sandbox_streamer.running == True
        assert isinstance(sandbox_streamer.subscribed_keys, set)
        assert len(sandbox_streamer.subscribed_keys) == 0
        
        logger.info("Streamer initialized successfully")
    
    @pytest.mark.asyncio
    async def test_subscribe_method_signature(self):
        """Test that subscribe method accepts mode parameter."""
        # Create a mock callback
        async def mock_callback(tick):
            pass
        
        # Create streamer with mock
        with patch('services.upstox_streamer.UpstoxService'):
            streamer = UpstoxLiveStream(mock_callback)
        
        # Test subscribe with mode parameter
        streamer.subscribe("NSE_EQ|INE002A01018", mode="ltpc")
        
        assert streamer._current_mode == "ltpc"
        assert "NSE_EQ|INE002A01018" in streamer.subscribed_keys
        
        logger.info("Subscribe with mode parameter works correctly")
    
    @pytest.mark.asyncio
    async def test_change_mode_method(self):
        """Test that change_mode method works correctly."""
        # Create a mock callback
        async def mock_callback(tick):
            pass
        
        # Create streamer with mock
        with patch('services.upstox_streamer.UpstoxService'):
            streamer = UpstoxLiveStream(mock_callback)
        
        # Subscribe to some instruments
        streamer.subscribed_keys.add("NSE_EQ|INE002A01018")
        
        # Test change_mode - returns bool
        result = streamer.change_mode("NSE_EQ|INE002A01018", "full_d30")
        
        assert result == True
        assert streamer._current_mode == "full_d30"
        
        logger.info("Change mode method works correctly")
    
    @pytest.mark.asyncio
    async def test_change_mode_invalid(self):
        """Test that change_mode rejects invalid modes."""
        # Create a mock callback
        async def mock_callback(tick):
            pass
        
        # Create streamer with mock
        with patch('services.upstox_streamer.UpstoxService'):
            streamer = UpstoxLiveStream(mock_callback)
        
        # Subscribe to some instruments
        streamer.subscribed_keys.add("NSE_EQ|INE002A01018")
        
        # Test change_mode with invalid mode - should return False
        result = streamer.change_mode("NSE_EQ|INE002A01018", "invalid_mode")
        
        assert result == False
        # Current mode should remain unchanged (default: full)
        assert streamer._current_mode == "full"
        
        logger.info("Change mode correctly rejects invalid modes")
    
    @pytest.mark.asyncio
    async def test_valid_streaming_modes(self):
        """Test that all valid streaming modes are accepted."""
        # Create a mock callback
        async def mock_callback(tick):
            pass
        
        # Create streamer with mock
        with patch('services.upstox_streamer.UpstoxService'):
            streamer = UpstoxLiveStream(mock_callback)
        
        # Test all valid modes
        valid_modes = ["ltpc", "full", "option_greeks", "full_d30"]
        
        for mode in valid_modes:
            result = streamer.change_mode("NSE_EQ|INE002A01018", mode)
            assert result == True, f"Mode '{mode}' should be accepted"
            assert streamer._current_mode == mode
        
        logger.info("All valid streaming modes are accepted")


# =============================================================================
# Test: Error Handling and Logging (Sandbox)
# =============================================================================

@pytest.mark.integration
@pytest.mark.sandbox
class TestErrorHandlingSandbox:
    """Test error handling in sandbox environment."""
    
    def test_unauthenticated_request(self):
        """Test that unauthenticated requests are handled gracefully."""
        service = UpstoxService()
        service.access_token = "invalid_token"
        service._is_authenticated = False
        
        # Try to search - should return None gracefully
        result = service.search_instruments(query="RELIANCE")
        
        # Should return None for unauthenticated requests
        assert result is None
        
        logger.info("Unauthenticated requests handled gracefully")
    
    def test_invalid_instrument_key(self, sandbox_service):
        """Test handling of invalid instrument keys."""
        if sandbox_service is None:
            pytest.skip("Sandbox service not available")
        
        # Try to fetch quote for invalid key
        quote = sandbox_service.fetch_market_quote("INVALID_KEY_THAT_DOES_NOT_EXIST")
        
        # Should handle gracefully (may return None or empty data)
        # The exact behavior depends on API response
        logger.info(f"Invalid key handling result: {quote}")


# =============================================================================
# Test: Rate Limiting (Sandbox)
# =============================================================================

@pytest.mark.integration
@pytest.mark.sandbox
class TestRateLimitingSandbox:
    """Test rate limiting behavior in sandbox environment."""
    
    def test_rate_limit_handling(self, sandbox_service):
        """Test that rate limiting is handled correctly."""
        if sandbox_service is None:
            pytest.skip("Sandbox service not available")
        
        # Make multiple rapid requests to test rate limiting
        # Note: This may hit rate limits in sandbox
        results = []
        for i in range(3):
            result = sandbox_service.search_instruments(query="RELIANCE")
            results.append(result)
        
        # Should handle rate limiting gracefully
        logger.info(f"Rate limit test: {len([r for r in results if r is not None])} successful requests")


# =============================================================================
# Test: Mode Change Functionality
# =============================================================================

@pytest.mark.integration
class TestModeChangeFunctionality:
    """Test mode change functionality (can run without sandbox)."""
    
    @pytest.mark.asyncio
    async def test_mode_change_sends_correct_request(self):
        """Test that mode change sends the correct WebSocket request."""
        # Create mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        
        # Create streamer with mock
        async def mock_callback(tick):
            pass
        
        with patch('services.upstox_streamer.UpstoxService'):
            streamer = UpstoxLiveStream(mock_callback)
            streamer.ws = mock_ws
            
            # Subscribe first
            streamer.subscribed_keys.add("NSE_EQ|INE002A01018")
            
            # Change mode
            await streamer._send_mode_change(["NSE_EQ|INE002A01018"], "full_d30")
            
            # Verify send was called
            assert mock_ws.send.called
            
            # Get the sent data
            sent_data = mock_ws.send.call_args[0][0]
            import json
            sent_json = json.loads(sent_data.decode('utf-8') if isinstance(sent_data, bytes) else sent)
            
            # Verify request structure
            assert sent_json["method"] == "changeMode"
            assert sent_json["data"]["mode"] == "full_d30"
            assert "NSE_EQ|INE002A01018" in sent_json["data"]["instrumentKeys"]
            
            logger.info(f"Mode change request: {sent_json}")
    
    @pytest.mark.asyncio
    async def test_subscribe_with_different_modes(self):
        """Test subscribing with different modes."""
        async def mock_callback(tick):
            pass
        
        with patch('services.upstox_streamer.UpstoxService'):
            # Test ltpc mode
            streamer1 = UpstoxLiveStream(mock_callback)
            streamer1.subscribe("NSE_EQ|INE002A01018", mode="ltpc")
            assert streamer1._current_mode == "ltpc"
            
            # Test full mode
            streamer2 = UpstoxLiveStream(mock_callback)
            streamer2.subscribe("NSE_EQ|INE002A01018", mode="full")
            assert streamer2._current_mode == "full"
            
            # Test option_greeks mode
            streamer3 = UpstoxLiveStream(mock_callback)
            streamer3.subscribe("NSE_EQ|INE002A01018", mode="option_greeks")
            assert streamer3._current_mode == "option_greeks"
            
            logger.info("Subscribe with different modes works correctly")


# =============================================================================
# Test: Enhanced Error Logging
# =============================================================================

@pytest.mark.integration
class TestEnhancedErrorLogging:
    """Test enhanced error logging functionality."""
    
    def test_error_payload_creation(self):
        """Test that error payloads are created correctly."""
        from services.upstox_service import create_error_payload, ErrorCategory
        
        # Create error payload
        payload = create_error_payload(
            endpoint="test_endpoint",
            method="GET",
            status_code=401,
            response_text="Unauthorized",
            request_params={"key": "value"},
            correlation_id="test-correlation-id"
        )
        
        # Verify payload structure
        assert payload["endpoint"] == "test_endpoint"
        assert payload["method"] == "GET"
        assert payload["status_code"] == 401
        assert payload["error_category"] == ErrorCategory.AUTHENTICATION.value
        assert payload["correlation_id"] == "test-correlation-id"
        assert payload["request_params"] == {"key": "value"}
        
        logger.info(f"Error payload created: {payload}")
    
    def test_error_categorization(self):
        """Test error categorization."""
        from services.upstox_service import categorize_error, ErrorCategory
        
        # Test authentication error
        category = categorize_error(response_status=401)
        assert category == ErrorCategory.AUTHENTICATION
        
        # Test rate limit error
        category = categorize_error(response_status=429)
        assert category == ErrorCategory.RATE_LIMIT
        
        # Test validation error
        category = categorize_error(response_status=400)
        assert category == ErrorCategory.VALIDATION
        
        logger.info("Error categorization works correctly")


# =============================================================================
# Helper: Run Integration Tests
# =============================================================================

if __name__ == "__main__":
    """
    Run integration tests manually:
    
    pytest tests/integration/test_upstox_sandbox.py --run-integration -v
    
    Or with specific markers:
    pytest tests/integration/test_upstox_sandbox.py --run-integration -v -m "not websocket"
    """
    pytest.main([__file__, "--run-integration", "-v"])
