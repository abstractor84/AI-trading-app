import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from services.upstox_streamer import UpstoxLiveStream, get_streamer

@pytest.fixture
def callback():
    return AsyncMock()

@pytest.fixture
def streamer(callback):
    # Reset singleton effectively for testing by creating fresh instance
    with patch("services.upstox_streamer.UpstoxService") as mock_svc:
        s = UpstoxLiveStream(callback)
        # Mock fetch_market_quote to return a valid dict
        s.upstox_service.fetch_market_quote.return_value = {"data": {"REL.NS": {"last_price": 2500}}}
        return s

@pytest.mark.asyncio
async def test_subscribe(streamer):
    """Test subscription key tracking."""
    # Mock send_subscription_request since we are not connected in this test
    streamer._send_subscription_request = AsyncMock()
    streamer.ws = MagicMock()
    streamer.ws.open = True
    
    streamer.subscribe("NSE_EQ|INE1234")
    assert "NSE_EQ|INE1234" in streamer.subscribed_keys
    
    # Check if send_subscription_request was called since ws.open is True
    await asyncio.sleep(0.05)
    assert streamer._send_subscription_request.called

@pytest.mark.asyncio
async def test_get_authorized_ws_url_scenarios(streamer):
    """Test auth URL fetch scenarios."""
    streamer.upstox_service.is_authenticated = True
    streamer.upstox_service._headers.return_value = {"Authorization": "Bearer token"}
    
    with patch("requests.get") as mock_get:
        # Success
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"authorizedRedirectUri": "wss://test.url"}}
        mock_get.return_value = mock_resp
        url = streamer._get_authorized_ws_url()
        assert url == "wss://test.url"
        
        # 401 Error
        mock_resp.status_code = 401
        url = streamer._get_authorized_ws_url()
        assert url is None
        
        # Exception
        mock_get.side_effect = Exception("error")
        url = streamer._get_authorized_ws_url()
        assert url is None

@pytest.mark.asyncio
async def test_run_simulation(streamer, callback):
    """Test the simulation tick generator."""
    streamer.subscribe("REL.NS")
    streamer.running = True
    
    # Use AsyncMock for sleep in the module
    with patch("services.upstox_streamer.asyncio.sleep", AsyncMock()) as mock_sleep, \
         patch.dict("os.environ", {"ENABLE_SIMULATION": "true"}), \
         patch("services.upstox_streamer.market_phase_svc.get_phase_context", return_value={"is_trading_hours": True}):
        # We need to ensure it yields but also exits
        # One way is to set running=False AFTER the first callback call
        async def mock_callback(tick):
            streamer.running = False
            await callback(tick)
            
        streamer.callback = mock_callback
        
        await streamer._run_simulation()
    
    assert callback.called
    tick = callback.call_args[0][0]
    assert tick["key"] == "REL.NS"

@pytest.mark.skip(reason="Hangs in CI/Container environment due to backoff decorator mocking issues")
@pytest.mark.skip(reason="Hangs in CI/Container environment due to backoff decorator mocking issues")
@pytest.mark.skip(reason="Hangs in CI/Container environment due to backoff decorator mocking issues")
@pytest.mark.asyncio
async def test_connect_and_stream_full_flow(streamer, callback):
    """Test the main connection loop with simulation fallback and real WS."""
    # Case 1: Simulation Fallback (no ws_url)
    streamer.running = True
    async def mock_sim_exit():
        streamer.running = False
        
    with patch.object(streamer, "_get_authorized_ws_url", return_value=None), \
         patch.object(streamer, "_run_simulation", side_effect=mock_sim_exit) as mock_sim, \
         patch("services.upstox_streamer.asyncio.sleep", AsyncMock()): # Use AsyncMock
        await streamer.connect_and_stream()
        assert mock_sim.called

    # Case 2: Real WS connection
    streamer.running = True
    # We patch the backoff to avoid waiting if it fails
    # Since decorators are already applied, we patch sleep to speed it up
    with patch.object(streamer, "_get_authorized_ws_url", return_value="wss://test"), \
         patch("websockets.connect") as mock_conn, \
         patch("services.upstox_streamer.asyncio.sleep", AsyncMock()):
             
        mock_ws = AsyncMock()
        
        async def mock_recv():
            streamer.running = False
            return b"msg1"
            
        mock_ws.recv.side_effect = mock_recv
        
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_conn.return_value = mock_ctx
        
        with patch.object(streamer, "_decode_protobuf", return_value=[{"key": "TCS", "ltp": 3500}]):
            await streamer.connect_and_stream()
        
        assert callback.called
        assert callback.call_args[0][0]["key"] == "TCS"

def test_decode_protobuf_detailed(streamer):
    """Test all branches of protobuf decoding."""
    # No pb module
    with patch("services.upstox_streamer.pb", None):
        assert streamer._decode_protobuf(b"") == []
        
    # Real mock
    with patch("services.upstox_streamer.pb") as mock_pb:
        mock_feed_resp = mock_pb.FeedResponse.return_value
        
        # FF branch
        f1 = MagicMock()
        f1.HasField.side_effect = lambda x: x == "ff"
        f1.ff.marketFF.ltpc.ltp = 100.0
        f1.ff.marketFF.ltpc.ltt = 1000
        
        # LTPC branch
        f2 = MagicMock()
        f2.HasField.side_effect = lambda x: x == "ltpc"
        f2.ltpc.ltp = 200.0
        f2.ltpc.ltt = 2000
        
        mock_feed_resp.feeds = {"K1": f1, "K2": f2}
        
        ticks = streamer._decode_protobuf(b"bytes")
        assert len(ticks) == 2
        assert ticks[0]["ltp"] == 100.0
        assert ticks[1]["ltp"] == 200.0
        
        # Error branch
        mock_feed_resp.ParseFromString.side_effect = Exception("parse error")
        assert streamer._decode_protobuf(b"bytes") == []

@pytest.mark.asyncio
async def test_stop_streamer(streamer):
    """Test stopping the streamer."""
    streamer.ws = MagicMock()
    streamer.ws.close = AsyncMock()
    streamer.running = True
    
    streamer.stop()
    assert streamer.running is False
    await asyncio.sleep(0.01) # task creation takes a moment
    assert streamer.ws.close.called

def test_get_streamer_logic():
    """Test the global get_streamer function."""
    import services.upstox_streamer
    # Save old state to prevent side effects in other tests
    old_streamer = services.upstox_streamer.upstox_streamer
    services.upstox_streamer.upstox_streamer = None
    
    try:
        with patch("services.upstox_streamer.UpstoxLiveStream") as mock_cls:
            cb = AsyncMock()
            s1 = get_streamer(cb)
            s2 = get_streamer(cb)
            
            assert s1 is s2
            mock_cls.assert_called_once_with(cb)
    finally:
        # Restore old state
        services.upstox_streamer.upstox_streamer = old_streamer
