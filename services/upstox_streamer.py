import asyncio
import json
import logging
import websockets
from websockets.exceptions import ConnectionClosed
import requests
import os
import random
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List, Union

import yfinance as yf
from services.market_phase import market_phase_svc
from services.upstox_service import UpstoxService, get_symbol_from_key
from services.yfinance_helper import fetch_quote_cached, is_market_open
from services.quota_service import quota_svc

# Import compiled protobuf
try:
    import services.MarketDataFeedV3_pb2 as pb
except ImportError:
    pass

logger = logging.getLogger(__name__)


class StreamingMode(str, Enum):
    """Supported streaming modes per Upstox documentation."""
    LTPC = "ltpc"
    FULL = "full"
    OPTION_GREEKS = "option_greeks"
    FULL_D30 = "full_d30"


class ErrorCategory(str, Enum):
    """Error categorization for better debugging and monitoring."""
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    VALIDATION = "validation"
    WEBSOCKET = "websocket"
    PARSING = "parsing"
    UNKNOWN = "unknown"


def categorize_error(exception: Exception, status_code: int = None) -> ErrorCategory:
    """Categorize an error for better debugging and monitoring."""
    if isinstance(exception, requests.exceptions.HTTPError):
        if status_code == 401 or status_code == 403:
            return ErrorCategory.AUTHENTICATION
        elif status_code == 429:
            return ErrorCategory.RATE_LIMIT
        elif status_code == 400 or status_code == 422:
            return ErrorCategory.VALIDATION
    elif isinstance(exception, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return ErrorCategory.NETWORK
    elif isinstance(exception, ConnectionClosed):
        return ErrorCategory.WEBSOCKET
    elif isinstance(exception, (json.JSONDecodeError, Exception)) and "decode" in str(exception).lower():
        return ErrorCategory.PARSING
    return ErrorCategory.UNKNOWN

# Track if we're using yfinance fallback
_using_yfinance_fallback = False

# Exponential backoff configuration for reconnection
MAX_RECONNECT_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1
MAX_BACKOFF_SECONDS = 60
BACKOFF_MULTIPLIER = 2
JITTER_FACTOR = 0.25

class UpstoxLiveStream:
    """
    Subscribes to Upstox v3 Market Data Feed and decodes protobuf ticks.
    Broadcasts real-time ticks to the frontend via the provided callback.
    Includes a simulation mode if the UPSTOX_ACCESS_TOKEN is missing.
    
    Event System:
    - 'open': Called when WebSocket connection is established
    - 'close': Called when WebSocket connection is closed
    - 'message': Called for each received message (tick)
    - 'error': Called when an error occurs
    - 'reconnecting': Called when attempting to reconnect (args: attempt, max_attempts)
    - 'autoReconnectStopped': Called when max reconnection attempts are reached
    
    Streaming Modes (per Upstox documentation):
    - 'ltpc': Last traded price and quantity
    - 'full': Full depth with 5-level bid/ask
    - 'option_greeks': Option greeks data
    - 'full_d30': Full depth with 30-level bid/ask
    """
    
    # Supported event types
    SUPPORTED_EVENTS = ['open', 'close', 'message', 'error', 'reconnecting', 'autoReconnectStopped']
    
    # Valid streaming modes
    VALID_MODES = [mode.value for mode in StreamingMode]
    
    def __init__(self, callback):
        self.callback = callback
        self.upstox_service = UpstoxService()
        self.ws = None
        self.running = True
        self.subscribed_keys = set()
        
        # Current streaming mode (default: full)
        self._current_mode = StreamingMode.FULL.value
        
        # Correlation ID for request tracing
        self._correlation_id = str(uuid.uuid4())
        
        # Event handlers: {event_name: [handler1, handler2, ...]}
        self._event_handlers = {event: [] for event in self.SUPPORTED_EVENTS}

    def on(self, event: str, handler: callable) -> None:
        """
        Register an event handler.
        
        Args:
            event: Event name ('open', 'close', 'message', 'error', 'reconnecting', 'autoReconnectStopped')
            handler: Callback function to handle the event
        
        Raises:
            ValueError: If event name is not supported
        """
        if event not in self.SUPPORTED_EVENTS:
            raise ValueError(f"Unsupported event: {event}. Supported events: {self.SUPPORTED_EVENTS}")
        
        if handler not in self._event_handlers[event]:
            self._event_handlers[event].append(handler)
            logger.debug(f"Registered handler for event: {event}")
    
    def off(self, event: str, handler: callable) -> None:
        """
        Unregister an event handler.
        
        Args:
            event: Event name
            handler: The handler function to remove
        """
        if event in self._event_handlers and handler in self._event_handlers[event]:
            self._event_handlers[event].remove(handler)
            logger.debug(f"Unregistered handler for event: {event}")
    
    def _emit(self, event: str, *args, **kwargs) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            event: Event name
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
        """
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in event handler for '{event}': {e}")
    
    def set_callback(self, callback):
        """Update the callback (e.g. if the connection manager changes)."""
        self.callback = callback
        
    def _get_authorized_ws_url(self):
        """Fetches the authorized WebSocket URL from Upstox REST API (V3)."""
        if not self.upstox_service.is_authenticated:
            logger.warning("Upstox WS Auth: Not authenticated, cannot get WS URL")
            return None
            
        url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
        correlation_id = str(uuid.uuid4())
        
        # Enhanced error logging with request details
        logger.debug(f"Upstox WS Auth: Requesting authorized URL (correlation_id={correlation_id})")
        
        try:
            resp = requests.get(url, headers=self.upstox_service._headers(), timeout=5)
            
            # Log response details for debugging
            logger.debug(
                f"Upstox WS Auth: Response received (correlation_id={correlation_id}, "
                f"status_code={resp.status_code})"
            )
            
            if resp.status_code == 200:
                response_data = resp.json()
                ws_url = response_data.get("data", {}).get("authorized_redirect_uri")
                logger.info(f"Upstox WS Auth: Successfully obtained WebSocket URL (correlation_id={correlation_id})")
                return ws_url
            else:
                # Structured error payload
                error_payload = {
                    "correlation_id": correlation_id,
                    "endpoint": "authorize",
                    "method": "GET",
                    "status_code": resp.status_code,
                    "response_text": resp.text[:500] if resp.text else None,
                    "error_category": ErrorCategory.AUTHENTICATION.value
                }
                logger.error(f"Upstox WS Auth Error: {json.dumps(error_payload)}")
        except requests.exceptions.ConnectionError as e:
            error_payload = {
                "correlation_id": correlation_id,
                "endpoint": "authorize",
                "method": "GET",
                "error_type": "ConnectionError",
                "error_message": str(e),
                "error_category": ErrorCategory.NETWORK.value
            }
            logger.error(f"Upstox WS Auth Connection Error: {json.dumps(error_payload)}")
        except requests.exceptions.Timeout as e:
            error_payload = {
                "correlation_id": correlation_id,
                "endpoint": "authorize",
                "method": "GET",
                "error_type": "Timeout",
                "error_message": str(e),
                "error_category": ErrorCategory.NETWORK.value
            }
            logger.error(f"Upstox WS Auth Timeout: {json.dumps(error_payload)}")
        except Exception as e:
            error_payload = {
                "correlation_id": correlation_id,
                "endpoint": "authorize",
                "method": "GET",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_category": categorize_error(e).value
            }
            logger.error(f"Upstox WS Auth Exception: {json.dumps(error_payload)}")
        return None

    def subscribe(self, instrument_keys, mode: str = None):
        """
        Adds instrument keys to the subscription list and sends a subscribe command if connected.
        
        Args:
            instrument_keys: Single key or list of instrument keys to subscribe
            mode: Streaming mode (optional). Valid modes: 'ltpc', 'full', 'option_greeks', 'full_d30'.
                  Defaults to 'full' if invalid mode provided.
        """
        # Validate and set mode
        if mode is None:
            mode = self._current_mode
        elif mode not in self.VALID_MODES:
            logger.warning(f"Invalid streaming mode '{mode}', defaulting to 'full'. Valid modes: {self.VALID_MODES}")
            mode = StreamingMode.FULL.value
        
        # Update current mode
        self._current_mode = mode
        
        if isinstance(instrument_keys, str):
            instrument_keys = [instrument_keys]
            
        for key in instrument_keys:
            self.subscribed_keys.add(key)
            
        logger.info(f"Subscribing to {len(instrument_keys)} instruments in '{mode}' mode")
        
        if self.ws:
            asyncio.create_task(self._send_subscription_request(instrument_keys, mode))
            
    def unsubscribe(self, instrument_keys):
        """Removes instrument keys from the subscription list and sends an unsubscribe command if connected."""
        if isinstance(instrument_keys, str):
            instrument_keys = [instrument_keys]
            
        # Remove keys from tracking set
        keys_to_remove = set()
        for key in instrument_keys:
            if key in self.subscribed_keys:
                keys_to_remove.add(key)
                self.subscribed_keys.discard(key)
        
        # Send unsubscription request if connected
        if self.ws and keys_to_remove:
            asyncio.create_task(self._send_unsubscription_request(list(keys_to_remove)))
            logger.info(f"Unsubscribed from {len(keys_to_remove)} instruments")
            
    async def _send_unsubscription_request(self, keys):
        """Send unsubscription request to Upstox WebSocket."""
        # Ensure all keys use pipe separator
        std_keys = [k.replace(":", "|") for k in keys]
        req = {
            "guid": f"unsub-{datetime.now().timestamp()}",
            "method": "unsub",
            "data": {
                "mode": "full",
                "instrumentKeys": std_keys
            }
        }
        await self.ws.send(json.dumps(req).encode('utf-8'))
        logger.debug(f"Sent unsubscription request for keys: {std_keys}")
            
    async def _send_subscription_request(self, keys, mode: str = None):
        """
        Send subscription request to Upstox WebSocket.
        
        Args:
            keys: List of instrument keys
            mode: Streaming mode (defaults to current mode)
        """
        if mode is None:
            mode = self._current_mode
        
        # Ensure all keys use pipe separator for subscription
        std_keys = [k.replace(":", "|") for k in keys]
        
        correlation_id = str(uuid.uuid4())
        req = {
            "guid": correlation_id,
            "method": "sub",
            "data": {
                "mode": mode,
                "instrumentKeys": std_keys
            }
        }
        
        # Enhanced logging with correlation ID
        logger.debug(
            f"Sending subscription request (correlation_id={correlation_id}, "
            f"mode={mode}, keys={len(std_keys)})"
        )
        
        await self.ws.send(json.dumps(req).encode('utf-8'))
        
        logger.info(f"Subscribed to {len(std_keys)} instruments in '{mode}' mode (guid={correlation_id})")
    
    def change_mode(self, instrument_keys, new_mode: str):
        """
        Change the streaming mode for already subscribed instruments.
        
        Args:
            instrument_keys: Single key or list of instrument keys to change mode for
            new_mode: New streaming mode ('ltpc', 'full', 'option_greeks', 'full_d30')
        
        Returns:
            bool: True if mode change was initiated, False if invalid mode
        """
        # Validate mode
        if new_mode not in self.VALID_MODES:
            logger.warning(
                f"Invalid mode '{new_mode}' for change_mode. Valid modes: {self.VALID_MODES}"
            )
            return False
        
        if isinstance(instrument_keys, str):
            instrument_keys = [instrument_keys]
        
        logger.info(f"Changing mode to '{new_mode}' for {len(instrument_keys)} instruments")
        
        # Update current mode
        self._current_mode = new_mode
        
        if self.ws:
            asyncio.create_task(self._send_mode_change(instrument_keys, new_mode))
        else:
            logger.warning("Cannot change mode: WebSocket not connected")
        
        return True
    
    async def _send_mode_change(self, keys, new_mode: str):
        """
        Send mode change request to Upstox WebSocket.
        
        Args:
            keys: List of instrument keys
            new_mode: New streaming mode
        """
        # Ensure all keys use pipe separator
        std_keys = [k.replace(":", "|") for k in keys]
        
        correlation_id = str(uuid.uuid4())
        req = {
            "guid": correlation_id,
            "method": "changeMode",
            "data": {
                "mode": new_mode,
                "instrumentKeys": std_keys
            }
        }
        
        # Enhanced logging with correlation ID
        logger.debug(
            f"Sending mode change request (correlation_id={correlation_id}, "
            f"mode={new_mode}, keys={len(std_keys)})"
        )
        
        await self.ws.send(json.dumps(req).encode('utf-8'))
        
        logger.info(f"Mode changed to '{new_mode}' for {len(std_keys)} instruments (guid={correlation_id})")

    def _decode_protobuf(self, buffer):
        """Decode the raw binary protobuf message into a Python dict."""
        correlation_id = str(uuid.uuid4())
        
        try:
            feed_response = pb.FeedResponse()
            feed_response.ParseFromString(buffer)
            
            ticks = []
            for instrument_key, feed in feed_response.feeds.items():
                try:
                    # We extract the FullFeed or LTPC feed
                    # Handle different feed types that Upstox may send
                    # SKEPTIC FIX: Field is 'fullFeed' not 'ff'
                    if feed.HasField('fullFeed'):
                        # fullFeed contains marketFF (market Full Feed) or indexFF
                        if feed.fullFeed.HasField('marketFF'):
                            meta = feed.fullFeed.marketFF.ltpc
                            ltp = meta.ltp
                            ltt = meta.ltt
                        elif feed.fullFeed.HasField('indexFF'):
                            meta = feed.fullFeed.indexFF.ltpc
                            ltp = meta.ltp
                            ltt = meta.ltt
                        else:
                            continue
                        ticks.append({"key": instrument_key, "ltp": ltp, "ltt": ltt})
                    elif feed.HasField('ltpc'):
                        ltp = feed.ltpc.ltp
                        ticks.append({"key": instrument_key, "ltp": ltp, "ltt": feed.ltpc.ltt})
                    else:
                        # Feed exists but doesn't have fullFeed or ltpc fields
                        # This is normal for some feed types, just skip silently
                        logger.debug(f"Feed for {instrument_key} has no fullFeed or ltpc field")
                except Exception as inner_e:
                    # Handle any inner parsing errors for individual feeds
                    # Enhanced error logging with structured payload
                    error_payload = {
                        "correlation_id": correlation_id,
                        "instrument_key": instrument_key,
                        "error_type": type(inner_e).__name__,
                        "error_message": str(inner_e),
                        "error_category": ErrorCategory.PARSING.value
                    }
                    logger.debug(f"Failed to parse feed for {instrument_key}: {json.dumps(error_payload)}")
                    continue
            
            if ticks:
                # SKEPTIC: Log first tick for debugging
                logger.info(f"Decoded {len(ticks)} ticks: {ticks[0] if ticks else 'none'}")
            
            return ticks
        except Exception as e:
            # Enhanced error logging with structured payload
            error_payload = {
                "correlation_id": correlation_id,
                "buffer_length": len(buffer) if buffer else 0,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_category": ErrorCategory.PARSING.value
            }
            logger.warning(f"Protobuf decode error: {json.dumps(error_payload)}")
            return []

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        # Calculate exponential backoff: initial * (multiplier ^ attempt)
        backoff = INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** attempt)
        # Cap at max backoff
        backoff = min(backoff, MAX_BACKOFF_SECONDS)
        # Add jitter to prevent thundering herd
        jitter = backoff * random.uniform(0, JITTER_FACTOR)
        return backoff + jitter
    
    async def connect_and_stream(self):
        """Main loop that connects, subscribes, and listens for data with exponential backoff reconnection."""
        from services.market_phase import market_phase_svc
        
        reconnect_attempt = 0
        
        while self.running: # Allow internal reconnect loop
            try:
                # Market Hour Check
                ctx = market_phase_svc.get_phase_context()
                if not ctx["is_trading_hours"]:
                    logger.debug("Outside trading hours. Streamer sleeping...")
                    await asyncio.sleep(60)
                    continue

                ws_url = self._get_authorized_ws_url()
                
                # Sim Mode Fallback
                if not ws_url:
                    logger.warning("No Upstox WS Token. Falling back to simulated ticking mechanism.")
                    await self._run_simulation()
                    return

                async with websockets.connect(ws_url) as websocket:
                    self.ws = websocket
                    logger.info("Connected to Upstox WebSocket Feed")
                    
                    # Emit 'open' event
                    self._emit('open')
                    
                    # Reset reconnect attempt on successful connection
                    reconnect_attempt = 0
                    
                    if self.subscribed_keys:
                        await self._send_subscription_request(list(self.subscribed_keys))
                        
                    while self.running:
                        try:
                            message = await websocket.recv()
                            ticks = self._decode_protobuf(message)
                            if ticks:
                                for tick in ticks:
                                    await self.callback(tick)
                                    # Emit 'message' event
                                    self._emit('message', tick)
                        except ConnectionClosed:
                            logger.warning("Upstox WS disconnected.")
                            # Emit 'close' event
                            self._emit('close', code=None, reason="Connection closed")
                            break
                        except Exception as e:
                            logger.error(f"Upstox WS loop error: {e}")
                            # Emit 'error' event
                            self._emit('error', e)
                            await asyncio.sleep(2) # Prevent spin
                            
            except Exception as e:
                if not self.running: break
                
                # Emit 'error' event
                self._emit('error', e)
                
                # Handle reconnection with exponential backoff
                if reconnect_attempt < MAX_RECONNECT_RETRIES:
                    backoff_time = self._calculate_backoff(reconnect_attempt)
                    reconnect_attempt += 1
                    logger.warning(
                        f"WS connection failed (attempt {reconnect_attempt}/{MAX_RECONNECT_RETRIES}). "
                        f"Retrying in {backoff_time:.1f}s..."
                    )
                    # Emit 'reconnecting' event
                    self._emit('reconnecting', reconnect_attempt, MAX_RECONNECT_RETRIES)
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(
                        f"Max reconnection attempts ({MAX_RECONNECT_RETRIES}) reached. "
                        f"Stopping reconnection. Error: {e}"
                    )
                    # Emit 'autoReconnectStopped' event
                    self._emit('autoReconnectStopped', MAX_RECONNECT_RETRIES, e)
                    
                    # Fall back to simulation if available
                    if self.subscribed_keys:
                        logger.warning("Falling back to simulation mode.")
                        await self._run_simulation()
                    break

    async def _run_simulation(self):
        """Simulates price ticks for the UI when real auth is unavailable."""
        enable_sim = os.getenv("ENABLE_SIMULATION", "false").lower() == "true" or os.getenv("SIMULATION", "false").lower() == "true"
        if not enable_sim:
            logger.warning("Simulation is disabled by default. Set ENABLE_SIMULATION=true in .env to enable.")
            return

        last_prices = {}
        
        while self.running:
            # Market Hour Check
            ctx = market_phase_svc.get_phase_context()
            if not ctx["is_trading_hours"]:
                await asyncio.sleep(60)
                continue

            for key in list(self.subscribed_keys):
                if key not in last_prices:
                    # 1. Try Upstox REST
                    quote = self.upstox_service.fetch_market_quote(key)
                    if quote and 'data' in quote:
                        last_prices[key] = quote['data'].get(key, {}).get('last_price', 0)
                    
                    # 2. Try yfinance fallback (using symbol from key)
                    # PRIORITY: Upstox first, yfinance only as fallback with quota check
                    if not last_prices.get(key):
                        try:
                            from services.upstox_service import get_symbol_from_key
                            symbol = get_symbol_from_key(key)
                            ticker_name = f"{symbol}.NS" if symbol else key
                            
                            # Check quota before making yfinance call
                            if quota_svc.check_yfinance_quota():
                                # Use cached yfinance helper
                                quote = fetch_quote_cached(ticker_name)
                                if quote and quote.get("price"):
                                    last_prices[key] = quote["price"]
                                    quota_svc.log_yfinance_usage()
                                    global _using_yfinance_fallback
                                    _using_yfinance_fallback = True
                                    logger.debug(f"yfinance fallback for {ticker_name}")
                            else:
                                logger.warning(f"yfinance quota exhausted, cannot fetch {ticker_name}")
                        except Exception as e:
                            logger.debug(f"yfinance fallback error for {key}: {e}")
                    
                    # 3. Final default
                    if not last_prices.get(key):
                        last_prices[key] = 1000.0
                
                # Random walk
                last_prices[key] += random.uniform(-1, 1)
                
                now = datetime.now()
                tick = {
                    "key": key,
                    "ltp": round(last_prices[key], 2),
                    "ltt": int(now.timestamp() * 1000)
                }
                await self.callback(tick)
            await asyncio.sleep(2.0)  # Fire a simulated tick every 2s

    def stop(self):
        self.running = False
        # Unsubscribe from all keys before closing
        if self.subscribed_keys:
            keys_to_unsubscribe = list(self.subscribed_keys.copy())
            if self.ws:
                asyncio.create_task(self._send_unsubscription_request(keys_to_unsubscribe))
        if self.ws:
            asyncio.create_task(self.ws.close())
        # Emit 'close' event for graceful stop
        self._emit('close', code=None, reason="Stopped by user")

    def restart(self):
        """Force a restart of the stream (e.g. after token update)."""
        logger.info("Restarting Upstox Live Stream...")
        self.stop()
        self.running = True

upstox_streamer = None

def get_streamer(callback=None):
    global upstox_streamer
    if upstox_streamer is None:
        # Default callback if none provided
        if callback is None:
            async def default_cb(tick):
                logging.debug(f"Tick received (no handler): {tick}")
            callback = default_cb
        upstox_streamer = UpstoxLiveStream(callback)
    elif callback:
        # Update callback if a new one is provided (e.g. from new ws manager)
        upstox_streamer.set_callback(callback)
        
    return upstox_streamer
