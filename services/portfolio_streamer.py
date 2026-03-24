"""
Portfolio Streamer Service for Upstox
======================================
Provides real-time streaming for portfolio updates including:
- Order updates
- Position updates  
- Holding updates
- GTT (Good Till Triggered) updates

Uses the Upstox Portfolio WebSocket API v3.
"""

import asyncio
import json
import logging
import websockets
from websockets.exceptions import ConnectionClosed
import requests
import os
import random
from datetime import datetime
from typing import Callable, Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Configuration
MAX_RECONNECT_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1
MAX_BACKOFF_SECONDS = 60
BACKOFF_MULTIPLIER = 2
JITTER_FACTOR = 0.25

# Update type constants
UPDATE_TYPE_ORDER = "order"
UPDATE_TYPE_POSITION = "position"
UPDATE_TYPE_HOLDING = "holding"
UPDATE_TYPE_GTT = "gtt"


class PortfolioStreamer:
    """
    Manages WebSocket connection for Upstox Portfolio Stream.
    
    Provides real-time updates for orders, positions, holdings, and GTT orders.
    """
    
    def __init__(
        self,
        callback: Callable = None,
        order_update: Callable[[Dict[str, Any]], None] = None,
        position_update: Callable[[Dict[str, Any]], None] = None,
        holding_update: Callable[[Dict[str, Any]], None] = None,
        gtt_update: Callable[[Dict[str, Any]], None] = None
    ):
        """
        Initialize the Portfolio Streamer.
        
        Args:
            callback: General callback for all message types
            order_update: Callback for order updates
            position_update: Callback for position updates
            holding_update: Callback for holding updates
            gtt_update: Callback for GTT order updates
        """
        self.callback = callback
        self.order_update = order_update
        self.position_update = position_update
        self.holding_update = holding_update
        self.gtt_update = gtt_update
        
        # Import UpstoxService lazily to avoid circular imports
        from services.upstox_service import UpstoxService
        self.upstox_service = UpstoxService()
        
        self.ws = None
        self.running = True
        self.subscribed_types = set()
        
        # Connection state
        self._is_connected = False
        self._reconnect_attempt = 0
        
    @property
    def is_connected(self) -> bool:
        """Check if the streamer is currently connected."""
        return self._is_connected
    
    def _get_authorized_ws_url(self) -> Optional[str]:
        """
        Fetch authorized portfolio WebSocket URL from Upstox API.
        
        Endpoint: POST https://api.upstox.com/v3/feed/portfolio/authorize
        
        Returns:
            WebSocket URL if authorized, None otherwise
        """
        if not self.upstox_service.is_authenticated:
            logger.warning("Cannot get WS URL: Not authenticated")
            return None
        
        url = "https://api.upstox.com/v3/feed/portfolio/authorize"
        
        try:
            # POST request with empty body for portfolio feed authorization
            resp = requests.post(
                url, 
                headers=self.upstox_service._headers(), 
                json={},
                timeout=10
            )
            
            if resp.status_code == 200:
                data = resp.json()
                ws_url = data.get("data", {}).get("authorized_redirect_uri")
                if ws_url:
                    logger.info("Successfully obtained portfolio WebSocket URL")
                    return ws_url
            else:
                logger.error(f"Portfolio WS Auth Error: {resp.status_code} - {resp.text[:200]}")
                self.upstox_service._handle_api_error(resp, "portfolio_ws_authorize")
                
        except requests.exceptions.Timeout:
            logger.error("Portfolio WS authorization timed out")
        except requests.exceptions.ConnectionError:
            logger.error("Portfolio WS authorization connection error")
        except Exception as e:
            logger.error(f"Portfolio WS authorization exception: {e}")
        
        return None
    
    def subscribe(self, update_types: List[str]) -> None:
        """
        Subscribe to specific update types.
        
        Args:
            update_types: List of update types to subscribe to.
                         Options: "order", "position", "holding", "gtt", "all"
        """
        if isinstance(update_types, str):
            update_types = [update_types]
        
        for update_type in update_types:
            if update_type == "all":
                self.subscribed_types.update([
                    UPDATE_TYPE_ORDER,
                    UPDATE_TYPE_POSITION,
                    UPDATE_TYPE_HOLDING,
                    UPDATE_TYPE_GTT
                ])
            else:
                self.subscribed_types.add(update_type.lower())
        
        logger.info(f"Subscribed to portfolio update types: {self.subscribed_types}")
        
        # If already connected, send subscription request
        if self.ws and self._is_connected:
            asyncio.create_task(self._send_subscribe())
    
    def unsubscribe(self, update_types: List[str]) -> None:
        """
        Unsubscribe from specific update types.
        
        Args:
            update_types: List of update types to unsubscribe from.
        """
        if isinstance(update_types, str):
            update_types = [update_types]
        
        for update_type in update_types:
            self.subscribed_types.discard(update_type.lower())
        
        logger.info(f"Unsubscribed from portfolio update types: {self.subscribed_types}")
    
    async def _send_subscribe(self) -> None:
        """Send subscription request for the configured update types."""
        if not self.subscribed_types:
            logger.warning("No update types to subscribe to")
            return
        
        # Map update types to Upstox subscription format
        subscription_data = {}
        
        if UPDATE_TYPE_ORDER in self.subscribed_types:
            subscription_data["order"] = True
        if UPDATE_TYPE_POSITION in self.subscribed_types:
            subscription_data["position"] = True
        if UPDATE_TYPE_HOLDING in self.subscribed_types:
            subscription_data["holding"] = True
        if UPDATE_TYPE_GTT in self.subscribed_types:
            subscription_data["gtt"] = True
        
        req = {
            "guid": f"portfolio-sub-{datetime.now().timestamp()}",
            "method": "sub",
            "data": subscription_data
        }
        
        try:
            await self.ws.send(json.dumps(req))
            logger.info(f"Portfolio subscription sent: {subscription_data}")
        except Exception as e:
            logger.error(f"Failed to send portfolio subscription: {e}")
    
    def _handle_message(self, message: Any) -> None:
        """
        Process incoming portfolio messages.
        
        Args:
            message: The raw message from WebSocket
        """
        try:
            # Handle binary/protobuf messages if needed
            if isinstance(message, bytes):
                # Try to parse as JSON first (some messages are text)
                try:
                    message = json.loads(message.decode('utf-8'))
                except:
                    # Binary protobuf message - log that we received it
                    logger.debug(f"Received binary portfolio message: {len(message)} bytes")
                    return
            elif isinstance(message, str):
                try:
                    message = json.loads(message)
                except:
                    logger.warning(f"Failed to parse portfolio message: {message[:100]}")
                    return
            
            # Extract message type and data
            if not isinstance(message, dict):
                return
                
            msg_type = message.get("type", "")
            data = message.get("data", {})
            
            # Route to appropriate callback
            if msg_type == "order" or "order" in str(message).lower():
                if self.order_update:
                    self.order_update(data)
                if self.callback:
                    self.callback({"type": "order", "data": data})
                    
            elif msg_type == "position" or "position" in str(message).lower():
                if self.position_update:
                    self.position_update(data)
                if self.callback:
                    self.callback({"type": "position", "data": data})
                    
            elif msg_type == "holding" or "holding" in str(message).lower():
                if self.holding_update:
                    self.holding_update(data)
                if self.callback:
                    self.callback({"type": "holding", "data": data})
                    
            elif msg_type == "gtt" or "gtt" in str(message).lower():
                if self.gtt_update:
                    self.gtt_update(data)
                if self.callback:
                    self.callback({"type": "gtt", "data": data})
            else:
                # Generic handling for unknown message types
                logger.debug(f"Received portfolio message: {msg_type}")
                if self.callback:
                    self.callback({"type": msg_type, "data": data})
                    
        except Exception as e:
            logger.error(f"Error handling portfolio message: {e}")
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff with jitter for reconnection.
        
        Args:
            attempt: The current reconnection attempt number
            
        Returns:
            Backoff time in seconds
        """
        backoff = INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** attempt)
        backoff = min(backoff, MAX_BACKOFF_SECONDS)
        jitter = backoff * random.uniform(0, JITTER_FACTOR)
        return backoff + jitter
    
    async def connect_and_stream(self) -> None:
        """
        Main connection loop with automatic reconnection.
        
        Connects to the portfolio WebSocket and processes incoming messages.
        """
        while self.running:
            try:
                ws_url = self._get_authorized_ws_url()
                
                if not ws_url:
                    logger.warning("Failed to get authorized portfolio WebSocket URL. Retrying...")
                    await asyncio.sleep(5)
                    continue
                
                async with websockets.connect(ws_url) as websocket:
                    self.ws = websocket
                    self._is_connected = True
                    self._reconnect_attempt = 0
                    logger.info("Connected to Upstox Portfolio WebSocket")
                    
                    # Send subscription if we have any
                    if self.subscribed_types:
                        await self._send_subscribe()
                    
                    # Main message loop
                    while self.running:
                        try:
                            message = await websocket.recv()
                            self._handle_message(message)
                        except ConnectionClosed:
                            logger.warning("Portfolio WebSocket disconnected")
                            break
                        except Exception as e:
                            logger.error(f"Portfolio message loop error: {e}")
                            await asyncio.sleep(1)
                            
            except Exception as e:
                if not self.running:
                    break
                    
                self._is_connected = False
                
                if self._reconnect_attempt < MAX_RECONNECT_RETRIES:
                    backoff_time = self._calculate_backoff(self._reconnect_attempt)
                    self._reconnect_attempt += 1
                    logger.warning(
                        f"Portfolio WS connection failed "
                        f"(attempt {self._reconnect_attempt}/{MAX_RECONNECT_RETRIES}). "
                        f"Retrying in {backoff_time:.1f}s..."
                    )
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(
                        f"Max portfolio reconnection attempts ({MAX_RECONNECT_RETRIES}) reached. "
                        f"Stopping. Error: {e}"
                    )
                    break
    
    def stop(self) -> None:
        """
        Stop the portfolio streamer gracefully.
        """
        logger.info("Stopping Portfolio Streamer...")
        self.running = False
        self._is_connected = False
        
        if self.ws:
            try:
                asyncio.create_task(self.ws.close())
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
    
    def restart(self) -> None:
        """
        Restart the portfolio streamer (e.g., after token update).
        """
        logger.info("Restarting Portfolio Streamer...")
        self.stop()
        self.running = True
        self._is_connected = False
        self._reconnect_attempt = 0


# Global instance
_portfolio_streamer = None


def get_portfolio_streamer(
    callback: Callable = None,
    order_update: Callable[[Dict[str, Any]], None] = None,
    position_update: Callable[[Dict[str, Any]], None] = None,
    holding_update: Callable[[Dict[str, Any]], None] = None,
    gtt_update: Callable[[Dict[str, Any]], None] = None
) -> PortfolioStreamer:
    """
    Get or create the global PortfolioStreamer instance.
    
    Args:
        callback: General callback for all message types
        order_update: Callback for order updates
        position_update: Callback for position updates  
        holding_update: Callback for holding updates
        gtt_update: Callback for GTT order updates
        
    Returns:
        The PortfolioStreamer instance
    """
    global _portfolio_streamer
    
    if _portfolio_streamer is None:
        _portfolio_streamer = PortfolioStreamer(
            callback=callback,
            order_update=order_update,
            position_update=position_update,
            holding_update=holding_update,
            gtt_update=gtt_update
        )
    else:
        # Update callbacks if provided
        if callback:
            _portfolio_streamer.callback = callback
        if order_update:
            _portfolio_streamer.order_update = order_update
        if position_update:
            _portfolio_streamer.position_update = position_update
        if holding_update:
            _portfolio_streamer.holding_update = holding_update
        if gtt_update:
            _portfolio_streamer.gtt_update = gtt_update
    
    return _portfolio_streamer


def stop_portfolio_streamer() -> None:
    """Stop the global portfolio streamer."""
    global _portfolio_streamer
    if _portfolio_streamer:
        _portfolio_streamer.stop()
        _portfolio_streamer = None
