import asyncio
import json
import logging
import backoff
import websockets
from websockets.exceptions import ConnectionClosed
import requests
import os
import random
from datetime import datetime

import yfinance as yf
from services.market_phase import market_phase_svc
from services.upstox_service import UpstoxService, get_symbol_from_key

# Import compiled protobuf
try:
    import services.MarketDataFeedV3_pb2 as pb
except ImportError:
    pass

logger = logging.getLogger(__name__)

class UpstoxLiveStream:
    """
    Subscribes to Upstox v3 Market Data Feed and decodes protobuf ticks.
    Broadcasts real-time ticks to the frontend via the provided callback.
    Includes a simulation mode if the UPSTOX_ACCESS_TOKEN is missing.
    """
    def __init__(self, callback):
        self.callback = callback
        self.upstox_service = UpstoxService()
        self.ws = None
        self.running = True
        self.subscribed_keys = set()
        
    def _get_authorized_ws_url(self):
        """Fetches the authorized WebSocket URL from Upstox REST API (V3)."""
        if not self.upstox_service.is_authenticated:
            return None
            
        url = "https://api.upstox.com/v3/feed/market-data-feed/authorize"
        try:
            resp = requests.get(url, headers=self.upstox_service._headers(), timeout=5)
            if resp.status_code == 200:
                return resp.json().get("data", {}).get("authorized_redirect_uri")
            else:
                logger.error(f"Upstox WS Auth Error: {resp.text}")
        except Exception as e:
            logger.error(f"Upstox WS Auth Exception: {e}")
        return None

    def subscribe(self, instrument_keys):
        """Adds instrument keys to the subscription list and sends a subscribe command if connected."""
        if isinstance(instrument_keys, str):
            instrument_keys = [instrument_keys]
            
        for key in instrument_keys:
            self.subscribed_keys.add(key)
            
        if self.ws and not self.ws.closed:
            asyncio.create_task(self._send_subscription_request(instrument_keys))
            
    async def _send_subscription_request(self, keys):
        # Ensure all keys use pipe separator for subscription
        std_keys = [k.replace(":", "|") for k in keys]
        req = {
            "guid": "sub-1",
            "method": "sub",
            "data": {
                "mode": "full", # 'full' gives detailed depth and ltp
                "instrumentKeys": std_keys
            }
        }
        await self.ws.send(json.dumps(req).encode('utf-8'))

    def _decode_protobuf(self, buffer):
        """Decode the raw binary protobuf message into a Python dict."""
        try:
            feed_response = pb.FeedResponse()
            feed_response.ParseFromString(buffer)
            
            ticks = []
            for instrument_key, feed in feed_response.feeds.items():
                # We extract the FullFeed or LTPC feed
                if feed.HasField('ff'):
                    meta = feed.ff.marketFF.ltpc
                    ltp = meta.ltp
                    ticks.append({"key": instrument_key, "ltp": ltp, "ltt": meta.ltt})
                elif feed.HasField('ltpc'):
                    ltp = feed.ltpc.ltp
                    ticks.append({"key": instrument_key, "ltp": ltp, "ltt": feed.ltpc.ltt})
            return ticks
        except Exception as e:
            logger.error(f"Protobuf decode error: {e}")
            return []

    async def connect_and_stream(self):
        """Main loop that connects, subscribes, and listens for data."""
        from services.market_phase import market_phase_svc
        
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
                    
                    if self.subscribed_keys:
                        await self._send_subscription_request(self.subscribed_keys)
                        
                    while self.running:
                        try:
                            message = await websocket.recv()
                            ticks = self._decode_protobuf(message)
                            if ticks:
                                for tick in ticks:
                                    await self.callback(tick)
                        except ConnectionClosed:
                            logger.warning("Upstox WS disconnected. Reconnecting...")
                            break
                        except Exception as e:
                            logger.error(f"Upstox WS loop error: {e}")
                            await asyncio.sleep(2) # Prevent spin
            except Exception as e:
                if not self.running: break
                logger.error(f"WS Outer Loop Error: {e}")
                await asyncio.sleep(5)

    async def _run_simulation(self):
        """Simulates price ticks for the UI when real auth is unavailable."""
        if os.getenv("ENABLE_SIMULATION", "false").lower() != "true":
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
                    if not last_prices.get(key):
                        try:
                            from services.upstox_service import get_symbol_from_key
                            symbol = get_symbol_from_key(key)
                            ticker_name = f"{symbol}.NS" if symbol else key
                            data = yf.download(ticker_name, period="1d", progress=False)
                            if not data.empty:
                                last_prices[key] = float(data['Close'].iloc[-1])
                        except: pass
                    
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
        if self.ws:
            asyncio.create_task(self.ws.close())

    def restart(self):
        """Force a restart of the stream (e.g. after token update)."""
        logger.info("Restarting Upstox Live Stream...")
        self.stop()
        self.running = True

    def restart(self):
        """Force a restart of the stream (e.g. after token update)."""
        logger.info("Restarting Upstox Live Stream...")
        self.stop()
        self.running = True

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
    return upstox_streamer
