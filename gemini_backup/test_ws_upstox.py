import os
import asyncio
import upstox_client
from upstox_client.feeder.market_data_streamer_v3 import MarketDataStreamerV3
from dotenv import load_dotenv

load_dotenv()

access_token = os.getenv("UPSTOX_ACCESS_TOKEN")

if not access_token:
    print("No access token!")
    exit(1)

def on_open():
    print("WS Opened!")
    streamer.subscribe(["NSE_EQ|INE002A01018"], "full") # RELIANCE

def on_message(message):
    print("WS Message:", message)
    # Just print one and exit
    asyncio.get_event_loop().stop()

def on_error(error):
    print("WS Error:", error)

def on_close(code, reason):
    print("WS Closed:", code, reason)

configuration = upstox_client.Configuration()
configuration.access_token = access_token

streamer = MarketDataStreamerV3(configuration=configuration)
streamer.on("open", on_open)
streamer.on("message", on_message)
streamer.on("error", on_error)
streamer.on("close", on_close)

print("Connecting...")
streamer.connect()
