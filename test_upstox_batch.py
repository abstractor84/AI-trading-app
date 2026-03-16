import os
import requests
from services.upstox_service import upstox_client, get_instrument_key
token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()
if token:
    keys = ",".join(filter(None, [get_instrument_key("RELIANCE.NS"), get_instrument_key("TCS.NS")]))
    res = upstox_client.fetch_market_quote(keys)
    print(res)
