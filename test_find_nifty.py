from services.upstox_service import _load_instrument_cache, _instrument_cache
_load_instrument_cache()
nifty_keys = {k: v for k, v in _instrument_cache.items() if "NIFTY 50" in k or "NIFTY50" in k}
print("NIFTY 50 keys:", nifty_keys)
