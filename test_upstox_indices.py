from services.upstox_service import _load_instrument_cache, _instrument_cache
_load_instrument_cache()
indices = {k: v for k, v in _instrument_cache.items() if "NSE_INDEX" in v}
for k in list(indices.keys())[:20]:
    print(k, indices[k])
