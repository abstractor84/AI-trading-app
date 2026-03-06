import asyncio
import json
import math
from services.state import AppState
from services.market_phase import market_phase_svc

async def test_state_json():
    state = AppState()
    
    # Simulate some state
    phase_ctx = market_phase_svc.get_phase_context()
    
    payload = {
        "type": "state_update",
        "capital": state.capital,
        "max_loss": state.max_loss_per_trade,
        "open_trades": state.open_trades,
        "closed_trades": state.closed_trades,
        "global_context": state.global_context,
        "market_phase": phase_ctx,
        "search_engine": getattr(state, 'search_engine', 'ddgs'),
        "data_provider": getattr(state, 'data_provider', 'yfinance'),
        "search_fallback": getattr(state, 'search_fallback', False),
        "auto_refresh": getattr(state, 'auto_refresh', True),
        "ai_provider": getattr(state, 'ai_provider', 'google'),
        "ai_model": getattr(state, 'ai_model', 'gemini-2.5-flash'),
        "ai_fallback": getattr(state, 'ai_fallback', True),
        "connection_status": state.connection_status if hasattr(state, 'connection_status') else None
    }
    
    try:
        json_str = json.dumps(payload)
        print("Initial state JSON serialization successful.")
        # But wait, python's json.dumps allows NaN without errors!
        # Let's check if there's any literal 'NaN' string in the output
        if 'NaN' in json_str or 'Infinity' in json_str:
            print("WARNING: 'NaN' or 'Infinity' found in JSON string. JS will fail to parse this!")
        else:
            print("JSON looks clean for JS.")
    except Exception as e:
        print(f"Failed to dump JSON: {e}")

asyncio.run(test_state_json())
