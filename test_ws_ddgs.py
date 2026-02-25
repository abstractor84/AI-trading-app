import asyncio
from services.state import AppState
from main import perform_ai_screening

async def main():
    import services.state as state_mod
    state_mod.state.search_engine = "ddgs"
    # mock broadcast
    import main
    async def mock_broadcast(*args, **kwargs):
        pass
    main.manager.broadcast = mock_broadcast
    
    await perform_ai_screening("s1", None)
    print("Signals len:", len(state_mod.state.ai_signals))
    if state_mod.state.ai_signals:
         print("Headlines from first signal:", state_mod.state.ai_signals[0]["headlines"])

asyncio.run(main())
