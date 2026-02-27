import asyncio
import websockets
import json
import sqlite3

async def test():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Get initial state
        msg = await ws.recv()
        print("Initial:", json.loads(msg).get("search_engine"), json.loads(msg).get("data_provider"))
        
        # Send update
        req = {
            "action": "update_settings",
            "capital": 100000,
            "max_loss": 1000,
            "search_engine": "tavily",
            "data_provider": "upstox",
            "ai_provider": "sambanova",
            "ai_model": "Meta-Llama-3.1-8B-Instruct",
            "auto_refresh": True
        }
        await ws.send(json.dumps(req))
        
        # Get broadcated state update from saving
        msg2 = await ws.recv()
        print("After update:", json.loads(msg2).get("search_engine"), json.loads(msg2).get("data_provider"))

asyncio.run(test())

import sqlite3
db = sqlite3.connect('trading_data.db')
c = db.cursor()
c.execute("SELECT search_engine, data_provider FROM app_settings WHERE id=1")
print("DB:", c.fetchone())
db.close()
