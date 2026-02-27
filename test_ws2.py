import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        msg = await ws.recv()
        data = json.loads(msg)
        print("Received keys:", list(data.keys()))
        print("Search engine:", data.get('search_engine'))
        print("Data provider:", data.get('data_provider'))

asyncio.run(test())
