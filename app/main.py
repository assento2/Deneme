from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import asyncio
import httpx
import gc
from contextlib import asynccontextmanager
from typing import List, Dict
from app.agents_logic import SimulationEngine
from app.utils.mexc_api import fetch_mexc_kline, market_scanner
from app.utils.notifier import send_telegram_msg, format_notification
import pandas as pd

# Initialize Engine with Shared Brain
engine = SimulationEngine()
trade_history = []
market_opportunities = []

async def perform_market_sweep():
    """Sequential scanner to minimize memory spikes."""
    global market_opportunities
    try:
        symbols = await market_scanner()
        target_list = symbols[:40] # Reduced from 60 to save RAM
        new_opps = []

        # Process in small serial chunks to avoid OOM
        chunk_size = 5
        for i in range(0, len(target_list), chunk_size):
            chunk = target_list[i:i+chunk_size]
            for sym in chunk:
                try:
                    data = await fetch_mexc_kline(sym, interval="5m", limit=200)
                    if data:
                        df = pd.DataFrame(data)
                        pred = engine.brain.predict(df)
                        if pred["side"] != "NONE":
                            new_opps.append({
                                "symbol": sym, "side": pred["side"],
                                "confidence": pred["confidence"],
                                "price": data[-1]['close']
                            })
                    del data # Explicit cleanup
                except: continue
            gc.collect() # Cleanup after each chunk
            await asyncio.sleep(0.5)

        market_opportunities = sorted(new_opps, key=lambda x: x["confidence"], reverse=True)
    except Exception as e:
        print(f"Sweep Error: {e}")

async def run_simulation():
    RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")
    step_count = 0

    while True:
        try:
            # Sweep every 5 minutes
            if step_count % 5 == 0:
                await perform_market_sweep() # Sequential wait to prevent spikes

            events = await engine.step(market_opportunities)
            step_count += 1

            for event in events:
                msg = format_notification(event)
                await send_telegram_msg(msg)

            all_trades = []
            for agent in engine.agents:
                for t in agent.trades:
                    all_trades.append({"agent": agent.name, **t})

            all_trades.sort(key=lambda x: x["timestamp"], reverse=True)
            global trade_history
            trade_history = all_trades[:50] # Reduced history length

            if RENDER_URL and step_count % 14 == 0:
                async with httpx.AsyncClient() as client:
                    await client.get(f"{RENDER_URL}/health")

        except Exception as e:
            print(f"Loop error: {e}")

        gc.collect()
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(run_simulation())
    yield

app = FastAPI(title="Aegis Omni V4 (RAM Optimized)", lifespan=lifespan)

@app.get("/agents")
async def get_agents(): return engine.get_status()

@app.get("/trades")
async def get_trades(): return trade_history

@app.get("/opportunities")
async def get_opportunities(): return market_opportunities

@app.get("/agent/{agent_name}/trades")
async def get_agent_trades(agent_name: str):
    for agent in engine.agents:
        if agent.name.lower() == agent_name.lower():
            return agent.trades[::-1]
    raise HTTPException(status_code=404)

@app.get("/kline")
async def get_kline(symbol: str):
    return await fetch_mexc_kline(symbol=symbol, interval="5m", limit=100)

@app.get("/health")
async def health(): return {"status": "optimized", "mem": "eco"}

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
