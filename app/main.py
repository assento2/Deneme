from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import asyncio
import httpx
from contextlib import asynccontextmanager
from typing import List
from app.agents_logic import SimulationEngine
from app.utils.mexc_api import fetch_mexc_kline

# Simulation state
engine = SimulationEngine()
trade_history = []

async def run_simulation():
    # Render Keep-Alive: Self-ping every 14 minutes to prevent sleep
    last_ping = 0

    while True:
        try:
            current_time = asyncio.get_event_loop().time()
            if current_time - last_ping > 840:
                try:
                    # Self-ping
                    async with httpx.AsyncClient() as client:
                        await client.get("http://localhost:8000/health", timeout=5)
                    last_ping = current_time
                except Exception: pass

            await engine.step()

            # Aggregating trade logs for global view
            all_trades = []
            for agent in engine.agents:
                for t in agent.trades:
                    all_trades.append({"agent": agent.name, "strategy": agent.strategy_name, **t})

            all_trades.sort(key=lambda x: x["timestamp"], reverse=True)
            global trade_history
            trade_history = all_trades[:50]

        except Exception as e:
            print(f"Simulation error: {e}")

        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    sim_task = asyncio.create_task(run_simulation())
    yield
    sim_task.cancel()
    try:
        await sim_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="CryptoSafe AI - Command Center", lifespan=lifespan)

@app.get("/agents")
async def get_agents():
    return engine.get_status()

@app.get("/trades")
async def get_trades():
    return trade_history

@app.get("/agent/{agent_name}/trades")
async def get_agent_trades(agent_name: str):
    search_name = agent_name.lower().strip()
    for agent in engine.agents:
        if agent.name.lower() == search_name:
            return agent.trades[::-1]
    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

@app.get("/kline")
async def get_kline(symbol: str = "BTC_USDT"):
    data = await fetch_mexc_kline(symbol=symbol, interval="5m", limit=200)
    if not data:
        raise HTTPException(status_code=503, detail="Market data unavailable")
    return data

@app.get("/health")
async def health():
    return {"status": "healthy", "agents_active": len(engine.agents)}

# Simplified scan endpoint for frontend
class ScanRequest(BaseModel):
    symbol: str

@app.post("/scan-project")
async def scan_project(req: ScanRequest):
    # Simulated project analysis for frontend UI
    import random
    confidence = random.randint(30, 95)
    return {"symbol": req.symbol, "confidence": confidence, "status": "success"}

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
