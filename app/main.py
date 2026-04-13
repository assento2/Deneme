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
from app.utils.notifier import send_telegram_msg, format_notification

# Initialize Simulation state
engine = SimulationEngine()
trade_history = []
market_opportunities = []

async def perform_market_sweep():
    """Scans top projects and performs ML inference for global opportunities."""
    from app.utils.mexc_api import market_scanner
    from app.agents_logic import IntelligenceAgent
    from datetime import datetime
    import pandas as pd

    symbols = await market_scanner()
    top_10 = symbols[:10]
    scanner = IntelligenceAgent("SweepScanner")

    new_opps = []
    for sym in top_10:
        try:
            data = await fetch_mexc_kline(sym, interval="5m", limit=300)
            if data:
                df = pd.DataFrame(data)
                scanner.train(df)
                pred = scanner.predict(df)
                if pred["side"] != "NONE":
                    new_opps.append({
                        "symbol": sym,
                        "side": pred["side"],
                        "confidence": pred["confidence"],
                        "timestamp": datetime.now().isoformat()
                    })
        except: continue

    global market_opportunities
    market_opportunities = sorted(new_opps, key=lambda x: x["confidence"], reverse=True)

async def run_simulation():
    """Background task to tick agents and update logs."""
    # Self-keep-alive mechanism
    RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")
    step_count = 0

    while True:
        try:
            # Global Market Sweep every 15 mins (approx 15 steps of 1m)
            if step_count % 15 == 0:
                asyncio.create_task(perform_market_sweep())

            events = await engine.step()
            step_count += 1

            # Send notifications
            for event in events:
                msg = format_notification(event)
                await send_telegram_msg(msg)

            # Update global history from engine
            all_trades = []
            for agent in engine.agents:
                for t in agent.trades:
                    all_trades.append({"agent": agent.name, "strategy": agent.strategy_name, **t})

            all_trades.sort(key=lambda x: x["timestamp"], reverse=True)
            global trade_history
            trade_history = all_trades[:100]

            # Ping self to stay awake
            if RENDER_URL:
                try:
                    async with httpx.AsyncClient() as client:
                        await client.get(f"{RENDER_URL}/health")
                except:
                    pass

        except Exception as e:
            print(f"Simulation error: {e}")

        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initial sweep on startup
    asyncio.create_task(perform_market_sweep())
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

@app.get("/opportunities")
async def get_opportunities():
    return market_opportunities

@app.get("/agent/{agent_name}/trades")
async def get_agent_trades(agent_name: str):
    search_name = agent_name.lower().strip()
    for agent in engine.agents:
        if agent.name.lower() == search_name:
            return agent.trades[::-1]
    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

@app.get("/kline")
async def get_kline(symbol: str = "BTC_USDT"):
    data = await fetch_mexc_kline(symbol=symbol, interval="5m", limit=100)
    if not data:
        raise HTTPException(status_code=503, detail="Market data unavailable")
    return data

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agents_active": len(engine.agents),
        "simulation_running": True
    }

class ScanRequest(BaseModel):
    symbol: str

@app.post("/scan-project")
async def scan_project(req: ScanRequest):
    from app.agents_logic import IntelligenceAgent
    import pandas as pd

    scanner = IntelligenceAgent("Scanner")
    data = await fetch_mexc_kline(req.symbol, interval="5m", limit=500)
    if not data:
        return {"symbol": req.symbol, "confidence": 0, "status": "failed", "error": "Symbol not found"}

    df = pd.DataFrame(data)
    scanner.train(df)
    prediction = scanner.predict(df)

    return {
        "symbol": req.symbol,
        "confidence": float(prediction.get("confidence", 0)),
        "side": prediction.get("side", "NONE"),
        "mode": prediction.get("mode", "learning"),
        "status": "success"
    }

# Serve frontend
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
