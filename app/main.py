from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import asyncio
import httpx
from contextlib import asynccontextmanager
from typing import List, Dict
from app.agents_logic import SimulationEngine, IntelligenceAgent
from app.utils.mexc_api import fetch_mexc_kline, market_scanner
from app.utils.notifier import send_telegram_msg, format_notification
import pandas as pd

# Initialize Global Components
engine = SimulationEngine()
trade_history = []
market_opportunities = []
global_scanner = IntelligenceAgent("Global-Radar")

async def perform_market_sweep():
    """Heavy-duty scanner evaluating 50+ symbols using ML."""
    global market_opportunities
    try:
        symbols = await market_scanner()
        target_list = symbols[:60] # Scan top 60 projects
        new_opps = []

        # Parallel scanning in chunks to avoid rate limits
        chunk_size = 10
        for i in range(0, len(target_list), chunk_size):
            chunk = target_list[i:i+chunk_size]
            tasks = [fetch_mexc_kline(sym, interval="5m", limit=300) for sym in chunk]
            results = await asyncio.gather(*tasks)

            for sym, data in zip(chunk, results):
                if data:
                    df = pd.DataFrame(data)
                    # For performance, we train only if not trained or periodically
                    if not global_scanner.is_trained:
                        global_scanner.train(df)

                    pred = global_scanner.predict(df)
                    if pred["side"] != "NONE":
                        new_opps.append({
                            "symbol": sym, "side": pred["side"],
                            "confidence": pred["confidence"],
                            "price": data[-1]['close']
                        })
            await asyncio.sleep(1) # Breath

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
                asyncio.create_task(perform_market_sweep())

            # Step engine with current opportunities
            events = await engine.step(market_opportunities)
            step_count += 1

            for event in events:
                msg = format_notification(event)
                await send_telegram_msg(msg)

            # Update history
            all_trades = []
            for agent in engine.agents:
                for t in agent.trades:
                    all_trades.append({"agent": agent.name, **t})

            all_trades.sort(key=lambda x: x["timestamp"], reverse=True)
            global trade_history
            trade_history = all_trades[:100]

            # Keep-alive
            if RENDER_URL and step_count % 14 == 0:
                async with httpx.AsyncClient() as client:
                    await client.get(f"{RENDER_URL}/health")

        except Exception as e:
            print(f"Loop error: {e}")

        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(perform_market_sweep())
    sim_task = asyncio.create_task(run_simulation())
    yield
    sim_task.cancel()

app = FastAPI(title="CryptoSafe AI - Aegis Omni V4", lifespan=lifespan)

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
    data = await fetch_mexc_kline(symbol=symbol, interval="5m", limit=100)
    return data

@app.get("/health")
async def health(): return {"status": "Aegis-Omni-Active"}

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
