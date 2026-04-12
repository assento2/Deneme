from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
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
    # Render Keep-Alive: Self-ping every 14 minutes
    last_ping = 0
    while True:
        current_time = asyncio.get_event_loop().time()
        # Self-ping to prevent sleep (Render free tier timeout is 15 mins)
        if current_time - last_ping > 840: # 14 minutes
            try:
                port = int(os.getenv("PORT", 8000))
                async with httpx.AsyncClient() as client:
                    await client.get(f"http://localhost:{port}/health", timeout=5)
                last_ping = current_time
            except Exception:
                pass

        await engine.step()
        for agent in engine.agents:
            if agent.is_active and agent.trades:
                last_trade = agent.trades[-1]
                # Avoid duplicate entries in global trade history
                if not trade_history or (trade_history[-1]["agent"] != agent.name or trade_history[-1]["timestamp"] != last_trade["timestamp"]):
                    trade_entry = {
                        "agent": agent.name,
                        "strategy": agent.strategy_name,
                        **last_trade
                    }
                    trade_history.append(trade_entry)
                    if len(trade_history) > 50:
                        trade_history.pop(0)
        await asyncio.sleep(15) # Longer interval for realism with 5m data

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    sim_task = asyncio.create_task(run_simulation())
    yield
    # Shutdown
    sim_task.cancel()
    try:
        await sim_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="CryptoSafe AI - Agent Command Center", lifespan=lifespan)

# Load ML model
MODEL_PATH = "app/models/model.joblib"
SCALER_PATH = "app/models/scaler.joblib"
FEATURES_PATH = "app/models/features.joblib"

model = None
scaler = None
features = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

class TransactionData(BaseModel):
    avg_min_sent: float
    avg_min_received: float
    time_diff: float
    unique_received_from: int
    min_val_received: float
    max_val_received: float
    avg_val_received: float
    min_val_sent: float
    avg_val_sent: float
    total_transactions: int
    total_ether_received: float
    total_ether_balance: float

@app.post("/predict")
async def predict(data: TransactionData):
    if model is None:
        raise HTTPException(status_code=503, detail="ML Model not available")
    try:
        input_data = pd.DataFrame([[
            data.avg_min_sent, data.avg_min_received, data.time_diff,
            data.unique_received_from, data.min_val_received, data.max_val_received,
            data.avg_val_received, data.min_val_sent, data.avg_val_sent,
            data.total_transactions, data.total_ether_received, data.total_ether_balance
        ]], columns=features)
        scaled_data = scaler.transform(input_data)
        probs = model.predict_proba(scaled_data)[0]
        reliability_score = probs[0]
        is_fraud = bool(model.predict(scaled_data)[0])
        return {"reliability_score": float(reliability_score), "is_fraud": is_fraud, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def get_agents():
    return engine.get_status()

@app.get("/trades")
async def get_trades():
    return trade_history[::-1]

@app.get("/agent/{agent_name}/trades")
async def get_agent_trades(agent_name: str):
    for agent in engine.agents:
        if agent.name == agent_name:
            return agent.trades[::-1]
    raise HTTPException(status_code=404, detail="Agent not found")

@app.get("/kline")
async def get_kline(symbol: str = "BTC_USDT"):
    data = await fetch_mexc_kline(symbol=symbol, limit=200)
    if not data:
        raise HTTPException(status_code=503, detail="Market data unavailable")
    return data

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for deployment flexibility
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
