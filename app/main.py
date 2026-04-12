from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import asyncio
from contextlib import asynccontextmanager
from typing import List
from app.agents_logic import SimulationEngine

# Simulation state
engine = SimulationEngine()
trade_history = []

async def run_simulation():
    while True:
        engine.step()
        for agent in engine.agents:
            if agent.is_active and agent.trades:
                last_trade = agent.trades[-1]
                trade_entry = {
                    "agent": agent.name,
                    "strategy": agent.strategy_name,
                    **last_trade
                }
                trade_history.append(trade_entry)
                if len(trade_history) > 50:
                    trade_history.pop(0)
        await asyncio.sleep(5)

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

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
