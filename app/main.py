from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import asyncio
import httpx
import gc
from datetime import datetime, time, timedelta
from contextlib import asynccontextmanager
from typing import List, Dict
from app.agents_logic import SimulationEngine
from app.utils.mexc_api import fetch_mexc_kline, market_scanner
from app.utils.notifier import send_telegram_msg, format_notification, format_daily_report
import pandas as pd

# Initialize Engine with Shared Brain
engine = SimulationEngine()
trade_history = []
market_opportunities = []
last_report_date = ""

async def perform_market_sweep():
    """Sequential scanner to minimize memory spikes."""
    global market_opportunities
    try:
        symbols = await market_scanner()
        target_list = symbols[:40]
        new_opps = []

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
                    del data
                except: continue
            gc.collect()
            await asyncio.sleep(0.5)

        market_opportunities = sorted(new_opps, key=lambda x: x["confidence"], reverse=True)
    except Exception as e:
        print(f"Sweep Error: {e}")

async def check_daily_report():
    """Checks if it's 03:00 TR (00:00 UTC) and sends report."""
    global last_report_date
    now = datetime.utcnow()
    # TR 03:00 is UTC 00:00
    if now.hour == 0 and now.minute < 5:
        today_str = now.strftime("%Y-%m-%d")
        if last_report_date != today_str:
            # Generate Stats
            total_trades = 0
            wins = 0
            total_profit = 0.0
            total_balance = 0.0

            # Look back 24h
            cutoff = now - timedelta(days=1)
            for agent in engine.agents:
                total_balance += agent.balance
                for t in agent.trades:
                    t_time = datetime.fromisoformat(t['timestamp'])
                    if t_time > cutoff:
                        total_trades += 1
                        total_profit += t['net_profit_loss']
                        if t['success']: wins += 1

            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            report_stats = {
                "date": today_str,
                "total_trades": total_trades,
                "win_rate": round(win_rate, 1),
                "profit": round(total_profit, 2),
                "total_balance": round(total_balance, 2)
            }

            msg = format_daily_report(report_stats)
            await send_telegram_msg(msg)
            last_report_date = today_str

async def run_simulation():
    RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")
    step_count = 0

    while True:
        try:
            # Daily Report Check
            await check_daily_report()

            # Sweep every 5 minutes
            if step_count % 5 == 0:
                await perform_market_sweep()

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
            trade_history = all_trades[:50]

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

app = FastAPI(title="Aegis Omni V4 (Report Enabled)", lifespan=lifespan)

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
async def health(): return {"status": "optimized", "report": "enabled"}

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
