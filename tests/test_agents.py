import pytest
import asyncio
import pandas as pd
import numpy as np
from app.agents_logic import TradingAgent, SimulationEngine, IntelligenceAgent

@pytest.mark.asyncio
async def test_agent_initialization():
    agent = TradingAgent("Test-Agent", 1000)
    assert agent.name == "Test-Agent"
    assert agent.balance == 1000
    assert agent.symbol == "WAITING..."

@pytest.mark.asyncio
async def test_agent_decision_logic():
    agent = TradingAgent("Test-Agent", 1000)
    agent.assign_to("BTC_USDT")
    df = pd.DataFrame({
        'close': np.random.rand(100) + 100,
        'high': np.random.rand(100) + 101,
        'low': np.random.rand(100) + 99,
        'vol': np.random.rand(100) * 1000
    })

    agent.tick(df)
    assert agent.symbol == "BTC_USDT"

@pytest.mark.asyncio
async def test_simulation_engine_initialization():
    engine = SimulationEngine()
    assert len(engine.agents) == 10
    assert engine.agents[0].name == "Aegis-Hunter-1"
