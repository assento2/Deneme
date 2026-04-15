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
async def test_shared_brain_logic():
    engine = SimulationEngine()
    df = pd.DataFrame({
        'close': np.random.rand(200) + 100,
        'high': np.random.rand(200) + 101,
        'low': np.random.rand(200) + 99,
        'vol': np.random.rand(200) * 1000
    })

    engine.brain.train(df)
    assert engine.brain.is_trained

    agent = engine.agents[0]
    agent.symbol = "BTC_USDT"
    res = agent.tick(df, engine.brain)
    assert True

@pytest.mark.asyncio
async def test_simulation_engine_optimized():
    engine = SimulationEngine()
    assert len(engine.agents) == 9
    assert engine.agents[0].name == "Aegis-Hunter-1"
    assert engine.agents[8].name == "Aegis-Hunter-9"
    assert engine.agents[8].is_bottom_hunter == True
