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
    # Provide 200 rows to satisfy min_train_size (150)
    df = pd.DataFrame({
        'close': np.random.rand(200) + 100,
        'high': np.random.rand(200) + 101,
        'low': np.random.rand(200) + 99,
        'vol': np.random.rand(200) * 1000
    })

    # Train the shared brain
    engine.brain.train(df)
    assert engine.brain.is_trained

    agent = engine.agents[0]
    agent.symbol = "BTC_USDT"
    res = agent.tick(df, engine.brain)
    # Just check if it runs without error
    assert True

@pytest.mark.asyncio
async def test_simulation_engine_optimized():
    engine = SimulationEngine()
    assert len(engine.agents) == 8
    assert engine.agents[0].name == "Aegis-Hunter-1"
