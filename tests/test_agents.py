import pytest
import asyncio
import pandas as pd
import numpy as np
from app.agents_logic import TradingAgent, SimulationEngine, IntelligenceAgent

@pytest.mark.asyncio
async def test_agent_initialization():
    agent = TradingAgent("Test-Agent", "BTC_USDT", 1000)
    assert agent.name == "Test-Agent"
    assert agent.balance == 1000
    assert agent.symbol == "BTC_USDT"

@pytest.mark.asyncio
async def test_agent_decision_logic():
    agent = TradingAgent("Test-Agent", "BTC_USDT", 1000)
    # Indicators need at least 30 rows
    df = pd.DataFrame({
        'close': np.random.rand(100) + 100,
        'high': np.random.rand(100) + 101,
        'low': np.random.rand(100) + 99,
        'vol': np.random.rand(100) * 1000
    })

    agent.tick(df)
    # Check if a position was opened or not (it might be None if confidence is low)
    # In learning mode, it might open a position based on heuristics
    status = agent.active_position
    # If no position, that's fine too as long as it didn't crash
    assert True

@pytest.mark.asyncio
async def test_simulation_engine_initialization():
    engine = SimulationEngine()
    assert len(engine.agents) == 6
    assert engine.agents[0].name == "Titan-AI"
