import pytest
import asyncio
import numpy as np
from app.agents_logic import IntelligenceAgent, SimulationEngine

@pytest.mark.asyncio
async def test_agent_initialization():
    agent = IntelligenceAgent("Test-Agent", "Test-Strategy", 1000)
    assert agent.name == "Test-Agent"
    assert agent.balance == 1000
    assert agent.is_active is True

@pytest.mark.asyncio
async def test_agent_decision_logic():
    agent = IntelligenceAgent("Test-Agent", "Test-Strategy", 1000)
    # Mock data
    md = {
        "closes": np.array([100.0]*100),
        "highs": np.array([105.0]*100),
        "lows": np.array([95.0]*100)
    }
    decision = agent.decide_trade(md)
    # Just verify it returns something (either None or an Entry dict)
    # The exact logic depends on the ML ensemble output which can be 0 or 100
    if decision:
        assert "action" in decision
        assert "confidence" in decision

@pytest.mark.asyncio
async def test_simulation_engine_initialization():
    engine = SimulationEngine()
    assert len(engine.agents) == 5
    assert engine.agents[0].name == "Titan-AI"
