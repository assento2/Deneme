import pytest
from app.agents_logic import IntelligenceAgent, SimulationEngine

def test_agent_initialization():
    agent = IntelligenceAgent("Test-Agent", "Test-Strategy", 1000)
    assert agent.name == "Test-Agent"
    assert agent.balance == 1000
    assert agent.is_active is True

@pytest.mark.asyncio
async def test_agent_execution():
    agent = IntelligenceAgent("Test-Agent", "Test-Strategy", 100)
    # success, amount, profit_loss, entry_price, reasoning, indicators, notifier=None
    await agent.execute_trade(True, 10, 10, 50000, "Test Reason", {"rsi": 50}, None)
    assert agent.balance == 110
    assert len(agent.trades) == 1
    assert agent.trades[0]["reasoning"] == "Test Reason"

@pytest.mark.asyncio
async def test_engine_init():
    engine = SimulationEngine()
    assert len(engine.agents) == 5
    assert engine.agents[0].name == "Titan-AI"
