import pytest
from app.agents_logic import RSITrendAgent, BollingerReversionAgent, MLAdaptiveAgent, SimulationEngine

def test_agent_initialization():
    agent = RSITrendAgent("Test-Agent", "Test-Strategy", "BTC_USDT", 1000)
    assert agent.name == "Test-Agent"
    assert agent.balance == 1000
    assert agent.is_active is True

@pytest.mark.asyncio
async def test_agent_destruction():
    agent = RSITrendAgent("Test-Agent", "Test-Strategy", "BTC_USDT", 10)
    await agent.execute_trade(False, 10, -10)
    assert agent.balance == 0
    assert agent.is_active is False

@pytest.mark.asyncio
async def test_engine_load():
    engine = SimulationEngine()
    assert len(engine.agents) == 5
    assert engine.agents[0].name == "Alpha-Trend"
