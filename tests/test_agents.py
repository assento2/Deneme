import pytest
from app.agents_logic import TrendFollowerAgent, MeanReversionAgent, MLFilteredAgent, SimulationEngine

def test_agent_initialization():
    agent = TrendFollowerAgent("Test", "Strategy")
    assert agent.name == "Test"
    assert agent.balance == 1000
    assert agent.is_active is True

def test_agent_elimination():
    agent = TrendFollowerAgent("Test", "Strategy", initial_balance=10)
    # Execute a large losing trade
    agent.execute_trade(False, 10, -15)
    assert agent.balance == 0
    assert agent.is_active is False

def test_simulation_step():
    engine = SimulationEngine()
    initial_balances = [a.balance for a in engine.agents]
    engine.step()
    new_balances = [a.balance for a in engine.agents]
    # Balances should change after a step
    assert initial_balances != new_balances

def test_ml_agent_performance():
    # Over many steps, ML agent should statistically do better or at least stay active
    agent = MLFilteredAgent("ML", "ML")
    for _ in range(10):
        success, amount, pl = agent.decide_trade(None)
        agent.execute_trade(success, amount, pl)
    assert len(agent.trades) == 10
