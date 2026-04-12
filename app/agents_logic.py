import random
import time
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime

STATE_FILE = "agent_state.json"

class TradingAgent(ABC):
    def __init__(self, name, strategy_name, initial_balance=1000):
        self.name = name
        self.strategy_name = strategy_name
        self.balance = initial_balance
        self.is_active = True
        self.trades = []
        self.creation_time = datetime.now()

    @abstractmethod
    def decide_trade(self, market_data):
        pass

    def execute_trade(self, success, amount, profit_loss):
        if not self.is_active:
            return

        self.balance += profit_loss
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "amount": amount,
            "profit_loss": profit_loss,
            "balance_after": self.balance,
            "success": success
        }
        self.trades.append(trade_record)

        # Keep trade history manageable
        if len(self.trades) > 20:
            self.trades.pop(0)

        if self.balance <= 0:
            self.balance = 0
            self.is_active = False

class TrendFollowerAgent(TradingAgent):
    def decide_trade(self, market_data):
        risk_amount = self.balance * random.uniform(0.1, 0.2)
        win_chance = 0.45
        success = random.random() < win_chance
        pl = risk_amount * (random.uniform(0.5, 1.5)) if success else -risk_amount
        return success, risk_amount, pl

class MeanReversionAgent(TradingAgent):
    def decide_trade(self, market_data):
        risk_amount = self.balance * random.uniform(0.02, 0.05)
        win_chance = 0.55
        success = random.random() < win_chance
        pl = risk_amount * (random.uniform(0.2, 0.6)) if success else -risk_amount
        return success, risk_amount, pl

class MLFilteredAgent(TradingAgent):
    def decide_trade(self, market_data):
        risk_amount = self.balance * random.uniform(0.05, 0.1)
        win_chance = 0.62
        success = random.random() < win_chance
        pl = risk_amount * (random.uniform(0.3, 0.8)) if success else -risk_amount
        return success, risk_amount, pl

class SimulationEngine:
    def __init__(self):
        self.agents = [
            TrendFollowerAgent("Alpha-Trend", "Trend Following"),
            MeanReversionAgent("Beta-Steady", "Mean Reversion"),
            MLFilteredAgent("Gamma-ML", "ML Optimized"),
            TrendFollowerAgent("Delta-High", "Aggressive Trend"),
            MLFilteredAgent("Epsilon-Bot", "ML Scalper")
        ]
        self.load_state()

    def step(self):
        for agent in self.agents:
            if agent.is_active:
                success, amount, pl = agent.decide_trade(None)
                agent.execute_trade(success, amount, pl)
        self.save_state()

    def save_state(self):
        state = []
        for a in self.agents:
            state.append({
                "name": a.name,
                "strategy": a.strategy_name,
                "balance": a.balance,
                "is_active": a.is_active,
                "trades": a.trades
            })
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                for i, s in enumerate(state):
                    if i < len(self.agents):
                        self.agents[i].balance = s['balance']
                        self.agents[i].is_active = s['is_active']
                        self.agents[i].trades = s['trades']
        except Exception:
            pass

    def get_status(self):
        return [
            {
                "name": a.name,
                "strategy": a.strategy_name,
                "balance": round(a.balance, 2),
                "is_active": a.is_active,
                "trade_count": len(a.trades),
                "last_trade": a.trades[-1] if a.trades else None
            }
            for a in self.agents
        ]
