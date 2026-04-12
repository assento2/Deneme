import random
import json
import os
import httpx
import asyncio
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from app.utils.mexc_api import fetch_mexc_kline
from app.utils.indicators import calculate_rsi, calculate_ema, calculate_bollinger_bands

STATE_FILE = "agent_state.json"

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

    async def notify(self, message):
        if not self.token or not self.chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            async with httpx.AsyncClient() as client:
                await client.post(url, json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except Exception as e:
            print(f"Telegram error: {e}")

class TradingAgent(ABC):
    def __init__(self, name, strategy_name, symbol="BTC_USDT", initial_balance=1000):
        self.name = name
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.balance = initial_balance
        self.is_active = True
        self.trades = []
        self.creation_time = datetime.now()
        self.performance_history = [] # To store win/loss for "online learning" adaptation

    @abstractmethod
    def decide_trade(self, closes, rsi, ema, bb):
        pass

    async def execute_trade(self, success, amount, profit_loss, notifier=None):
        if not self.is_active:
            return

        self.balance += profit_loss
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "amount": round(amount, 2),
            "profit_loss": round(profit_loss, 2),
            "balance_after": round(self.balance, 2),
            "success": success
        }
        self.trades.append(trade_record)
        self.performance_history.append(1 if success else 0)

        if len(self.trades) > 50:
            self.trades.pop(0)
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)

        if notifier:
            emoji = "✅" if success else "❌"
            pl_sign = "+" if profit_loss >= 0 else ""
            msg = (f"🤖 <b>Agent {self.name}</b>\n"
                   f"Strategy: {self.strategy_name}\n"
                   f"{emoji} Trade Amount: ${amount:.2f}\n"
                   f"💰 P/L: {pl_sign}{profit_loss:.2f}\n"
                   f"🏦 New Balance: ${self.balance:.2f}")
            await notifier.notify(msg)

        if self.balance <= 0:
            self.balance = 0
            self.is_active = False
            if notifier:
                await notifier.notify(f"💀 <b>Agent {self.name} has been DESTROYED!</b>")

class RSITrendAgent(TradingAgent):
    """Trend follower using RSI and EMA"""
    def decide_trade(self, closes, rsi, ema, bb):
        current_price = closes[-1]
        current_rsi = rsi[-1]
        current_ema = ema[-1]

        # Simple trend logic: RSI > 50 and price > EMA
        is_bullish = current_rsi > 50 and current_price > current_ema
        is_bearish = current_rsi < 50 and current_price < current_ema

        # Risk management: Adaptive based on win rate
        win_rate = sum(self.performance_history) / len(self.performance_history) if self.performance_history else 0.5
        risk_percent = 0.05 + (win_rate * 0.1) # Risk more if winning
        risk_amount = self.balance * risk_percent

        if is_bullish or is_bearish:
            # Simulation of market outcome based on technical alignment
            # In a real system, we'd wait for next candle. Here we simulate 'per step'
            success = random.random() < (0.55 if is_bullish or is_bearish else 0.4)
            pl = risk_amount * random.uniform(0.5, 1.2) if success else -risk_amount
            return success, risk_amount, pl
        return None

class BollingerReversionAgent(TradingAgent):
    """Mean reversion using Bollinger Bands"""
    def decide_trade(self, closes, rsi, ema, bb):
        upper, mid, lower = bb
        current_price = closes[-1]

        # Reversion logic: Price hits bands
        is_oversold = current_price < lower[-1]
        is_overbought = current_price > upper[-1]

        win_rate = sum(self.performance_history) / len(self.performance_history) if self.performance_history else 0.5
        risk_percent = 0.02 + (win_rate * 0.05)
        risk_amount = self.balance * risk_percent

        if is_oversold or is_overbought:
            success = random.random() < 0.58
            pl = risk_amount * random.uniform(0.3, 0.8) if success else -risk_amount
            return success, risk_amount, pl
        return None

class MLAdaptiveAgent(TradingAgent):
    """Advanced agent using a simple Reinforcement Learning (Perceptron) model that evolves"""
    def __init__(self, name, strategy_name, symbol="BTC_USDT", initial_balance=1000):
        super().__init__(name, strategy_name, symbol, initial_balance)
        # Weights for [RSI, Price/EMA, Volatility, Momentum]
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])
        self.learning_rate = 0.01

    def decide_trade(self, closes, rsi, ema, bb):
        upper, mid, lower = bb
        # Normalize features
        f1 = rsi[-1] / 100.0
        f2 = closes[-1] / ema[-1] if ema[-1] != 0 else 1.0
        f3 = (upper[-1] - lower[-1]) / mid[-1] if mid[-1] != 0 else 0.1
        f4 = closes[-1] / closes[-5] if len(closes) > 5 else 1.0

        features = np.array([f1, f2, f3, f4])
        prediction_score = np.dot(features, self.weights)

        # Adaptive risk
        win_rate = sum(self.performance_history) / len(self.performance_history) if self.performance_history else 0.5
        risk_amount = self.balance * (0.05 + (win_rate * 0.05))

        # Decision based on weighted activation
        if prediction_score > 0.6: # Bullish
            success = random.random() < (0.60 + (win_rate * 0.1))
            pl = risk_amount * random.uniform(0.5, 1.5) if success else -risk_amount
            self._update_weights(features, 1 if success else -1)
            return success, risk_amount, pl
        elif prediction_score < 0.4: # Bearish
            success = random.random() < (0.58 + (win_rate * 0.1))
            pl = risk_amount * random.uniform(0.5, 1.5) if success else -risk_amount
            self._update_weights(features, -1 if success else 1)
            return success, risk_amount, pl
        return None

    def _update_weights(self, features, direction):
        # Basic online learning update
        self.weights += self.learning_rate * direction * features
        # Keep weights normalized
        self.weights = self.weights / np.sum(np.abs(self.weights))

class SimulationEngine:
    def __init__(self):
        self.notifier = TelegramNotifier()
        self.agents = [
            RSITrendAgent("Alpha-Trend", "RSI Trend Follower", "BTC_USDT"),
            BollingerReversionAgent("Beta-Steady", "BB Mean Reversion", "ETH_USDT"),
            MLAdaptiveAgent("Gamma-ML", "ML Adaptive Optimized", "SOL_USDT"),
            RSITrendAgent("Delta-High", "Aggressive Trend", "BNB_USDT"),
            MLAdaptiveAgent("Epsilon-Bot", "ML Scalper High-Freq", "BTC_USDT")
        ]
        self.load_state()

    async def step(self):
        # In a real system, we'd fetch data for each agent's symbol
        # To optimize, we'll fetch BTC_USDT for now as a proxy or fetch all
        symbols = list(set(a.symbol for a in self.agents))
        market_data = {}

        for sym in symbols:
            data = await fetch_mexc_kline(symbol=sym, limit=100)
            if data:
                closes = np.array([float(d['close']) for d in data])
                market_data[sym] = {
                    "closes": closes,
                    "rsi": calculate_rsi(closes),
                    "ema": calculate_ema(closes),
                    "bb": calculate_bollinger_bands(closes)
                }

        for agent in self.agents:
            if agent.is_active and agent.symbol in market_data:
                md = market_data[agent.symbol]
                trade = agent.decide_trade(md["closes"], md["rsi"], md["ema"], md["bb"])
                if trade:
                    success, amount, pl = trade
                    await agent.execute_trade(success, amount, pl, self.notifier)

        self.save_state()

    def save_state(self):
        state = []
        for a in self.agents:
            state.append({
                "name": a.name,
                "strategy": a.strategy_name,
                "symbol": a.symbol,
                "balance": a.balance,
                "is_active": a.is_active,
                "trades": a.trades,
                "performance_history": a.performance_history
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
                        self.agents[i].trades = s.get('trades', [])
                        self.agents[i].performance_history = s.get('performance_history', [])
        except Exception:
            pass

    def get_status(self):
        return [
            {
                "name": a.name,
                "strategy": a.strategy_name,
                "symbol": a.symbol,
                "balance": round(a.balance, 2),
                "is_active": a.is_active,
                "trade_count": len(a.trades),
                "last_trade": a.trades[-1] if a.trades else None
            }
            for a in self.agents
        ]
