import random
import json
import os
import httpx
import asyncio
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from app.utils.mexc_api import fetch_mexc_kline, market_scanner
from app.utils.indicators import *

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
    def __init__(self, name, strategy_name, initial_balance=1000):
        self.name = name
        self.strategy_name = strategy_name
        self.symbol = "BTC_USDT" # Will be dynamically updated by scanner
        self.balance = initial_balance
        self.is_active = True
        self.trades = []
        self.performance_history = []
        self.weights = np.ones(8) / 8 # Weights for different indicators

    @abstractmethod
    def decide_trade(self, md):
        pass

    async def execute_trade(self, success, amount, profit_loss, entry_price, reasoning, indicators, notifier=None):
        if not self.is_active: return

        self.balance += profit_loss
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "amount": round(amount, 2),
            "entry_price": entry_price,
            "profit_loss": round(profit_loss, 2),
            "balance_after": round(self.balance, 2),
            "success": success,
            "reasoning": reasoning,
            "indicators": indicators
        }
        self.trades.append(trade_record)
        self.performance_history.append(1 if success else 0)

        if len(self.trades) > 100: self.trades.pop(0)

        if notifier:
            emoji = "🚀" if success else "📉"
            pl_sign = "+" if profit_loss >= 0 else ""
            msg = (f"{emoji} <b>{self.name} - Trade Execution</b>\n"
                   f"Symbol: {self.symbol} @ {entry_price}\n"
                   f"Reason: <i>{reasoning}</i>\n"
                   f"P/L: {pl_sign}{profit_loss:.2f}\n"
                   f"Balance: ${self.balance:.2f}")
            await notifier.notify(msg)

        if self.balance <= 0:
            self.balance = 0
            self.is_active = False
            if notifier: await notifier.notify(f"💀 <b>Agent {self.name} has been LIQUIDATED!</b>")

class IntelligenceAgent(TradingAgent):
    """Advanced AI Agent using Multi-Indicator Weighted Logic & Online Learning"""
    def decide_trade(self, md):
        closes, highs, lows = md["closes"], md["highs"], md["lows"]
        current_price = closes[-1]

        # Calculate Indicators
        rsi = calculate_rsi(closes)[-1]
        macd, signal = calculate_macd(closes)
        macd_val, signal_val = macd[-1], signal[-1]
        upper, mid, lower = calculate_bollinger_bands(closes)
        k, d = calculate_stochastic(highs, lows, closes)
        adx = calculate_adx(highs, lows, closes)[-1]
        atr = calculate_atr(highs, lows, closes)[-1]

        # Signals (-1 to 1)
        s_rsi = 1 if rsi < 30 else -1 if rsi > 70 else 0
        s_macd = 1 if macd_val > signal_val else -1
        s_bb = 1 if current_price < lower[-1] else -1 if current_price > upper[-1] else 0
        s_stoch = 1 if k[-1] < 20 else -1 if k[-1] > 80 else 0
        s_trend = 1 if current_price > mid[-1] else -1
        s_adx = 1 if adx > 25 else 0 # Strength filter

        # Weighted Decision
        signals = np.array([s_rsi, s_macd, s_bb, s_stoch, s_trend, s_adx, 0, 0])[:len(self.weights)]
        score = np.dot(signals, self.weights)

        indicator_snapshot = {
            "rsi": round(rsi, 2), "macd": round(macd_val, 2),
            "bb_lower": round(lower[-1], 2), "stoch_k": round(k[-1], 2),
            "adx": round(adx, 2)
        }

        risk_amount = self.balance * 0.1

        if score > 0.1: # Bullish signal
            reason = f"Combined Bullish Score ({score:.2f}) | RSI: {rsi:.1f}, MACD: {macd_val:.1f} crossover"
            success = random.random() < (0.55 + (0.1 * s_adx)) # Better odds in strong trends
            pl = risk_amount * random.uniform(0.5, 2.0) if success else -risk_amount
            return success, risk_amount, pl, current_price, reason, indicator_snapshot

        elif score < -0.1: # Bearish signal
            reason = f"Combined Bearish Score ({score:.2f}) | BB Overbought, Stoch D: {d[-1]:.1f}"
            success = random.random() < (0.53 + (0.1 * s_adx))
            pl = risk_amount * random.uniform(0.5, 2.0) if success else -risk_amount
            return success, risk_amount, pl, current_price, reason, indicator_snapshot

        return None

class SimulationEngine:
    def __init__(self):
        self.notifier = TelegramNotifier()
        self.agents = [
            IntelligenceAgent("Titan-AI", "Multi-Factor Aggressive"),
            IntelligenceAgent("Oracle-Bot", "Trend Strength Specialist"),
            IntelligenceAgent("Nexus-Alpha", "Mean Reversion Expert"),
            IntelligenceAgent("Shadow-Tracer", "Volatility Scalper"),
            IntelligenceAgent("Aura-ML", "Adaptive Neural Filter")
        ]
        self.load_state()

    async def step(self):
        symbols = await market_scanner()

        for i, agent in enumerate(self.agents):
            if not agent.is_active: continue

            # Rotate symbols among agents for maximum coverage
            target_symbol = symbols[i % len(symbols)]
            agent.symbol = target_symbol

            data = await fetch_mexc_kline(symbol=target_symbol, limit=100)
            if data:
                md = {
                    "closes": np.array([d['close'] for d in data]),
                    "highs": np.array([d['high'] for d in data]),
                    "lows": np.array([d['low'] for d in data])
                }
                decision = agent.decide_trade(md)
                if decision:
                    success, amount, pl, price, reason, indicators = decision
                    await agent.execute_trade(success, amount, pl, price, reason, indicators, self.notifier)
                    # Online learning: update weights
                    self._update_agent_weights(agent, success, indicators)

        self.save_state()

    def _update_agent_weights(self, agent, success, indicators):
        # Extremely simplified RL: boost weights of indicators that were extreme during success
        lr = 0.05
        factor = 1 if success else -0.5
        if indicators["rsi"] < 30 or indicators["rsi"] > 70: agent.weights[0] += lr * factor
        if abs(indicators["macd"]) > 10: agent.weights[1] += lr * factor
        # Re-normalize
        agent.weights = np.clip(agent.weights, 0.01, 0.5)
        agent.weights /= np.sum(agent.weights)

    def save_state(self):
        state = [{"name": a.name, "strategy": a.strategy_name, "balance": a.balance,
                  "is_active": a.is_active, "trades": a.trades, "weights": a.weights.tolist()} for a in self.agents]
        with open(STATE_FILE, 'w') as f: json.dump(state, f)

    def load_state(self):
        if not os.path.exists(STATE_FILE): return
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                for i, s in enumerate(state):
                    if i < len(self.agents):
                        self.agents[i].balance, self.agents[i].is_active = s['balance'], s['is_active']
                        self.agents[i].trades = s.get('trades', [])
                        if 'weights' in s: self.agents[i].weights = np.array(s['weights'])
        except Exception: pass

    def get_status(self):
        return [{"name": a.name, "strategy": a.strategy_name, "symbol": a.symbol,
                 "balance": round(a.balance, 2), "is_active": a.is_active,
                 "trade_count": len(a.trades), "last_trade": a.trades[-1] if a.trades else None,
                 "weights": a.weights.tolist()} for a in self.agents]
