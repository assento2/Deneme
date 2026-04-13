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
            print("Telegram NOT configured.")
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            async with httpx.AsyncClient() as client:
                await client.post(url, json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}, timeout=5)
        except Exception as e:
            print(f"Telegram error: {e}")

class Position:
    def __init__(self, symbol, side, entry_price, amount, tp_price, sl_price, timestamp, confidence):
        self.symbol = symbol
        self.side = side
        self.entry_price = float(entry_price)
        self.amount = float(amount)
        self.tp_price = float(tp_price)
        self.sl_price = float(sl_price)
        self.entry_time = timestamp
        self.confidence = confidence

class MLModel(ABC):
    @abstractmethod
    def predict(self, md): pass

class TrendModel(MLModel):
    def predict(self, md):
        closes = md["closes"]
        ema_short = calculate_ema(closes, 9)[-1]
        ema_long = calculate_ema(closes, 21)[-1]
        return 1 if ema_short > ema_long else -1

class VolatilityModel(MLModel):
    def predict(self, md):
        closes = md["closes"]
        rsi = calculate_rsi(closes)[-1]
        if rsi < 40: return 1
        if rsi > 60: return -1
        return 0

class SentimentConsensus:
    def __init__(self):
        self.models = [TrendModel(), VolatilityModel()]
    def get_confidence(self, md):
        votes = [m.predict(md) for m in self.models]
        score = 50
        if 1 in votes: score += 25 * votes.count(1)
        if -1 in votes: score -= 25 * votes.count(-1)
        return min(100, max(0, score))

class TradingAgent(ABC):
    def __init__(self, name, strategy_name, initial_balance=1000):
        self.name = name
        self.strategy_name = strategy_name
        self.symbol = "BTC_USDT"
        self.balance = float(initial_balance)
        self.leverage = 10
        self.is_active = True
        self.trades = []
        self.active_position = None
        self.ml_ensemble = SentimentConsensus()
        self.notifier = TelegramNotifier()

    @abstractmethod
    def decide_trade(self, md): pass

    async def open_position(self, symbol, side, price, amount, tp, sl, confidence):
        self.active_position = Position(symbol, side, price, amount, tp, sl, datetime.now().isoformat(), confidence)
        msg = (f"🚀 <b>{self.name} - İŞLEME GİRİLDİ</b>\n"
               f"━━━━━━━━━━━━━━━\n"
               f"<b>Parite:</b> {symbol}\n"
               f"<b>Yön:</b> {'🟢 LONG' if side == 'LONG' else '🔴 SHORT'}\n"
               f"<b>Giriş Fiyatı:</b> {price:.6f}\n"
               f"<b>Kaldıraç:</b> {self.leverage}x\n"
               f"<b>Güven Skoru:</b> %{confidence}\n"
               f"<b>Hedef (TP):</b> {tp:.6f}\n"
               f"<b>Zarar Durdur:</b> {sl:.6f}")
        await self.notifier.notify(msg)

    async def close_position(self, exit_price, timestamp, reason):
        if not self.active_position: return
        pos = self.active_position
        price_diff = float(exit_price) - pos.entry_price
        if pos.side == "SHORT": price_diff = -price_diff
        profit_pct = (price_diff / pos.entry_price) * 100
        profit_loss = (pos.amount * profit_pct / 100) * self.leverage
        self.balance += float(profit_loss)

        trade_record = {
            "timestamp": timestamp, "symbol": pos.symbol, "side": pos.side,
            "amount": round(float(pos.amount), 2), "entry_price": round(float(pos.entry_price), 6),
            "exit_price": round(float(exit_price), 6), "profit_loss": round(float(profit_loss), 2),
            "profit_pct": round(float(profit_pct * self.leverage), 2), "balance_after": round(float(self.balance), 2),
            "success": bool(profit_loss > 0), "reasoning": reason, "entry_time": pos.entry_time
        }
        self.trades.append(trade_record)

        emoji = "✅" if profit_loss > 0 else "❌"
        msg = (f"{emoji} <b>{self.name} - İŞLEM KAPATILDI</b>\n"
               f"━━━━━━━━━━━━━━━\n"
               f"<b>Parite:</b> {pos.symbol}\n"
               f"<b>Kâr/Zarar:</b> {profit_loss:+.2f}$ (%{profit_pct*self.leverage:+.2f})\n"
               f"<b>Giriş:</b> {pos.entry_price:.6f}\n"
               f"<b>Çıkış:</b> {exit_price:.6f}\n"
               f"<b>Neden:</b> {reason}\n"
               f"<b>Yeni Bakiye:</b> {self.balance:.2f}$")
        await self.notifier.notify(msg)

        self.active_position = None
        if len(self.trades) > 100: self.trades.pop(0)
        if self.balance <= 0:
            self.balance = 0.0
            self.is_active = False
            await self.notifier.notify(f"💀 <b>{self.name} TASFİYE OLDU (LIQUIDATED)</b>")

class IntelligenceAgent(TradingAgent):
    async def decide_trade(self, md):
        if self.active_position or not self.is_active: return None
        closes = md["closes"]
        current_price = float(closes[-1])
        confidence = int(self.ml_ensemble.get_confidence(md))
        if confidence < 70 and confidence > 30: return None
        atr = float(calculate_atr(md["highs"], md["lows"], closes)[-1])
        side = "LONG" if confidence >= 70 else "SHORT"
        tp_mult, sl_mult = 1.2, 1.0
        if side == "LONG":
            tp = current_price + (atr * tp_mult)
            sl = current_price - (atr * sl_mult)
        else:
            tp = current_price - (atr * tp_mult)
            sl = current_price + (atr * sl_mult)
        amount = self.balance * 0.2
        await self.open_position(self.symbol, side, current_price, amount, tp, sl, confidence)

class SimulationEngine:
    def __init__(self):
        self.agents = [
            IntelligenceAgent("Titan-AI", "Trend-Follower ML"),
            IntelligenceAgent("Oracle-Bot", "Sentiment Sniper"),
            IntelligenceAgent("Nexus-Alpha", "Mean-Reversion Neural"),
            IntelligenceAgent("Shadow-Tracer", "Volatility Master"),
            IntelligenceAgent("Aura-ML", "Adaptive Filter AI")
        ]
        self.load_state()

    async def step(self):
        symbols = await market_scanner()
        for i, agent in enumerate(self.agents):
            if not agent.is_active: continue
            target_symbol = symbols[i % len(symbols)]
            agent.symbol = target_symbol
            data = await fetch_mexc_kline(symbol=target_symbol, interval="5m", limit=100)
            if not data: continue
            md = {"closes": np.array([float(d['close']) for d in data]), "highs": np.array([float(d['high']) for d in data]), "lows": np.array([float(d['low']) for d in data])}
            if agent.active_position:
                pos = agent.active_position
                current_candle = data[-1]
                hit_tp = (pos.side == "LONG" and float(current_candle['high']) >= pos.tp_price) or (pos.side == "SHORT" and float(current_candle['low']) <= pos.tp_price)
                hit_sl = (pos.side == "LONG" and float(current_candle['low']) <= pos.sl_price) or (pos.side == "SHORT" and float(current_candle['high']) >= pos.sl_price)
                if hit_tp: await agent.close_position(pos.tp_price, datetime.now().isoformat(), "Take Profit")
                elif hit_sl: await agent.close_position(pos.sl_price, datetime.now().isoformat(), "Stop Loss")
            await agent.decide_trade(md)
        self.save_state()

    def save_state(self):
        def pos_to_dict(p):
            if not p: return None
            return {"symbol": p.symbol, "side": p.side, "entry_price": float(p.entry_price), "amount": float(p.amount), "tp_price": float(p.tp_price), "sl_price": float(p.sl_price), "entry_time": p.entry_time, "confidence": p.confidence}
        state = [{"name": a.name, "strategy": a.strategy_name, "balance": float(a.balance), "leverage": a.leverage, "is_active": bool(a.is_active), "trades": a.trades, "active_position": pos_to_dict(a.active_position)} for a in self.agents]
        with open(STATE_FILE, 'w') as f: json.dump(state, f)

    def load_state(self):
        if not os.path.exists(STATE_FILE): return
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                for i, s in enumerate(state):
                    if i < len(self.agents):
                        a = self.agents[i]
                        a.balance, a.is_active, a.trades = float(s['balance']), bool(s['is_active']), s.get('trades', [])
                        a.leverage = s.get('leverage', 10)
                        ap = s.get('active_position')
                        if ap: a.active_position = Position(ap['symbol'], ap['side'], ap['entry_price'], ap['amount'], ap['tp_price'], ap['sl_price'], ap['entry_time'], ap.get('confidence', 0))
        except Exception: pass

    def get_status(self):
        return [{
            "name": a.name, "strategy": a.strategy_name, "symbol": a.symbol, "balance": round(float(a.balance), 2), "leverage": a.leverage, "is_active": bool(a.is_active), "trade_count": len(a.trades), "last_trade": a.trades[-1] if a.trades else None,
            "active_position": {"side": a.active_position.side, "entry": round(float(a.active_position.entry_price), 6), "tp": round(float(a.active_position.tp_price), 6), "leverage": a.leverage} if a.active_position else None
        } for a in self.agents]
