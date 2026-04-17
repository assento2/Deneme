import pandas as pd
import numpy as np
from flaml import AutoML
from typing import List, Dict, Optional
from datetime import datetime
import logging
import os
import json
import warnings
import asyncio
import gc

logger = logging.getLogger(__name__)

class Position:
    def __init__(self, side: str, entry_price: float, leverage: int = 10, confidence: float = 0):
        self.side = "LONG" # Enforce LONG only for bottom hunting
        self.entry_price = entry_price
        self.leverage = leverage
        self.confidence = confidence
        self.entry_time = datetime.now().isoformat()
        self.fee_rate = 0.0006

        # Optimized targets for wick reversals
        tp_dist = 0.025 # 2.5% move (25% pnl at 10x)
        sl_dist = 0.012 # 1.2% move (12% pnl at 10x) - Added breathing room

        self.tp_price = entry_price * (1 + tp_dist)
        self.sl_price = entry_price * (1 - sl_dist)

    def calculate_pnl_pct(self, current_price: float) -> float:
        total_fees = self.fee_rate * 2 * self.leverage
        raw_pnl = (current_price - self.entry_price) / self.entry_price
        return (raw_pnl * self.leverage) - total_fees

    def to_dict(self):
        return {
            "side": "LONG",
            "entry": float(self.entry_price),
            "tp": round(float(self.tp_price), 6),
            "sl": round(float(self.sl_price), 6),
            "leverage": int(self.leverage),
            "confidence": round(float(self.confidence), 1),
            "entry_time": str(self.entry_time)
        }

class IntelligenceAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoML()
        self.is_trained = False
        self.min_train_size = 150

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        low_rsi = df['rsi'].rolling(window=14).min()
        high_rsi = df['rsi'].rolling(window=14).max()
        df['stoch_rsi'] = (df['rsi'] - low_rsi) / (high_rsi - low_rsi)
        df['stoch_k'] = df['stoch_rsi'].rolling(window=3).mean() * 100
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        df['volatility'] = df['close'].rolling(window=14).std() / df['close'].rolling(window=14).mean()
        df['mom'] = df['close'].pct_change(periods=5)

        for i in [1, 3]:
            df[f'rsi_lag_{i}'] = df['rsi'].shift(i)

        return df.bfill().dropna()

    def _prepare_features(self, df: pd.DataFrame):
        base_cols = ['rsi', 'volatility', 'mom', 'rsi_lag_1', 'rsi_lag_3', 'stoch_k', 'stoch_d']
        features = df[base_cols].copy()
        features['ema_diff'] = (df['ema_9'] - df['ema_21']) / df['ema_21']
        return features.astype(np.float32)

    def train(self, df: pd.DataFrame):
        if len(df) < self.min_train_size: return
        df_ind = self.calculate_indicators(df).tail(300)
        features = self._prepare_features(df_ind)
        # Focus target only on LONG potential
        target = (df_ind['close'].shift(-5) > df_ind['close']).astype(int)
        X = features.iloc[:-5]
        y = target.iloc[:-5]

        try:
            settings = {"time_budget": 5, "metric": 'accuracy', "task": 'classification',
                        "estimator_list": ['lgbm'], "log_file_name": "", "verbose": 0, "n_jobs": 1}
            X_clean = pd.DataFrame(X.values, columns=list(X.columns), dtype=np.float32)
            X_clean.columns = X_clean.columns.astype(object)
            self.model.fit(X_train=X_clean, y_train=y.values, **settings)
            self.is_trained = True
            gc.collect()
        except: pass

    def get_ml_insights(self) -> Dict:
        if not self.is_trained:
            return {"status": "Wick-Hunting Mode", "focus": "BOTTOMS", "importance": {}, "insight": "Scanning for extreme oversold signatures..."}
        try:
            best_est = self.model.best_estimator
            dummy_data = pd.DataFrame({'close': [100.0]*50, 'high': [101.0]*50, 'low': [99.0]*50, 'vol': [1000.0]*50})
            feature_names = list(self._prepare_features(self.calculate_indicators(dummy_data)).columns)
            if hasattr(self.model.model.estimator, 'feature_importances_'):
                importances = self.model.model.estimator.feature_importances_
                importance_dict = {name: round(float(imp), 3) for name, imp in zip(feature_names, importances)}
            else:
                importance_dict = {name: 1.0/len(feature_names) for name in feature_names}
            return {
                "status": f"Active ({best_est.upper()})", "focus": max(importance_dict, key=importance_dict.get).upper(),
                "importance": importance_dict, "insight": "Shared Brain optimized for high-precision LONG reversals."
            }
        except Exception as e: return {"status": "Active", "insight": f"Neural mapping online."}

    def predict(self, df: pd.DataFrame) -> Dict:
        if len(df) < 30: return {"side": "NONE", "confidence": 0}
        df_ind = self.calculate_indicators(df).tail(10)
        last_row = df_ind.iloc[-1:]

        if not self.is_trained:
            rsi = last_row['rsi'].values[0]
            if rsi < 15: return {"side": "LONG", "confidence": 80.0}
            return {"side": "NONE", "confidence": 0}

        X_test_clean = pd.DataFrame(self._prepare_features(last_row).values, columns=list(self._prepare_features(last_row).columns), dtype=np.float32)
        X_test_clean.columns = X_test_clean.columns.astype(object)

        try:
            prob = self.model.predict_proba(X_test_clean)[0][1]
            side = "NONE"; confidence = 0
            if prob > 0.70: side = "LONG"; confidence = prob * 100
            # Removed SHORT predictions to ensure LONG-ONLY
            return {"side": side, "confidence": round(float(confidence), 2)}
        except: return {"side": "NONE", "confidence": 0}

class TradingAgent:
    def __init__(self, name: str, balance: float = 1000.0):
        self.name = name
        self.symbol = "WAITING..."
        self.balance = balance
        self.active_position: Optional[Position] = None
        self.trades = []
        self.leverage = 10
        self.is_special = True

    def tick(self, df: pd.DataFrame, shared_brain: IntelligenceAgent):
        if self.symbol == "WAITING..." or len(df) < 30: return None

        df_ind = shared_brain.calculate_indicators(df)
        last_row = df_ind.iloc[-1]
        prediction = shared_brain.predict(df)
        current_price = df.iloc[-1]['close']

        if self.active_position:
            should_exit = False; exit_reason = ""; exit_price = current_price
            pnl_high = self.active_position.calculate_pnl_pct(df.iloc[-1]['high'])
            pnl_low = self.active_position.calculate_pnl_pct(df.iloc[-1]['low'])

            if pnl_high >= 0.25: should_exit = True; exit_reason = "Take Profit"; exit_price = self.active_position.tp_price
            elif pnl_low <= -0.12: should_exit = True; exit_reason = "Stop Loss"; exit_price = self.active_position.sl_price
            elif prediction['side'] == "NONE" and last_row['rsi'] > 60: should_exit = True; exit_reason = "Overbought Reversal"

            if should_exit:
                pnl_pct = float(self.active_position.calculate_pnl_pct(exit_price))
                net_profit = float(self.balance * pnl_pct)
                fees = float(self.balance * (self.active_position.fee_rate * 2 * self.leverage))
                self.balance += net_profit
                trade_record = {
                    "symbol": str(self.symbol), "side": "LONG",
                    "entry_price": float(self.active_position.entry_price), "exit_price": float(exit_price),
                    "net_profit_loss": round(float(net_profit), 2), "profit_pct": round(float(pnl_pct * 100), 2),
                    "fees": round(float(fees), 2), "balance_after": round(float(self.balance), 2),
                    "size": round(float(self.balance - net_profit), 2),
                    "leveraged_size": round(float((self.balance - net_profit) * self.leverage), 2),
                    "leverage": int(self.leverage),
                    "reasoning": str(exit_reason), "success": bool(pnl_pct > 0), "timestamp": datetime.now().isoformat()
                }
                self.trades.append(trade_record)
                self.active_position = None; self.symbol = "WAITING..."
                return {"type": "EXIT", "agent": self.name, **trade_record}
        else:
            can_enter = False; conf = 0; entry_reason = ""
            rsi_val = round(float(last_row['rsi']), 2)
            stoch_val = round(float(last_row['stoch_k']), 2)

            # Deep Bottom Trigger: RSI < 20 and StochK < 10
            if rsi_val < 20 and stoch_val < 10:
                can_enter = True; conf = 90.0
                entry_reason = f"Aşırı Satım (Wick-Hunt) | RSI: {rsi_val}, StochRSI: {stoch_val}"
            elif prediction['side'] == "LONG" and prediction['confidence'] >= 75:
                can_enter = True; conf = prediction['confidence']
                entry_reason = f"ML Güçlü Alım Sinyali | RSI: {rsi_val}, Güven: %{conf}"

            if can_enter:
                self.active_position = Position("LONG", float(current_price), self.leverage, float(conf))
                return {
                    "type": "ENTRY", "agent": str(self.name), "symbol": str(self.symbol),
                    "side": "LONG", "price": float(current_price), "confidence": float(conf),
                    "tp": float(self.active_position.tp_price),
                    "sl": float(self.active_position.sl_price),
                    "size": round(float(self.balance), 2),
                    "leveraged_size": round(float(self.balance * self.leverage), 2),
                    "leverage": int(self.leverage),
                    "reasoning": entry_reason,
                    "rsi": rsi_val,
                    "stoch": stoch_val
                }
        return None

class SimulationEngine:
    STATE_FILE = "agent_state.json"

    def __init__(self):
        self.agents = [TradingAgent(f"Aegis-Hunter-{i+1}", 1000.0) for i in range(9)]
        self.brain = IntelligenceAgent("GlobalBrain")
        self.load_state()
        self.global_opportunities = []

    def save_state(self):
        state = [{"name": a.name, "balance": a.balance, "trades": a.trades, "symbol": a.symbol,
                  "active_position": a.active_position.to_dict() if a.active_position else None} for a in self.agents]
        with open(self.STATE_FILE, "w") as f: json.dump(state, f)

    def load_state(self):
        if not os.path.exists(self.STATE_FILE): return
        try:
            with open(self.STATE_FILE, "r") as f: state = json.load(f)
            for s in state[:9]:
                agent = next((a for a in self.agents if a.name == s["name"]), None)
                if agent:
                    agent.balance, agent.trades, agent.symbol = s["balance"], s["trades"], s["symbol"]
                    if s["active_position"]:
                        p = s["active_position"]
                        agent.active_position = Position(p["side"], p["entry"], p["leverage"], p["confidence"])
                        agent.active_position.entry_time = p["entry_time"]
        except: pass

    async def step(self, opportunities: List[Dict]):
        self.global_opportunities = opportunities
        notifications = []
        active_symbols = [a.symbol for a in self.agents if a.active_position]
        idle_agents = [a for a in self.agents if not a.active_position]

        valid_opps = sorted(opportunities, key=lambda x: (x.get('rsi', 100), -x['confidence']))
        valid_opps = [o for o in valid_opps if o['symbol'] not in active_symbols]

        for agent in idle_agents:
            if valid_opps:
                opp = valid_opps.pop(0)
                agent.symbol = opp['symbol']

        from app.utils.mexc_api import fetch_mexc_kline
        for agent in self.agents:
            if agent.symbol == "WAITING...": continue
            try:
                data = await fetch_mexc_kline(agent.symbol, interval="5m", limit=300)
                if data:
                    df = pd.DataFrame(data)
                    if not self.brain.is_trained: self.brain.train(df)
                    res = agent.tick(df, self.brain)
                    if res: notifications.append(res)
            except: pass

        if notifications: self.save_state()
        gc.collect()
        return notifications

    def get_status(self):
        brain_insights = self.brain.get_ml_insights()
        if "importance" in brain_insights:
            brain_insights["importance"] = {k: float(v) for k, v in brain_insights["importance"].items()}
        return [{
            "name": str(a.name), "symbol": str(a.symbol), "balance": round(float(a.balance), 2),
            "trade_count": int(len(a.trades)), "leverage": int(a.leverage), "is_special": bool(a.is_special),
            "active_position": a.active_position.to_dict() if a.active_position else None,
            "ml_insights": brain_insights
        } for a in self.agents]
