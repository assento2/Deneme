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
        self.side = side
        self.entry_price = entry_price
        self.leverage = leverage
        self.confidence = confidence
        self.entry_time = datetime.now().isoformat()
        self.fee_rate = 0.0006

        tp_dist = 0.015
        sl_dist = 0.005

        if side == "LONG":
            self.tp_price = entry_price * (1 + tp_dist)
            self.sl_price = entry_price * (1 - sl_dist)
        else:
            self.tp_price = entry_price * (1 - tp_dist)
            self.sl_price = entry_price * (1 + sl_dist)

    def calculate_pnl_pct(self, current_price: float) -> float:
        total_fees = self.fee_rate * 2 * self.leverage
        if self.side == "LONG":
            raw_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            raw_pnl = (self.entry_price - current_price) / self.entry_price
        return (raw_pnl * self.leverage) - total_fees

    def to_dict(self):
        return {
            "side": str(self.side),
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

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Stochastic RSI
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
        df_ind = self.calculate_indicators(df).tail(250)
        features = self._prepare_features(df_ind)
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
            return {"status": "Learning (RAM Eco)", "focus": "HEURISTIC", "importance": {}, "insight": "Scanning market structure..."}

        try:
            best_est = self.model.best_estimator
            importance_dict = {}
            # Robust feature name extraction
            dummy_data = pd.DataFrame({
                'close': [100.0]*50, 'high': [101.0]*50, 'low': [99.0]*50, 'vol': [1000.0]*50
            })
            feature_names = list(self._prepare_features(self.calculate_indicators(dummy_data)).columns)

            if hasattr(self.model.model.estimator, 'feature_importances_'):
                importances = self.model.model.estimator.feature_importances_
                importance_dict = {name: round(float(imp), 3) for name, imp in zip(feature_names, importances)}
            else:
                importance_dict = {name: 1.0/len(feature_names) for name in feature_names}

            top_feature = max(importance_dict, key=importance_dict.get)
            return {
                "status": f"Active ({best_est.upper()})",
                "focus": top_feature.upper(),
                "importance": importance_dict,
                "insight": f"Shared Brain is prioritizing {top_feature.upper()} for global inference."
            }
        except Exception as e:
            return {"status": "Active", "insight": f"Neural mapping online. (Error: {e})"}

    def predict(self, df: pd.DataFrame) -> Dict:
        if len(df) < 30: return {"side": "NONE", "confidence": 0}
        df_ind = self.calculate_indicators(df).tail(10)
        last_row = df_ind.iloc[-1:]

        if not self.is_trained:
            rsi = last_row['rsi'].values[0]
            side = "NONE"
            if rsi < 30: side = "LONG"
            elif rsi > 70: side = "SHORT"
            return {"side": side, "confidence": 55.0}

        X_test = self._prepare_features(last_row)
        X_test_clean = pd.DataFrame(X_test.values, columns=list(X_test.columns), dtype=np.float32)
        X_test_clean.columns = X_test_clean.columns.astype(object)

        try:
            prob = self.model.predict_proba(X_test_clean)[0][1]
            side = "NONE"
            confidence = 0
            if prob > 0.70: side = "LONG"; confidence = prob * 100
            elif prob < 0.30: side = "SHORT"; confidence = (1 - prob) * 100
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
        self.is_bottom_hunter = "Hunter-9" in name

    def tick(self, df: pd.DataFrame, shared_brain: IntelligenceAgent):
        if self.symbol == "WAITING...": return None
        if len(df) < 30: return None

        df_ind = shared_brain.calculate_indicators(df)
        last_row = df_ind.iloc[-1]
        prediction = shared_brain.predict(df)
        current_price = df.iloc[-1]['close']

        if self.active_position:
            should_exit = False
            exit_reason = ""
            exit_price = current_price
            pnl_high = self.active_position.calculate_pnl_pct(df.iloc[-1]['high'])
            pnl_low = self.active_position.calculate_pnl_pct(df.iloc[-1]['low'])

            if self.active_position.side == "LONG":
                if pnl_high >= 0.15: should_exit = True; exit_reason = "Take Profit"; exit_price = self.active_position.tp_price
                elif pnl_low <= -0.05: should_exit = True; exit_reason = "Stop Loss"; exit_price = self.active_position.sl_price
                elif prediction['side'] == "SHORT": should_exit = True; exit_reason = "Brain Reversal"
            else:
                if pnl_low >= 0.15: should_exit = True; exit_reason = "Take Profit"; exit_price = self.active_position.tp_price
                elif pnl_high <= -0.05: should_exit = True; exit_reason = "Stop Loss"; exit_price = self.active_position.sl_price
                elif prediction['side'] == "LONG": should_exit = True; exit_reason = "Brain Reversal"

            if should_exit:
                pnl_pct = float(self.active_position.calculate_pnl_pct(exit_price))
                net_profit = float(self.balance * pnl_pct)
                fees = float(self.balance * (self.active_position.fee_rate * 2 * self.leverage))
                self.balance += net_profit
                trade_record = {
                    "symbol": str(self.symbol), "side": str(self.active_position.side),
                    "entry_price": float(self.active_position.entry_price), "exit_price": float(exit_price),
                    "net_profit_loss": round(float(net_profit), 2), "profit_pct": round(float(pnl_pct * 100), 2),
                    "fees": round(float(fees), 2), "balance_after": round(float(self.balance), 2),
                    "reasoning": str(exit_reason), "success": bool(pnl_pct > 0), "timestamp": datetime.now().isoformat()
                }
                self.trades.append(trade_record)
                self.active_position = None
                self.symbol = "WAITING..."
                return {"type": "EXIT", "agent": self.name, **trade_record}
        else:
            can_enter = False
            conf = 0
            side = "NONE"

            # Bottom Hunter (Hunter-9) Logic: Extremes
            if self.is_bottom_hunter:
                # Needle in a haystack: RSI < 20 and StochRSI < 10
                if last_row['rsi'] < 20 and last_row['stoch_k'] < 10:
                    can_enter = True; side = "LONG"; conf = 85.0 # High conf for bottom catch
            else:
                if prediction['side'] != "NONE" and prediction['confidence'] >= 75:
                    can_enter = True; side = prediction['side']; conf = prediction['confidence']

            if can_enter:
                self.active_position = Position(side, float(current_price), self.leverage, float(conf))
                return {
                    "type": "ENTRY", "agent": str(self.name), "symbol": str(self.symbol),
                    "side": str(side), "price": float(current_price), "confidence": float(conf)
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

        # Dispatch Hunter-9 (Bottom Hunter) to extreme oversold projects
        hunter_9 = next((a for a in self.agents if a.name == "Aegis-Hunter-9"), None)
        if hunter_9 and not hunter_9.active_position:
            # Hunter-9 targets projects with RSI < 25 first
            bottom_opps = [o for o in opportunities if o.get('rsi', 100) < 25 and o['symbol'] not in active_symbols]
            if bottom_opps:
                hunter_9.symbol = bottom_opps[0]['symbol']
                idle_agents.remove(hunter_9)

        valid_opps = [o for o in opportunities if o['symbol'] not in active_symbols and o['confidence'] >= 75]
        valid_opps.sort(key=lambda x: x['confidence'], reverse=True)

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
        # Ensure insights values are also serializable
        if "importance" in brain_insights:
            brain_insights["importance"] = {k: float(v) for k, v in brain_insights["importance"].items()}

        return [{
            "name": str(a.name), "symbol": str(a.symbol), "balance": round(float(a.balance), 2),
            "trade_count": int(len(a.trades)), "leverage": int(a.leverage), "is_special": bool(a.is_bottom_hunter),
            "active_position": a.active_position.to_dict() if a.active_position else None,
            "ml_insights": brain_insights
        } for a in self.agents]
