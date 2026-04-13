import pandas as pd
import numpy as np
from flaml import AutoML
from typing import List, Dict, Optional
from datetime import datetime
import logging
import os
import json

logger = logging.getLogger(__name__)

class Position:
    def __init__(self, side: str, entry_price: float, leverage: int = 10, confidence: float = 0):
        self.side = side
        self.entry_price = entry_price
        self.leverage = leverage
        self.confidence = confidence
        self.entry_time = datetime.now().isoformat()
        self.fee_rate = 0.0006  # 0.06% Taker Fee

        # Calculate SL/TP prices for UI
        tp_dist = 0.015 # 1.5% move * 10x = 15% PnL
        sl_dist = 0.005 # 0.5% move * 10x = 5% PnL

        if side == "LONG":
            self.tp_price = entry_price * (1 + tp_dist)
            self.sl_price = entry_price * (1 - sl_dist)
        else:
            self.tp_price = entry_price * (1 - tp_dist)
            self.sl_price = entry_price * (1 + sl_dist)

    def calculate_pnl_pct(self, current_price: float) -> float:
        """Returns NET profit/loss percentage including fees."""
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
    """A self-developing ML agent using XGBoost."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoML()
        self.is_trained = False
        self.min_train_size = 200

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Basic Technical Indicators
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volatility
        df['volatility'] = df['close'].rolling(window=14).std() / df['close'].rolling(window=14).mean()

        # Momentum
        df['mom'] = df['close'].pct_change(periods=5)

        return df.dropna()

    def _prepare_features(self, df: pd.DataFrame):
        features = df[['rsi', 'volatility', 'mom']].copy()
        features['ema_diff'] = (df['ema_9'] - df['ema_21']) / df['ema_21']
        # Explicitly cast to float32 to avoid StringDtype issues in some pandas versions with FLAML
        return features.astype(np.float32)

    def train(self, df: pd.DataFrame):
        if len(df) < self.min_train_size: return
        df_ind = self.calculate_indicators(df)
        if len(df_ind) < 50: return
        features = self._prepare_features(df_ind)
        target = (df_ind['close'].shift(-5) > df_ind['close']).astype(int)
        X = features.iloc[:-5]
        y = target.iloc[:-5]
        try:
            settings = {
                "time_budget": 10,  # Increased for better accuracy
                "metric": 'accuracy',
                "task": 'classification',
                "log_file_name": "",
                "verbose": 0
            }
            self.model.fit(X_train=X, y_train=y, **settings)
            self.is_trained = True
        except Exception as e:
            logger.error(f"AutoML Training Error for {self.model_name}: {e}")

    def get_ml_insights(self) -> Dict:
        if not self.is_trained:
            return {
                "status": "Learning (AutoML Warmup)",
                "focus": "Heuristic Indicators",
                "importance": {"RSI": 0.4, "EMA": 0.4, "Volatility": 0.2},
                "insight": "AutoML is currently evaluating the best model architecture for this symbol."
            }

        try:
            best_est = self.model.best_estimator
            importance_dict = {}
            feature_names = ['rsi', 'volatility', 'mom', 'ema_diff']

            # Attempt to get feature importance from the best model
            if hasattr(self.model.model.estimator, 'feature_importances_'):
                importances = self.model.model.estimator.feature_importances_
                importance_dict = {name: round(float(imp), 3) for name, imp in zip(feature_names, importances)}
            else:
                importance_dict = {name: 0.25 for name in feature_names}

            top_feature = max(importance_dict, key=importance_dict.get)

            insights_map = {
                "rsi": "Strong focus on oversold/overbought cycles to predict reversals.",
                "volatility": "Prioritizing market stability and breakout volatility as key signal filters.",
                "mom": "Momentum tracking is currently the primary driver for directionality prediction.",
                "ema_diff": "Trend structural alignment (EMA crosses) is yielding the highest confidence."
            }

            return {
                "status": f"Operational (AutoML: {best_est.upper()})",
                "focus": top_feature.upper(),
                "importance": importance_dict,
                "insight": insights_map.get(top_feature, f"The optimized {best_est} model is detecting deep price correlations.")
            }
        except Exception as e:
            logger.error(f"Insights error: {e}")
            return {"status": "Error", "insight": "AutoML metrics temporarily unavailable."}

    def predict(self, df: pd.DataFrame) -> Dict:
        if len(df) < 30: return {"side": "NONE", "confidence": 0, "mode": "learning"}
        df_ind = self.calculate_indicators(df)
        if df_ind.empty: return {"side": "NONE", "confidence": 0, "mode": "learning"}
        last_row = df_ind.iloc[-1:]

        if not self.is_trained:
            rsi = last_row['rsi'].values[0]
            ema_diff = (last_row['ema_9'].values[0] - last_row['ema_21'].values[0])
            side = "NONE"
            if rsi < 35 and ema_diff > 0: side = "LONG"
            elif rsi > 65 and ema_diff < 0: side = "SHORT"
            return {"side": side, "confidence": 50.0, "mode": "learning"}

        X_test = self._prepare_features(last_row)
        prob = self.model.predict_proba(X_test)[0][1]
        side = "NONE"
        confidence = 0
        if prob > 0.65:
            side = "LONG"
            confidence = prob * 100
        elif prob < 0.35:
            side = "SHORT"
            confidence = (1 - prob) * 100
        return {"side": side, "confidence": round(float(confidence), 2), "mode": "ml"}

class TradingAgent:
    def __init__(self, name: str, symbol: str, balance: float = 1000.0, strategy: str = "ML Default"):
        self.name = name
        self.symbol = symbol
        self.balance = balance
        self.strategy_name = strategy
        self.intelligence = IntelligenceAgent(name)
        self.active_position: Optional[Position] = None
        self.trades = []
        self.leverage = 10

    def tick(self, df: pd.DataFrame):
        if len(df) < 2: return
        if not self.intelligence.is_trained and len(df) >= self.intelligence.min_train_size:
            self.intelligence.train(df)

        prediction = self.intelligence.predict(df)
        last_candle = df.iloc[-1]
        current_price = last_candle['close']

        if self.active_position:
            should_exit = False
            exit_reason = ""
            exit_price = current_price

            pnl_high = self.active_position.calculate_pnl_pct(last_candle['high'])
            pnl_low = self.active_position.calculate_pnl_pct(last_candle['low'])

            if self.active_position.side == "LONG":
                if pnl_high >= 0.15: # TP
                    should_exit = True; exit_reason = "Take Profit"; exit_price = self.active_position.tp_price
                elif pnl_low <= -0.05: # SL
                    should_exit = True; exit_reason = "Stop Loss"; exit_price = self.active_position.sl_price
                elif prediction['side'] == "SHORT":
                    should_exit = True; exit_reason = "Signal Reversal"
            else:
                if pnl_low >= 0.15: # TP for short (price down)
                    should_exit = True; exit_reason = "Take Profit"; exit_price = self.active_position.tp_price
                elif pnl_high <= -0.05: # SL for short (price up)
                    should_exit = True; exit_reason = "Stop Loss"; exit_price = self.active_position.sl_price
                elif prediction['side'] == "LONG":
                    should_exit = True; exit_reason = "Signal Reversal"

            if should_exit:
                pnl_pct = self.active_position.calculate_pnl_pct(exit_price)
                net_profit = self.balance * pnl_pct
                fees = self.balance * (self.active_position.fee_rate * 2 * self.leverage)

                self.balance += net_profit
                trade_record = {
                    "symbol": str(self.symbol),
                    "side": str(self.active_position.side),
                    "entry_price": round(float(self.active_position.entry_price), 6),
                    "exit_price": round(float(exit_price), 6),
                    "net_profit_loss": round(float(net_profit), 2),
                    "profit_pct": round(float(pnl_pct * 100), 2),
                    "fees": round(float(fees), 2),
                    "balance_after": round(float(self.balance), 2),
                    "reasoning": str(exit_reason),
                    "success": bool(pnl_pct > 0),
                    "timestamp": datetime.now().isoformat()
                }
                self.trades.append(trade_record)
                self.active_position = None
                return trade_record # Signal for notification
        else:
            if prediction['side'] != "NONE" and prediction['confidence'] >= 65:
                self.active_position = Position(prediction['side'], current_price, self.leverage, prediction['confidence'])
                return {
                    "type": "ENTRY",
                    "agent": self.name,
                    "symbol": self.symbol,
                    "side": prediction['side'],
                    "price": current_price,
                    "confidence": prediction['confidence']
                }
        return None

class SimulationEngine:
    STATE_FILE = "agent_state.json"

    def __init__(self):
        self.agents = [
            TradingAgent("Titan-AI", "BTC_USDT", 1000.0, "Neural Momentum"),
            TradingAgent("Oracle-Bot", "ETH_USDT", 1000.0, "XGB-Trend"),
            TradingAgent("Nexus Alpha", "SOL_USDT", 1000.0, "Ensemble Scalp"),
            TradingAgent("Cyber-Whale", "BNB_USDT", 1000.0, "Deep Liquidity"),
            TradingAgent("Aegis-Trader", "XRP_USDT", 1000.0, "Risk-Adjusted ML")
        ]
        self.load_state()

    def save_state(self):
        state = []
        for a in self.agents:
            state.append({
                "name": a.name,
                "balance": a.balance,
                "trades": a.trades,
                "active_position": a.active_position.to_dict() if a.active_position else None
            })
        with open(self.STATE_FILE, "w") as f:
            json.dump(state, f)

    def load_state(self):
        if not os.path.exists(self.STATE_FILE): return
        try:
            with open(self.STATE_FILE, "r") as f:
                state = json.load(f)
            for s in state:
                agent = next((a for a in self.agents if a.name == s["name"]), None)
                if agent:
                    agent.balance = s["balance"]
                    agent.trades = s["trades"]
                    if s["active_position"]:
                        pos = s["active_position"]
                        agent.active_position = Position(pos["side"], pos["entry"], pos["leverage"], pos["confidence"])
                        agent.active_position.entry_time = pos["entry_time"]
        except Exception as e:
            logger.error(f"Load state error: {e}")

    async def step(self):
        from app.utils.mexc_api import fetch_mexc_kline, market_scanner

        # Periodic Market Realignment: Ensure agents are on the most relevant pairs
        try:
            top_pairs = await market_scanner()
            # If an agent is not in a position, consider re-assigning it to a top trending pair
            for i, agent in enumerate(self.agents):
                if not agent.active_position and i < len(top_pairs):
                    # Rotate pairs if not currently in trade to catch new trends
                    if agent.symbol not in top_pairs[:10]:
                        new_symbol = top_pairs[i % 20]
                        if new_symbol != agent.symbol:
                            logger.info(f"Re-aligning {agent.name} from {agent.symbol} to {new_symbol}")
                            agent.symbol = new_symbol
                            agent.intelligence.is_trained = False # Re-train for new pair
        except Exception as e:
            logger.error(f"Market realignment error: {e}")

        notifications = []
        for agent in self.agents:
            try:
                data = await fetch_mexc_kline(agent.symbol, interval="5m", limit=500)
                if data:
                    df = pd.DataFrame(data)
                    result = agent.tick(df)
                    if result:
                        if "type" in result: # Entry
                            notifications.append(result)
                        else: # Exit
                            notifications.append({"type": "EXIT", "agent": agent.name, **result})
            except Exception as e:
                logger.error(f"Error in {agent.name}: {e}")

        if notifications:
            self.save_state()
        return notifications

    def get_status(self):
        return [
            {
                "name": str(a.name),
                "symbol": str(a.symbol),
                "strategy": str(a.strategy_name),
                "balance": round(float(a.balance), 2),
                "trade_count": int(len(a.trades)),
                "leverage": int(a.leverage),
                "active_position": a.active_position.to_dict() if a.active_position else None,
                "last_trade": a.trades[-1] if a.trades else None,
                "ml_insights": a.intelligence.get_ml_insights()
            }
            for a in self.agents
        ]
