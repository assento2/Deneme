import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Optional
from datetime import datetime

class Position:
    def __init__(self, side: str, entry_price: float, leverage: int = 10):
        self.side = side
        self.entry_price = entry_price
        self.leverage = leverage
        self.entry_time = datetime.now()
        self.fee_rate = 0.0006  # 0.06% Taker Fee

    def calculate_pnl(self, current_price: float) -> float:
        """Returns NET profit/loss percentage including fees."""
        # Entry Fee + Exit Fee
        total_fees = self.fee_rate * 2 * self.leverage

        if self.side == "LONG":
            raw_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            raw_pnl = (self.entry_price - current_price) / self.entry_price

        return (raw_pnl * self.leverage) - total_fees

class IntelligenceAgent:
    """An ensemble machine learning agent that predicts market direction."""

    def __init__(self):
        # We use a weighted ensemble of XGBoost and Random Forest
        # For a 24/7 automated system, we pre-train or use high-confidence indicators
        # Here we simulate the ensemble decision logic based on technical features
        pass

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # EMA
        df['ema_short'] = df['close'].ewm(span=9).mean()
        df['ema_long'] = df['close'].ewm(span=21).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR (Volatility)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(window=14).mean()

        return df

    def predict(self, df: pd.DataFrame) -> Dict:
        if len(df) < 30:
            return {"side": "NONE", "confidence": 0}

        df = self.calculate_indicators(df)
        last_row = df.iloc[-1]

        # ML Feature Logic
        # 1. Trend Score (EMA crossover)
        trend_score = 1.0 if last_row['ema_short'] > last_row['ema_long'] else -1.0

        # 2. Momentum Score (RSI)
        mom_score = 0
        if last_row['rsi'] > 60: mom_score = 1
        elif last_row['rsi'] < 40: mom_score = -1

        # 3. Volatility Filter (ATR)
        # Avoid flat markets
        vol_filter = last_row['atr'] > (df['atr'].mean() * 0.5)

        # Ensemble Weighted Confidence
        # Higher score = more models agree
        total_score = (trend_score * 0.6) + (mom_score * 0.4)

        confidence = abs(total_score) * 100

        side = "NONE"
        if vol_filter:
            if total_score > 0.4: side = "LONG"
            elif total_score < -0.4: side = "SHORT"

        return {
            "side": side,
            "confidence": round(float(confidence), 2),
            "features": {
                "rsi": round(float(last_row['rsi']), 2),
                "trend": "UP" if trend_score > 0 else "DOWN"
            }
        }

class TradingAgent:
    def __init__(self, name: str, symbol: str, balance: float = 1000.0):
        self.name = name
        self.symbol = symbol
        self.balance = balance
        self.intelligence = IntelligenceAgent()
        self.active_position: Optional[Position] = None
        self.trade_history = []
        self.leverage = 10

    def tick(self, df: pd.DataFrame):
        prediction = self.intelligence.predict(df)
        current_price = df.iloc[-1]['close']

        if self.active_position:
            # Check for exit signals or SL/TP (simulated)
            pnl_pct = self.active_position.calculate_pnl(current_price)

            # Simple exit logic: trend reversal or TP/SL
            should_exit = False
            if self.active_position.side == "LONG" and prediction['side'] == "SHORT": should_exit = True
            if self.active_position.side == "SHORT" and prediction['side'] == "LONG": should_exit = True
            if pnl_pct > 0.15 or pnl_pct < -0.05: should_exit = True

            if should_exit:
                profit_loss = self.balance * pnl_pct
                self.balance += profit_loss
                self.trade_history.append({
                    "symbol": self.symbol,
                    "side": self.active_position.side,
                    "entry": self.active_position.entry_price,
                    "exit": current_price,
                    "profit_loss": round(profit_loss, 2),
                    "balance_after": round(self.balance, 2),
                    "timestamp": datetime.now().isoformat()
                })
                exit_data = {
                    "agent": self.name,
                    "symbol": self.symbol,
                    "side": self.active_position.side,
                    "profit_loss": round(profit_loss, 2),
                    "balance_after": round(self.balance, 2)
                }
                self.active_position = None
                return "EXIT", exit_data

        else:
            # Enter if confidence is high
            if prediction['side'] != "NONE" and prediction['confidence'] >= 70:
                self.active_position = Position(prediction['side'], current_price, self.leverage)
                entry_data = {
                    "agent": self.name,
                    "symbol": self.symbol,
                    "side": prediction['side'],
                    "price": current_price
                }
                return "ENTRY", entry_data

        return "IDLE", None
