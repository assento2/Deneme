import pandas as pd
import numpy as np
from app.agents_logic import TradingAgent, Position
from datetime import datetime

def test_trade_accuracy():
    agent = TradingAgent("Test-Agent", "BTC_USDT", 1000.0)
    # Mock a position
    agent.active_position = Position("LONG", 100.0, 10) # Entry at 100

    # Case 1: Price goes up but high doesn't hit TP, low doesn't hit SL
    # Indicators need at least 30 rows for prediction to not return NONE immediately
    df_base = pd.DataFrame([{
        'close': 100.0, 'high': 100.1, 'low': 99.9
    }] * 35)

    agent.tick(df_base)
    assert agent.active_position is not None, "Should not exit if SL/TP not hit"

    # Case 2: High hits TP (15% net gain)
    df_tp = df_base.copy()
    df_tp.iloc[-1] = {'close': 101.0, 'high': 102.0, 'low': 100.0}

    agent.tick(df_tp)
    assert agent.active_position is None, "Should exit when high hits TP"
    assert agent.trades[-1]["reason"] == "Take Profit"
    print(f"TP Exit Price: {agent.trades[-1]['exit']}")

    print("Test passed!")

if __name__ == "__main__":
    test_trade_accuracy()
