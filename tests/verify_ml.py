import pandas as pd
import numpy as np
from app.agents_logic import IntelligenceAgent
import asyncio

def generate_mock_data(n=300):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n))
    high = close + np.random.rand(n)
    low = close - np.random.rand(n)
    df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low
    })
    return df

def test_ml_agent():
    agent = IntelligenceAgent("Test-Agent")
    df = generate_mock_data(300)

    print("Testing initial prediction (should be learning mode)...")
    pred = agent.predict(df)
    print(f"Prediction: {pred}")

    print("\nTesting training...")
    agent.train(df)
    print(f"Is trained: {agent.is_trained}")

    print("\nTesting ML prediction...")
    pred_ml = agent.predict(df)
    print(f"Prediction: {pred_ml}")

    assert agent.is_trained == True
    assert "mode" in pred_ml
    assert pred_ml["mode"] == "ml"
    print("\nTest passed!")

if __name__ == "__main__":
    test_ml_agent()
