import asyncio
from app.utils.mexc_api import fetch_mexc_kline

async def verify_mexc():
    print("Testing MEXC API with 500 limit...")
    data = await fetch_mexc_kline("BTC_USDT", interval="5m", limit=500)
    if data:
        print(f"Successfully fetched {len(data)} data points.")
        print(f"Sample: {data[-1]}")
        assert len(data) >= 100
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    asyncio.run(verify_mexc())
