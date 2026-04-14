import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)

async def fetch_ranked_symbols(limit=100):
    """Fetches all USDT-M contracts and ranks them by 24h volume/volatility."""
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(3):
            try:
                response = await client.get(url)
                data = response.json()
                if data.get("success"):
                    tickers = data["data"]
                    valid = [t for t in tickers if t["symbol"].endswith("_USDT") and t["amount24"] > 50000]

                    for t in valid:
                        # Composite Score: Volume * Abs(24h Change)
                        t["score"] = float(t["amount24"]) * abs(float(t["riseFallRate"]))

                    ranked = sorted(valid, key=lambda x: x["score"], reverse=True)
                    return [r["symbol"] for r in ranked[:limit]]
            except Exception as e:
                await asyncio.sleep(1 * (attempt + 1))

        return ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT"]

async def fetch_mexc_kline(symbol="BTC_USDT", interval="Min5", limit=100):
    mapping = {
        "1m": "Min1", "5m": "Min5", "15m": "Min15",
        "1h": "Min60", "4h": "Hour4", "1d": "Day1"
    }
    mexc_interval = mapping.get(interval, interval)
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(3):
            try:
                response = await client.get(url, params={"interval": mexc_interval})
                data = response.json()
                if data.get("success"):
                    raw = data["data"]
                    times, opens, closes, highs, lows, vols = raw["time"], raw["open"], raw["close"], raw["high"], raw["low"], raw["vol"]
                    result = []
                    count = len(times)
                    start_idx = max(0, count - limit)
                    for i in range(start_idx, count):
                        result.append({
                            "time": int(times[i]), "open": float(opens[i]), "close": float(closes[i]),
                            "high": float(highs[i]), "low": float(lows[i]), "vol": float(vols[i])
                        })
                    return result
            except:
                await asyncio.sleep(0.5 * (attempt + 1))

        return []

async def market_scanner():
    return await fetch_ranked_symbols(limit=150)
