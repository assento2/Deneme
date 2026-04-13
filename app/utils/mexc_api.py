import httpx
import logging
import random
import asyncio

logger = logging.getLogger(__name__)

async def fetch_ranked_symbols(limit=50):
    """Fetches all USDT-M contracts and ranks them by 24h volume/volatility."""
    url = "https://contract.mexc.com/api/v1/contract/ticker"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            data = response.json()
            if data.get("success"):
                tickers = data["data"]
                # Filter for USDT pairs and valid volume
                valid = [t for t in tickers if t["symbol"].endswith("_USDT") and t["volume24"] > 1000]

                # Rank by Volume * Abs(Change) to find volatile trending projects
                for t in valid:
                    t["score"] = float(t["amount24"]) * abs(float(t["riseFallRate"]))

                ranked = sorted(valid, key=lambda x: x["score"], reverse=True)
                return [r["symbol"] for r in ranked[:limit]]
            return ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT"]
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT"]

async def fetch_mexc_kline(symbol="BTC_USDT", interval="Min5", limit=100):
    # Mapping for common intervals
    mapping = {
        "1m": "Min1",
        "5m": "Min5",
        "15m": "Min15",
        "1h": "Min60",
        "4h": "Hour4",
        "1d": "Day1"
    }
    mexc_interval = mapping.get(interval, interval)

    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {"interval": mexc_interval}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            data = response.json()
            if data.get("success"):
                raw = data["data"]
                times, opens, closes, highs, lows, vols = raw["time"], raw["open"], raw["close"], raw["high"], raw["low"], raw["vol"]
                result = []
                # MEXC returns data in chronological order (oldest first)
                count = len(times)
                start_idx = max(0, count - limit)
                for i in range(start_idx, count):
                    result.append({
                        "time": int(times[i]), # Seconds
                        "open": float(opens[i]),
                        "close": float(closes[i]),
                        "high": float(highs[i]),
                        "low": float(lows[i]),
                        "vol": float(vols[i])
                    })
                return result
            return None
        except Exception as e:
            logger.error(f"Failed to fetch MEXC data for {symbol}: {e}")
            return None

async def market_scanner():
    return await fetch_ranked_symbols(limit=100)
