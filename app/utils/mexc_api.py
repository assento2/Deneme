import httpx
import logging
import random

logger = logging.getLogger(__name__)

async def fetch_all_symbols():
    """Fetches all available USDT-M trading pairs from MEXC."""
    url = "https://contract.mexc.com/api/v1/contract/detail"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            data = response.json()
            if data.get("success"):
                # Filter for USDT pairs and active ones
                return [s["symbol"] for s in data["data"] if s["quoteCoin"] == "USDT" and s["state"] == 0]
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
                for i in range(max(0, len(times) - limit), len(times)):
                    result.append({
                        "time": times[i] * 1000, # Normalize to ms if needed by frontend, wait MEXC is already in seconds mostly but check
                        "open": float(opens[i]), "close": float(closes[i]),
                        "high": float(highs[i]), "low": float(lows[i]), "vol": float(vols[i])
                    })
                return result
            return None
        except Exception as e:
            logger.error(f"Failed to fetch MEXC data for {symbol}: {e}")
            return None

async def market_scanner():
    """Returns all available symbols to fulfill 'scan all projects' requirement."""
    all_symbols = await fetch_all_symbols()
    # Shuffle or rank them to give different agents different opportunities
    random.shuffle(all_symbols)
    return all_symbols
