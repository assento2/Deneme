import httpx
import logging

logger = logging.getLogger(__name__)

async def fetch_mexc_kline(symbol="BTC_USDT", interval="Min5", limit=100):
    """
    Fetches kline data from MEXC Futures API.
    Docs: https://mexcdevelop.github.io/apidocs/contract_v1_en/#k-line-data
    """
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {
        "interval": interval
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                # Data structure: time, open, close, high, low, vol, amount
                # The API returns them as lists in the 'data' object
                # We'll zip them into a more usable format
                raw = data["data"]
                times = raw["time"]
                opens = raw["open"]
                closes = raw["close"]
                highs = raw["high"]
                lows = raw["low"]

                # We take the last 'limit' items
                result = []
                for i in range(max(0, len(times) - limit), len(times)):
                    result.append({
                        "time": times[i],
                        "open": opens[i],
                        "close": closes[i],
                        "high": highs[i],
                        "low": lows[i]
                    })
                return result
            else:
                logger.error(f"MEXC API error: {data.get('code')}")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch MEXC data: {e}")
            return None
