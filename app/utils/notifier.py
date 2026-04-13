import httpx
import os
import logging

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def send_telegram_msg(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not found. Skipping notification.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload)
    except Exception as e:
        logger.error(f"Telegram notification failed: {e}")

def format_notification(event: dict) -> str:
    if event["type"] == "ENTRY":
        return (
            f"<b>🚀 YENİ İŞLEM / NEW ENTRY</b>\n"
            f"🤖 Ajan: {event['agent']}\n"
            f"📈 Parite: {event['symbol']}\n"
            f"↕️ Yön: {event['side']}\n"
            f"💰 Fiyat: {event['price']}\n"
            f"🎯 Güven: %{event['confidence']}"
        )
    else: # EXIT
        emoji = "✅" if event["success"] else "❌"
        return (
            f"<b>{emoji} İŞLEM KAPATILDI / EXIT</b>\n"
            f"🤖 Ajan: {event['agent']}\n"
            f"📈 Parite: {event['symbol']}\n"
            f"💰 Kar/Zarar: ${event['net_profit_loss']}\n"
            f"📊 Yüzde: %{event['profit_pct']}\n"
            f"💡 Sebep: {event['reasoning']}"
        )
