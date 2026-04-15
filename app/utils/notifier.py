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
            f"<b>🚀 YENİ İŞLEM / NEW ENTRY</b>\n\n"
            f"🤖 Ajan: <code>{event['agent']}</code>\n"
            f"📈 Parite: <b>{event['symbol']}</b>\n"
            f"↕️ Yön: {event['side']}\n"
            f"💰 Fiyat: {event['price']}\n"
            f"🎯 Güven: %{event['confidence']}"
        )
    elif event["type"] == "EXIT":
        emoji = "✅" if event["success"] else "❌"
        return (
            f"<b>{emoji} İŞLEM KAPATILDI / EXIT</b>\n\n"
            f"🤖 Ajan: <code>{event['agent']}</code>\n"
            f"📈 Parite: <b>{event['symbol']}</b>\n"
            f"💰 Kar/Zarar: <b>${event['net_profit_loss']}</b>\n"
            f"📊 Yüzde: %{event['profit_pct']}\n"
            f"💸 Fees: ${event['fees']}\n"
            f"💡 Sebep: {event['reasoning']}"
        )
    return ""

def format_daily_report(stats: dict) -> str:
    emoji = "📈" if stats['profit'] >= 0 else "📉"
    return (
        f"<b>📊 GÜNLÜK ÖZET RAPORU / DAILY REPORT</b>\n"
        f"<i>Tarih: {stats['date']}</i>\n\n"
        f"✅ Toplam İşlem: <b>{stats['total_trades']}</b>\n"
        f"🎯 Başarı Oranı: <b>%{stats['win_rate']}</b>\n"
        f"{emoji} Net Kar/Zarar: <b>${stats['profit']}</b>\n"
        f"💰 Toplam Sermaye: <b>${stats['total_balance']}</b>\n\n"
        f"⚡ <i>Aegis-Omni V4 24/7 Otonom Sistem</i>"
    )
