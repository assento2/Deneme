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
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload)
    except Exception as e:
        logger.error(f"Telegram notification failed: {e}")

def format_notification(event: dict) -> str:
    size_val = event.get('size', 0)
    lev_val = event.get('leverage', 10)
    lev_size = event.get('leveraged_size', 0)
    size_str = f"<b>${size_val}</b> ({lev_val}X <b>${lev_size}</b>)"

    if event["type"] == "ENTRY":
        return (
            f"<b>🚀 YENİ İŞLEM / NEW ENTRY</b>\n\n"
            f"🤖 Ajan: <code>{event['agent']}</code>\n"
            f"📈 Parite: <b>{event['symbol']}</b>\n"
            f"💰 <b>İşlem Büyüklüğü:</b> {size_str}\n"
            f"↕️ Yön: LONG (BOTTOM HUNT)\n"
            f"💵 Giriş Fiyatı: <code>{event['price']}</code>\n"
            f"🎯 <b>Hedef (TP):</b> <code>{event.get('tp', '--')}</code>\n"
            f"🛡️ <b>Durdurma (SL):</b> <code>{event.get('sl', '--')}</code>\n"
            f"✨ Güven: %{event['confidence']}"
        )
    elif event["type"] == "EXIT":
        emoji = "✅" if event["success"] else "❌"
        return (
            f"<b>{emoji} İŞLEM KAPATILDI / EXIT</b>\n\n"
            f"🤖 Ajan: <code>{event['agent']}</code>\n"
            f"📈 Parite: <b>{event['symbol']}</b>\n"
            f"💰 <b>İşlem Büyüklüğü:</b> {size_str}\n"
            f"📥 <b>Giriş Fiyatı:</b> <code>{event['entry_price']}</code>\n"
            f"📤 <b>Çıkış Fiyatı:</b> <code>{event['exit_price']}</code>\n"
            f"💰 Kar/Zarar: <b>${event['net_profit_loss']}</b>\n"
            f"📊 Yüzde: %{event['profit_pct']}\n"
            f"💸 Fees: ${event['fees']}\n"
            f"💡 Sebep: {event['reasoning']}"
        )
    return ""

def format_daily_report(stats: dict) -> str:
    emoji = "📈" if stats['profit'] >= 0 else "📉"
    comments = stats.get('comments', [])
    comment_str = "\n".join([f"• {c}" for c in comments])

    return (
        f"<b>📊 GÜNLÜK ÖZET RAPORU / DAILY REPORT</b>\n"
        f"<i>Tarih: {stats['date']}</i>\n\n"
        f"🌐 <b>Piyasa Hakimiyeti & Analiz:</b>\n"
        f"• BTC.D: %{stats.get('btc_d', '--')}\n"
        f"• ETH.D: %{stats.get('eth_d', '--')}\n"
        f"• USD.D: %{stats.get('usd_d', '--')}\n"
        f"<i>Yorum: {stats.get('dom_comment', 'Piyasa yapısı inceleniyor.')}</i>\n\n"
        f"✅ Toplam İşlem: <b>{stats['total_trades']}</b>\n"
        f"🎯 Başarı Oranı: <b>%{stats['win_rate']}</b>\n"
        f"{emoji} Net Kar/Zarar: <b>${stats['profit']}</b>\n"
        f"💰 Toplam Sermaye: <b>${stats['total_balance']}</b>\n\n"
        f"🔍 <b>Yapay Zeka Analiz Notları:</b>\n{comment_str}\n\n"
        f"⚡ <i>Aegis-Omni V5 Precision Hunter Engine</i>"
    )
