"""
Dmytro Gold Signals — TDA Sync + News Filter
Telegram scanner only. It does NOT place trades.
Data: Yahoo Finance OHLCV (approximate; not broker/exchange order flow).
News: Finnhub economic calendar API.
"""
import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
SCAN_MINUTES = max(5, int(os.getenv("SCAN_MINUTES", "15")))
NEWS_LEAD_MINUTES = max(5, int(os.getenv("NEWS_LEAD_MINUTES", "30")))
NEWS_BLACKOUT_MINUTES = max(0, int(os.getenv("NEWS_BLACKOUT_MINUTES", "30")))
NEWS_LOOKAHEAD_HOURS = max(1, int(os.getenv("NEWS_LOOKAHEAD_HOURS", "48")))
MIN_RR = max(0.1, float(os.getenv("MIN_RR", "0.8")))
NEWS_MIN_IMPACT = os.getenv("NEWS_MIN_IMPACT", "high").lower()

SUB_FILE = Path("subscribers.json")
ALERT_FILE = Path("alerts.json")

SYMBOLS = {
    "BTCUSD": "BTC-USD",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "EURJPY": "EURJPY=X",
    "AUDUSD": "AUDUSD=X",
    "XAGUSD": "XAGUSD=X",
    "EURAUD": "EURAUD=X",
    "USDCHF": "CHF=X",
    "GBPAUD": "GBPAUD=X",
    "GER40": "^GDAXI",
}
CURRENCIES = {
    "BTCUSD": ["USD"], "EURUSD": ["EUR", "USD"], "GBPUSD": ["GBP", "USD"],
    "EURJPY": ["EUR", "JPY"], "AUDUSD": ["AUD", "USD"], "XAGUSD": ["USD"],
    "EURAUD": ["EUR", "AUD"], "USDCHF": ["USD", "CHF"], "GBPAUD": ["GBP", "AUD"],
    "GER40": ["EUR"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmytro-gold-signals")
SUBSCRIBERS = set()
ALERTED = set()
NEWS_CACHE = {"fetched_at": None, "events": []}
NEWS_ALERTED = set()

def read_json(path, default):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return default

def save_json(path, data):
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        log.exception("Could not save %s", path)

def load_state():
    global SUBSCRIBERS, ALERTED, NEWS_ALERTED
    SUBSCRIBERS = {int(x) for x in read_json(SUB_FILE, [])}
    ALERTED = set(read_json(ALERT_FILE, []))
    NEWS_ALERTED = set(read_json(Path("news_alerted.json"), []))

def get_data(ticker, interval, period):
    try:
        frame = yf.download(
            ticker, interval=interval, period=period, progress=False,
            auto_adjust=False, threads=False, timeout=20
        )
        if frame is None or frame.empty:
            return pd.DataFrame()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
        required = ["open", "high", "low", "close"]
        if not all(c in frame.columns for c in required):
            return pd.DataFrame()
        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        frame = frame[required + ["volume"]].apply(pd.to_numeric, errors="coerce")
        return frame.dropna(subset=required).sort_index()
    except Exception:
        log.exception("Yahoo Finance error: %s %s", ticker, interval)
        return pd.DataFrame()

def resample(frame, rule):
    if frame.empty:
        return pd.DataFrame()
    return frame.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"])

def trend_bias(frame):
    if frame.empty or len(frame) < 55:
        return None
    close = frame["close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    last = float(close.iloc[-1])
    if last > ema20 > ema50:
        return "BUY"
    if last < ema20 < ema50:
        return "SELL"
    return None

def find_h4_poi(frame, side):
    if frame.empty or len(frame) < 15:
        return None
    price = float(frame.close.iloc[-1])
    recent = frame.iloc[-16:-1]
    for _, candle in recent.iloc[::-1].iterrows():
        o, c = float(candle.open), float(candle.close)
        lo, hi = float(candle.low), float(candle.high)
        if side == "BUY" and c < o:
            zone = (lo, max(o, c))
            if zone[0] <= price <= zone[1] * 1.008:
                return zone
        if side == "SELL" and c > o:
            zone = (min(o, c), hi)
            if zone[0] * 0.992 <= price <= zone[1]:
                return zone
    # Conservative recent-range POI fallback.
    recent = frame.iloc[-8:-1]
    if side == "BUY":
        zone = (float(recent.low.min()), float(recent.low.quantile(0.35)))
        if zone[0] <= price <= zone[1] * 1.01:
            return zone
    else:
        zone = (float(recent.high.quantile(0.65)), float(recent.high.max()))
        if zone[0] * 0.99 <= price <= zone[1]:
            return zone
    return None

def h1_bos_retest(frame, side):
    if frame.empty or len(frame) < 25:
        return None
    prior = frame.iloc[-22:-2]
    previous, last = frame.iloc[-2], frame.iloc[-1]
    if side == "BUY":
        level = float(prior.high.max())
        broke = float(previous.close) > level or float(last.close) > level
        retested = float(last.low) <= level * 1.002 and float(last.close) >= level
    else:
        level = float(prior.low.min())
        broke = float(previous.close) < level or float(last.close) < level
        retested = float(last.high) >= level * 0.998 and float(last.close) <= level
    return level if broke and retested else None

def m15_confirm(frame, side):
    if frame.empty or len(frame) < 3:
        return False
    last, prev = frame.iloc[-1], frame.iloc[-2]
    if side == "BUY":
        return float(last.close) > float(last.open) and float(last.close) > float(prev.high)
    return float(last.close) < float(last.open) and float(last.close) < float(prev.low)

def structural_target(frame, side, entry):
    if frame.empty or len(frame) < 25:
        return None
    recent = frame.iloc[-62:-2]
    if side == "BUY":
        levels = sorted(float(x) for x in recent.high if float(x) > entry)
        return levels[0] if levels else None
    levels = sorted((float(x) for x in recent.low if float(x) < entry), reverse=True)
    return levels[0] if levels else None

def build_signal(symbol):
    ticker = SYMBOLS[symbol]
    weekly = get_data(ticker, "1wk", "2y")
    daily = get_data(ticker, "1d", "1y")
    hourly_raw = get_data(ticker, "1h", "60d")
    m15_raw = get_data(ticker, "15m", "60d")
    if any(x.empty for x in (weekly, daily, hourly_raw, m15_raw)):
        return None, "немає котирувань на одному з таймфреймів"

    weekly = weekly.iloc[:-1]
    daily = daily.iloc[:-1]
    hourly = hourly_raw.iloc[:-1]
    m15 = m15_raw.iloc[:-1]
    h4 = resample(hourly_raw, "4h").iloc[:-1]
    w_bias, d_bias = trend_bias(weekly), trend_bias(daily)

    if w_bias and d_bias and w_bias != d_bias:
        return None, "Weekly і Daily суперечать один одному"
    side = d_bias or w_bias
    if not side:
        return None, "немає чіткого Weekly/Daily напрямку"
    h4_bias = trend_bias(h4)
    if h4_bias and h4_bias != side:
        return None, "H4 суперечить старшому контексту"

    poi = find_h4_poi(h4, side)
    if poi is None:
        return None, "ціна поза H4 POI"
    if h1_bos_retest(hourly, side) is None:
        return None, "немає H1 BOS + Retest"
    if not m15_confirm(m15, side):
        return None, "немає M15 підтвердження"

    entry = float(m15.close.iloc[-1])
    if side == "BUY":
        sl = min(float(hourly.low.iloc[-12:].min()), poi[0])
        risk = entry - sl
    else:
        sl = max(float(hourly.high.iloc[-12:].max()), poi[1])
        risk = sl - entry
    if risk <= 0:
        return None, "некоректний структурний Stop Loss"

    tp = structural_target(hourly, side, entry)
    if tp is None:
        return None, "не знайдено структурну ціль попереду"
    reward = tp - entry if side == "BUY" else entry - tp
    rr = reward / risk
    if reward <= 0 or rr < MIN_RR:
        return None, f"потенціал малий: RR 1:{rr:.2f}, мінімум 1:{MIN_RR:.2f}"

    # Quality score: hard confirmations remain mandatory; score distinguishes
    # stronger alignment from merely acceptable setups.
    score = 0
    score += 25 if w_bias == side and d_bias == side else 18 if d_bias == side else 12
    score += 15 if h4_bias == side else 8
    score += 20  # price is inside the H4 POI
    score += 20  # H1 BOS + retest confirmed
    score += 20  # M15 candle confirmation
    # Reward a clean impulse candle, but never treat volume as true order flow.
    last_m15 = m15.iloc[-1]
    candle_range = max(float(last_m15.high) - float(last_m15.low), 1e-12)
    body_ratio = abs(float(last_m15.close) - float(last_m15.open)) / candle_range
    if body_ratio >= 0.65:
        score += 5
    score = min(score, 100)
    if score < 80:
        return None, f"якість сетапу {score}/100 — нижче порогу 80"

    return {
        "symbol": symbol, "side": side, "entry": entry, "sl": sl, "tp": tp,
        "rr": rr, "score": score, "time": str(m15.index[-1]),
        "w": w_bias or "нейтральний", "d": d_bias or "нейтральний",
        "h4": h4_bias or "нейтральний"
    }, None

def fetch_news(force=False):
    global NEWS_CACHE
    now = datetime.now(timezone.utc)
    cached_at = NEWS_CACHE.get("fetched_at")
    if not force and cached_at and (now - cached_at).total_seconds() < 300:
        return NEWS_CACHE.get("events", [])
    if not FINNHUB_API_KEY:
        return []
    start = now.date().isoformat()
    end = (now + timedelta(hours=NEWS_LOOKAHEAD_HOURS)).date().isoformat()
    try:
        response = requests.get(
            "https://finnhub.io/api/v1/calendar/economic",
            params={"from": start, "to": end, "token": FINNHUB_API_KEY},
            timeout=20
        )
        response.raise_for_status()
        payload = response.json()
        events = payload.get("economicCalendar", [])
        normalized = []
        for event in events:
            raw_time = event.get("time")
            if not raw_time:
                continue
            try:
                event_time = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
                else:
                    event_time = event_time.astimezone(timezone.utc)
            except ValueError:
                continue
            normalized.append({
                "time": event_time,
                "country": str(event.get("country", "")).upper(),
                "currency": str(event.get("currency", "")).upper(),
                "event": str(event.get("event", "Economic event")),
                "impact": str(event.get("impact", "")).lower(),
                "actual": event.get("actual"),
                "estimate": event.get("estimate"),
                "prev": event.get("prev")
            })
        NEWS_CACHE = {"fetched_at": now, "events": normalized}
        return normalized
    except Exception:
        log.exception("Finnhub calendar request failed")
        return NEWS_CACHE.get("events", [])

def event_currency(event):
    currency = event.get("currency", "")
    if currency:
        return currency
    country = event.get("country", "")
    return {"UNITED STATES": "USD", "UNITED KINGDOM": "GBP",
            "EURO AREA": "EUR", "GERMANY": "EUR", "JAPAN": "JPY",
            "AUSTRALIA": "AUD", "SWITZERLAND": "CHF", "CANADA": "CAD"}.get(country, "")

def impact_is_high(event):
    impact = event.get("impact", "")
    if NEWS_MIN_IMPACT == "all":
        return impact in ("high", "medium", "low")
    if NEWS_MIN_IMPACT == "medium":
        return impact in ("high", "medium")
    return impact == "high"

def relevant_events(symbol, events):
    currencies = set(CURRENCIES.get(symbol, []))
    return [e for e in events if event_currency(e) in currencies and impact_is_high(e)]

def news_block(symbol, now=None):
    now = now or datetime.now(timezone.utc)
    events = relevant_events(symbol, fetch_news())
    for event in events:
        delta = (event["time"] - now).total_seconds() / 60
        if -NEWS_BLACKOUT_MINUTES <= delta <= NEWS_BLACKOUT_MINUTES:
            return event
    return None

def fmt_price(symbol, value):
    if symbol in ("BTCUSD", "GER40"):
        return f"{value:.2f}"
    if "JPY" in symbol:
        return f"{value:.3f}"
    if symbol == "XAGUSD":
        return f"{value:.3f}"
    return f"{value:.5f}" if abs(value) < 10 else f"{value:.4f}"

def signal_text(s):
    emoji = "🟢" if s["side"] == "BUY" else "🔴"
    return (
        f"{emoji} {s['side']} — {s['symbol']}\n"
        "Стратегія: TDA Sync Intraday\n"
        f"W1: {s['w']} | D1: {s['d']} | H4: {s['h4']}\n"
        f"Entry: {fmt_price(s['symbol'], s['entry'])}\n"
        f"SL: {fmt_price(s['symbol'], s['sl'])}\n"
        f"TP: {fmt_price(s['symbol'], s['tp'])}\n"
        f"Фактичний RR: 1:{s['rr']:.2f}\n"
        "Підтвердження: H1 BOS + Retest → M15 Confirm\n"
        f"Час свічки: {s['time']}\n"
        "TP — найближча знайдена структурна ціль; RR гнучкий.\n"
        "⚠️ Yahoo Finance може мати затримку. Перевір ціни у своєму брокері."
    )

def news_text(event, prefix="📰 NEWS"):
    time_text = event["time"].strftime("%d.%m %H:%M UTC")
    currency = event_currency(event) or event.get("country", "—")
    return (
        f"{prefix}\n"
        f"Валюта: {currency}\n"
        f"Подія: {event['event']}\n"
        f"Вплив: {event['impact'] or 'невідомий'}\n"
        f"Час: {time_text}\n"
        f"Факт: {event.get('actual') if event.get('actual') is not None else 'ще немає'} | "
        f"Прогноз: {event.get('estimate') if event.get('estimate') is not None else '—'} | "
        f"Попереднє: {event.get('prev') if event.get('prev') is not None else '—'}"
    )

async def send_to_subscribers(app, text):
    for chat_id in list(SUBSCRIBERS):
        try:
            await app.bot.send_message(chat_id=chat_id, text=text)
        except Exception:
            log.exception("Telegram send failed for chat %s", chat_id)

async def check_news_alerts(app):
    now = datetime.now(timezone.utc)
    events = fetch_news()
    for event in events:
        if not impact_is_high(event):
            continue
        mins = (event["time"] - now).total_seconds() / 60
        if 0 < mins <= NEWS_LEAD_MINUTES:
            key = f"{event['time'].isoformat()}|{event_currency(event)}|{event['event']}"
            if key not in NEWS_ALERTED:
                await send_to_subscribers(app, news_text(event, "⚠️ NEWS ALERT — скоро важлива новина"))
                NEWS_ALERTED.add(key)
                save_json(Path("news_alerted.json"), sorted(NEWS_ALERTED)[-1000:])

async def run_scan(app, report_chat=None):
    await check_news_alerts(app)
    found, reasons = [], []
    for symbol in SYMBOLS:
        blocked = news_block(symbol)
        if blocked:
            reasons.append(f"{symbol}: NEWS blackout — {blocked['event']}")
            continue
        try:
            signal, reason = await asyncio.to_thread(build_signal, symbol)
            if signal:
                found.append(signal)
            elif reason:
                reasons.append(f"{symbol}: {reason}")
        except Exception as exc:
            log.exception("Scan failed for %s", symbol)
            reasons.append(f"{symbol}: помилка {type(exc).__name__}")

    for signal in found:
        key = f"{signal['symbol']}|{signal['side']}|{signal['time']}"
        if key in ALERTED:
            continue
        await send_to_subscribers(app, signal_text(signal))
        ALERTED.add(key)
        save_json(ALERT_FILE, sorted(ALERTED)[-2500:])

    if report_chat is not None:
        lines = [
            "Сканування завершено.",
            f"Інструментів: {len(SYMBOLS)} | Сигналів: {len(found)}",
            f"News filter: {'увімкнений' if FINNHUB_API_KEY else 'без API-ключа — не працює'}",
            "Причини відмови:"
        ]
        lines.extend(reasons[:15] if reasons else ["немає"])
        await app.bot.send_message(chat_id=report_chat, text="\n".join(lines))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)
    save_json(SUB_FILE, sorted(SUBSCRIBERS))
    await update.message.reply_text(
        "✅ Підписка активна. Бот надсилатиме сигнали та важливі новини.\n"
        "Команди: /status /scan /news /stop"
    )

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.discard(chat_id)
    save_json(SUB_FILE, sorted(SUBSCRIBERS))
    await update.message.reply_text("Підписку вимкнено.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Dmytro Gold Signals\n"
        f"Статус: працює\nІнструментів: {len(SYMBOLS)}\n"
        "Модель: W1/D1 Bias → H4 POI → H1 BOS+Retest → M15 Confirm\n"
        "TP: одна структурна ціль; RR гнучкий\n"
        f"Мінімальний RR-фільтр: 1:{MIN_RR:.2f}\n"
        f"Сканування: кожні {SCAN_MINUTES} хв\n"
        f"News calendar: {'Finnhub підключений' if FINNHUB_API_KEY else 'НЕ налаштований — додай FINNHUB_API_KEY'}\n"
        f"News alert: за {NEWS_LEAD_MINUTES} хв; блокування ±{NEWS_BLACKOUT_MINUTES} хв\n"
        "Джерело котирувань: Yahoo Finance; не справжній біржовий Order Flow."
    )

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 Сканую ринок і перевіряю новини…")
    await run_scan(context.application, update.effective_chat.id)

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    events = fetch_news(force=True)
    now = datetime.now(timezone.utc)
    upcoming = [e for e in events if e["time"] >= now and impact_is_high(e)]
    upcoming.sort(key=lambda e: e["time"])
    if not FINNHUB_API_KEY:
        await update.message.reply_text("❌ Finnhub не підключений. Додай FINNHUB_API_KEY у Railway Variables.")
    elif not upcoming:
        await update.message.reply_text("На найближчий період важливих подій не знайдено.")
    else:
        for event in upcoming[:12]:
            await update.message.reply_text(news_text(event, "📅 ЕКОНОМІЧНИЙ КАЛЕНДАР"))

async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    await run_scan(context.application)

async def post_init(app):
    if app.job_queue:
        app.job_queue.run_repeating(scheduled_scan, interval=SCAN_MINUTES * 60, first=20)
    else:
        log.error("JobQueue unavailable. Install python-telegram-bot[job-queue].")

def main():
    if not TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable")
    load_state()
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("news", news_command))
    log.info("Bot starting. Subscribers=%s symbols=%s", len(SUBSCRIBERS), len(SYMBOLS))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
