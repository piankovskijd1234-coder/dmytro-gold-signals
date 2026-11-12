import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes
)

# ============================================================
# DMYTRO GOLD SIGNALS — TDA SYNC INTRADAY
# Manual alerts only. The bot never opens trades.
# Yahoo Finance data are indicative; CFDs/broker prices may differ.
# ============================================================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
SCAN_MINUTES = int(os.getenv("SCAN_MINUTES", "15"))
MIN_RR = 2.0
DATA_PERIOD = "60d"

SUBSCRIBERS_FILE = Path("subscribers.json")
ALERTS_FILE = Path("alerts.json")

SYMBOLS = {
    "BTCUSD": {"ticker": "BTC-USD", "kind": "crypto"},
    "EURUSD": {"ticker": "EURUSD=X", "kind": "fx"},
    "GBPUSD": {"ticker": "GBPUSD=X", "kind": "fx"},
    "EURJPY": {"ticker": "EURJPY=X", "kind": "fx"},
    "AUDUSD": {"ticker": "AUDUSD=X", "kind": "fx"},
    "XAGUSD": {"ticker": "XAGUSD=X", "fallback": "SI=F", "kind": "metal"},
    "EURAUD": {"ticker": "EURAUD=X", "kind": "fx"},
    "USDCHF": {"ticker": "CHF=X", "kind": "fx"},
    "GBPAUD": {"ticker": "GBPAUD=X", "kind": "fx"},
    "GER40": {"ticker": "^GDAXI", "kind": "index"},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("dmytro-gold-signals")

def read_json(path: Path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except (OSError, ValueError, TypeError):
        log.exception("Could not read %s", path)
    return default

def write_json(path: Path, data):
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        log.exception("Could not write %s", path)

SUBSCRIBERS = set(int(x) for x in read_json(SUBSCRIBERS_FILE, []))
ALERTED = set(read_json(ALERTS_FILE, []))

def add_subscriber(chat_id: int):
    SUBSCRIBERS.add(chat_id)
    write_json(SUBSCRIBERS_FILE, sorted(SUBSCRIBERS))

def normalize_frame(frame):
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.copy()
    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    return frame

def download_data(ticker: str, interval: str, period: str = DATA_PERIOD):
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
            timeout=15,
        )
        return normalize_frame(raw)
    except Exception as exc:
        log.warning("Yahoo download failed for %s %s: %s", ticker, interval, exc)
        return pd.DataFrame()

def download_symbol_data(symbol: str):
    cfg = SYMBOLS[symbol]
    ticker = cfg["ticker"]
    # Weekly and daily for higher-timeframe bias; 1h is resampled to 4h.
    weekly = download_data(ticker, "1wk", "2y")
    daily = download_data(ticker, "1d", "1y")
    hourly = download_data(ticker, "1h", "60d")
    m15 = download_data(ticker, "15m", "60d")

    if cfg.get("fallback") and (weekly.empty or daily.empty or hourly.empty):
        fallback = cfg["fallback"]
        weekly = weekly if not weekly.empty else download_data(fallback, "1wk", "2y")
        daily = daily if not daily.empty else download_data(fallback, "1d", "1y")
        hourly = hourly if not hourly.empty else download_data(fallback, "1h", "60d")
        m15 = m15 if not m15.empty else download_data(fallback, "15m", "60d")

    h4 = pd.DataFrame()
    if not hourly.empty:
        h4 = hourly.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last",
            **({"volume": "sum"} if "volume" in hourly.columns else {})
        }).dropna(subset=["open", "high", "low", "close"])
    return weekly, daily, h4, hourly, m15

def bias(frame: pd.DataFrame):
    if frame.empty or len(frame) < 30:
        return None
    close = frame["close"]
    fast = close.ewm(span=20, adjust=False).mean()
    slow = close.ewm(span=50, adjust=False).mean()
    last = float(close.iloc[-1])
    if fast.iloc[-1] > slow.iloc[-1] and last > fast.iloc[-1]:
        return "BUY"
    if fast.iloc[-1] < slow.iloc[-1] and last < fast.iloc[-1]:
        return "SELL"
    return None

def find_poi(frame: pd.DataFrame, direction: str):
    """Approximate recent H4 demand/supply zone from recent candle range."""
    if frame.empty or len(frame) < 12:
        return None
    recent = frame.iloc[-12:-1]
    last = frame.iloc[-1]
    if direction == "BUY":
        candle = recent.loc[recent["close"] < recent["open"]]
        if candle.empty:
            return None
        row = candle.iloc[-1]
        low, high = float(row["low"]), float(max(row["open"], row["close"]))
        price = float(last["close"])
        if low <= price <= high * 1.003:
            return (low, high)
    else:
        candle = recent.loc[recent["close"] > recent["open"]]
        if candle.empty:
            return None
        row = candle.iloc[-1]
        low, high = float(min(row["open"], row["close"])), float(row["high"])
        price = float(last["close"])
        if low * 0.997 <= price <= high:
            return (low, high)
    return None

def h1_bos_retest(hourly: pd.DataFrame, direction: str):
    if hourly.empty or len(hourly) < 25:
        return None
    # Exclude current candle when defining the structure level.
    hist = hourly.iloc[-21:-2]
    last = hourly.iloc[-1]
    prev = hourly.iloc[-2]
    close = float(last["close"])
    if direction == "BUY":
        level = float(hist["high"].max())
        broke = float(prev["close"]) > level or close > level
        retest = float(last["low"]) <= level * 1.0015 and close >= level
        if broke and retest:
            return level
    else:
        level = float(hist["low"].min())
        broke = float(prev["close"]) < level or close < level
        retest = float(last["high"]) >= level * 0.9985 and close <= level
        if broke and retest:
            return level
    return None

def m15_confirm(m15: pd.DataFrame, direction: str):
    if m15.empty or len(m15) < 5:
        return False
    last = m15.iloc[-1]
    prev = m15.iloc[-2]
    if direction == "BUY":
        return float(last["close"]) > float(last["open"]) and float(last["close"]) > float(prev["high"])
    return float(last["close"]) < float(last["open"]) and float(last["close"]) < float(prev["low"])

def make_signal(symbol: str):
    weekly, daily, h4, hourly, m15 = download_symbol_data(symbol)
    if any(x.empty for x in (weekly, daily, h4, hourly, m15)):
        return None, "немає даних на одному з таймфреймів"

    wb = bias(weekly)
    db = bias(daily)
    if wb is None or db is None or wb != db:
        return None, "Weekly/Daily Bias не синхронізовані"

    direction = wb
    poi = find_poi(h4, direction)
    if poi is None:
        return None, "ціна не в алгоритмічній H4 POI"

    bos_level = h1_bos_retest(hourly, direction)
    if bos_level is None:
        return None, "немає підтвердження H1 BOS + Retest"

    if not m15_confirm(m15, direction):
        return None, "немає підтвердження M15"

    entry = float(hourly["close"].iloc[-1])
    recent = hourly.iloc[-12:]
    if direction == "BUY":
        sl = min(float(recent["low"].min()), float(poi[0]))
        risk = entry - sl
        if risk <= 0:
            return None, "некоректний стоп для BUY"
        tp = entry + MIN_RR * risk
    else:
        sl = max(float(recent["high"].max()), float(poi[1]))
        risk = sl - entry
        if risk <= 0:
            return None, "некоректний стоп для SELL"
        tp = entry - MIN_RR * risk

    if tp <= 0:
        return None, "некоректний розрахунок TP"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    signal_id = f"{symbol}-{direction}-{stamp}"
    return {
        "id": signal_id, "symbol": symbol, "direction": direction,
        "entry": entry, "sl": sl, "tp": tp, "rr": MIN_RR,
        "poi": poi, "time": datetime.now(timezone.utc).isoformat()
    }, None

def fmt_price(value):
    if abs(value) >= 1000:
        return f"{value:.2f}"
    if abs(value) >= 10:
        return f"{value:.4f}"
    return f"{value:.5f}"

def format_signal(s):
    arrow = "🟢" if s["direction"] == "BUY" else "🔴"
    return (
        f"{arrow} TDA SYNC INTRADAY — {s['symbol']} {s['direction']}\\n"
        f"Entry: {fmt_price(s['entry'])}\\n"
        f"SL: {fmt_price(s['sl'])}\\n"
        f"TP: {fmt_price(s['tp'])}\\n"
        f"RR: 1:{s['rr']:.1f}\\n"
        f"TF: W1/D1 → H4 POI → H1 BOS+Retest → M15 Confirm\\n"
        f"Час UTC: {s['time'][11:16]}\\n"
        f"⚠️ Аналітичний сигнал, не фінансова рекомендація. Перевір котирування у своєму брокері."
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    add_subscriber(chat_id)
    await update.message.reply_text(
        "Dmytro Gold Signals активний.\\n"
        "Ти підписаний на сигнали. Бот не відкриває угоди.\\n"
        "Команди: /status /symbols /scan"
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Бот активний. Інструментів: {len(SYMBOLS)}.\\n"
        f"Модель: TDA Sync Intraday. W1/D1 Bias → H4 POI → "
        f"H1 BOS+Retest → M15 Confirm.\\n"
        f"Один TP; мінімальний RR 1:2.\\n"
        f"Сканування: кожні {SCAN_MINUTES} хв.\\n"
        f"Підписників: {len(SUBSCRIBERS)}.\\n"
        "Дані Yahoo Finance; POI/Order Flow — алгоритмічні наближення, не біржовий Delta."
    )

async def symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Інструменти: " + ", ".join(SYMBOLS.keys()))

async def scan_all(application: Application, notify: bool = True):
    results = []
    errors = []
    for symbol in SYMBOLS:
        try:
            signal, reason = await asyncio.to_thread(make_signal, symbol)
            if signal:
                results.append(signal)
            else:
                errors.append(f"• {symbol}: {reason}")
        except Exception as exc:
            log.exception("Scan error for %s", symbol)
            errors.append(f"• {symbol}: технічна помилка ({type(exc).__name__})")

    sent = 0
    for signal in results:
        if signal["id"] in ALERTED:
            continue
        msg = format_signal(signal)
        for chat_id in list(SUBSCRIBERS):
            try:
                await application.bot.send_message(chat_id=chat_id, text=msg)
                sent += 1
            except Exception:
                log.exception("Could not send signal to chat %s", chat_id)
        ALERTED.add(signal["id"])
        write_json(ALERTS_FILE, sorted(ALERTED))

    if not notify:
        return sent, errors, len(results)

    return sent, errors, len(results)

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Сканую {len(SYMBOLS)} інструментів. Це може тривати кілька хвилин."
    )
    sent, errors, found = await scan_all(context.application, notify=False)
    summary = [
        "Сканування завершено.",
        f"Інструментів: {len(SYMBOLS)}.",
        f"Знайдено сигналів: {found}.",
        f"Надіслано повідомлень: {sent}.",
    ]
    if errors:
        summary.append("Причини відсутності сигналів:\\n" + "\\n".join(errors[:10]))
    await update.message.reply_text("\\n".join(summary))

async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan_all(context.application, notify=False)

async def post_init(application: Application):
    if application.job_queue:
        application.job_queue.run_repeating(
            scheduled_scan,
            interval=SCAN_MINUTES * 60,
            first=30,
            name="scheduled_market_scan",
        )
    else:
        log.error("JobQueue is unavailable. Install python-telegram-bot[job-queue].")

def main():
    if not TOKEN:
        raise RuntimeError("Встанови TELEGRAM_BOT_TOKEN у Railway → Variables.")
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("symbols", symbols))
    app.add_handler(CommandHandler("scan", scan_command))
    log.info("Starting Dmytro Gold Signals with %d symbols", len(SYMBOLS))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
