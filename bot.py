import os
import json
import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = "".join(os.getenv("TELEGRAM_BOT_TOKEN", "").split())
INTERVAL = max(5, int(os.getenv("SCAN_INTERVAL_MINUTES", "5")))
SUBSCRIBERS_FILE = "subscribers.json"
ALERTS_FILE = "alerts.json"
ACTIVE_FILE = "active_signals.json"

# Nine requested instruments; no MT5 connection is used.
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
}
FALLBACK_TICKERS = {"XAGUSD": ["SI=F"]}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmytro-gold-signals")
SUBSCRIBERS = set()
ALERTED = set()
ACTIVE_SIGNALS = {}


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return default


def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        log.exception("Cannot save %s", path)


def load_state():
    global SUBSCRIBERS, ALERTED, ACTIVE_SIGNALS
    SUBSCRIBERS = {int(x) for x in load_json(SUBSCRIBERS_FILE, [])}
    ALERTED = set(load_json(ALERTS_FILE, []))
    ACTIVE_SIGNALS = load_json(ACTIVE_FILE, {})


def download(ticker, period, interval):
    try:
        frame = yf.download(ticker, period=period, interval=interval, progress=False,
                            threads=False, auto_adjust=False)
        if frame is None or frame.empty:
            return pd.DataFrame()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
        required = ["open", "high", "low", "close"]
        if not all(c in frame.columns for c in required):
            return pd.DataFrame()
        frame = frame[required].apply(pd.to_numeric, errors="coerce").dropna()
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
        return frame.sort_index()
    except Exception:
        log.exception("Quote download failed: %s %s", ticker, interval)
        return pd.DataFrame()


def download_symbol(symbol, period, interval):
    candidates = [SYMBOLS[symbol]] + FALLBACK_TICKERS.get(symbol, [])
    for ticker in candidates:
        data = download(ticker, period, interval)
        if len(data) >= 30:
            if ticker != SYMBOLS[symbol]:
                log.warning("%s uses fallback %s; price may differ from broker", symbol, ticker)
            return data, ticker
    return pd.DataFrame(), candidates[0]


def trend_bias(frame):
    if frame.empty or len(frame) < 50:
        return "NONE"
    close = frame.close.astype(float)
    e20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    e50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]
    if price > e20 > e50:
        return "BUY"
    if price < e20 < e50:
        return "SELL"
    return "NONE"


def _fvg_at(df, i, side):
    """Return an FVG zone formed at i, or None. i is the third candle of the pattern."""
    if i < 2:
        return None
    left, right = df.iloc[i - 2], df.iloc[i]
    if side == "BUY" and float(left.high) < float(right.low):
        return (float(left.high), float(right.low))
    if side == "SELL" and float(left.low) > float(right.high):
        return (float(right.high), float(left.low))
    return None


def _fvg_status(df, formed_i, zone, side):
    """Assess subsequent candles; returns open/partial/filled."""
    low, high = zone
    later = df.iloc[formed_i + 1:]
    if later.empty:
        return "open"
    if side == "BUY":
        deepest = float(later.low.min())
        if deepest <= low:
            return "filled"
        if deepest < high:
            return "partially filled"
    else:
        highest = float(later.high.max())
        if highest >= high:
            return "filled"
        if highest > low:
            return "partially filled"
    return "open"


def detect_smc(h1_closed):
    """SMC proxy: liquidity sweep -> BOS/CHoCH -> still-valid FVG; SNR is contextual."""
    if h1_closed.empty or len(h1_closed) < 60:
        return None, "Недостатньо закритих H1-свічок (потрібно щонайменше 60)"
    df = h1_closed.copy().reset_index(drop=False)
    # Search recent 12 closed candles for a sweep, then structure shift and an unfilled/partial FVG.
    first = max(25, len(df) - 12)
    candidates = []
    for sweep_i in range(first, len(df) - 2):
        candle = df.iloc[sweep_i]
        prior = df.iloc[max(0, sweep_i - 20):sweep_i]
        if len(prior) < 12:
            continue
        prev_low, prev_high = float(prior.low.min()), float(prior.high.max())
        sweep_buy = float(candle.low) < prev_low and float(candle.close) > prev_low
        sweep_sell = float(candle.high) > prev_high and float(candle.close) < prev_high
        for side, swept in (("BUY", sweep_buy), ("SELL", sweep_sell)):
            if not swept:
                continue
            structure_i = None
            for j in range(sweep_i + 1, min(sweep_i + 6, len(df))):
                before = df.iloc[max(0, j - 5):j]
                if len(before) < 3:
                    continue
                if side == "BUY" and float(df.iloc[j].close) > float(before.high.max()):
                    structure_i = j
                    break
                if side == "SELL" and float(df.iloc[j].close) < float(before.low.min()):
                    structure_i = j
                    break
            if structure_i is None:
                continue
            # Find a matching FVG formed around the displacement/structure break.
            for fvg_i in range(max(2, structure_i - 2), min(len(df), structure_i + 3)):
                zone = _fvg_at(df, fvg_i, side)
                if zone is None:
                    continue
                state = _fvg_status(df, fvg_i, zone, side)
                if state == "filled":
                    continue
                last = df.iloc[-1]
                zlow, zhigh = zone
                # The latest closed candle must interact with the FVG zone, not be far away.
                touches = float(last.low) <= zhigh and float(last.high) >= zlow
                directional_close = float(last.close) > float(last.open) if side == "BUY" else float(last.close) < float(last.open)
                if not touches or not directional_close:
                    continue
                candidates.append({"side": side, "sweep_i": sweep_i, "structure_i": structure_i,
                                   "fvg_i": fvg_i, "zone": zone, "fvg_state": state})
    if not candidates:
        return None, "Немає повного SMC-набору: sweep ліквідності + BOS/CHoCH + дотик до незаповненого/часткового FVG"

    setup = max(candidates, key=lambda x: x["fvg_i"])
    side = setup["side"]
    last = df.iloc[-1]
    entry = float(last.close)
    recent = df.iloc[-6:]
    candle_range = max(float(last.high) - float(last.low), abs(entry) * 0.0003)
    if side == "BUY":
        sl = min(float(recent.low.min()), setup["zone"][0]) - candle_range * 0.1
        risk = entry - sl
        if risk <= 0:
            return None, "Не вдалося розрахувати SL для BUY"
        tp1, tp2 = entry + risk, entry + 2 * risk
    else:
        sl = max(float(recent.high.max()), setup["zone"][1]) + candle_range * 0.1
        risk = sl - entry
        if risk <= 0:
            return None, "Не вдалося розрахувати SL для SELL"
        tp1, tp2 = entry - risk, entry - 2 * risk

    # SNR context: prior 20-candle support/resistance zones.
    prior20 = df.iloc[-21:-1]
    support = float(prior20.low.min())
    resistance = float(prior20.high.max())
    tolerance = max(abs(entry) * 0.001, (resistance - support) * 0.05)
    if abs(entry - support) <= tolerance:
        snr = "поблизу підтримки"
    elif abs(entry - resistance) <= tolerance:
        snr = "поблизу опору"
    else:
        snr = "не біля найближчого краю діапазону"

    dt = df.iloc[-1][df.columns[0]]
    candle_time = pd.Timestamp(dt).isoformat()
    return {
        "side": side, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
        "fvg_state": setup["fvg_state"], "fvg_zone": setup["zone"],
        "sweep": "зняття ліквідності нижче мінімумів" if side == "BUY" else "зняття ліквідності вище максимумів",
        "structure": "BOS/CHoCH bullish" if side == "BUY" else "BOS/CHoCH bearish",
        "snr": snr, "h1_time": candle_time,
    }, "Signal"


def fmt(symbol, value):
    if symbol == "BTCUSD":
        return f"{value:,.2f}"
    if symbol == "XAGUSD":
        return f"{value:,.3f}"
    if "JPY" in symbol:
        return f"{value:.3f}"
    return f"{value:.5f}" if abs(value) < 10 else f"{value:.4f}"


def make_message(symbol, sig, context):
    icon = "🟢" if sig["side"] == "BUY" else "🔴"
    low, high = sig["fvg_zone"]
    return (
        f"{icon} <b>SMART MONEY — {symbol}</b>\n"
        f"Напрямок: <b>{sig['side']}</b>\n"
        f"Entry: {fmt(symbol, sig['entry'])}\n"
        f"SL: {fmt(symbol, sig['sl'])}\n"
        f"TP1: {fmt(symbol, sig['tp1'])}\n"
        f"TP2: {fmt(symbol, sig['tp2'])}\n\n"
        f"Підтвердження SMC:\n"
        f"• {sig['sweep']}\n• {sig['structure']}\n"
        f"• FVG: {sig['fvg_state']} ({fmt(symbol, low)}–{fmt(symbol, high)})\n"
        f"• SNR: {sig['snr']}\n"
        f"Контекст W1/D1/H4: {context}\n"
        f"Закрита H1-свічка (UTC): {sig['h1_time']}\n\n"
        "⚠️ Інформаційний сигнал, не гарантія прибутку. Перевір котирування, спред і ризик. Yahoo Finance може відрізнятися від цін CFD брокера."
    )


async def scan(app, report_chat_id=None):
    sent = 0
    data_errors = 0
    diagnostics = []
    for symbol in SYMBOLS:
        try:
            h1, ticker = download_symbol(symbol, "60d", "1h")
            d1, _ = download_symbol(symbol, "2y", "1d")
            w1, _ = download_symbol(symbol, "5y", "1wk")
            if h1.empty or d1.empty or w1.empty:
                data_errors += 1
                diagnostics.append(f"{symbol}: Yahoo Finance не повернув достатньо даних (тикер {ticker})")
                log.warning("Missing data for %s ticker=%s", symbol, ticker)
                continue
            # Only use fully closed candles; discard current H1/H4/D1/W1 bars.
            h1c, d1c, w1c = h1.iloc[:-1], d1.iloc[:-1], w1.iloc[:-1]
            h4 = h1.resample("4h", label="left", closed="left").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
            h4c = h4.iloc[:-1]
            setup, reason = detect_smc(h1c)
            if not setup:
                diagnostics.append(f"{symbol}: {reason}")
                continue
            biases = {"H4": trend_bias(h4c), "D1": trend_bias(d1c), "W1": trend_bias(w1c)}
            expected = setup["side"]
            conflicts = [tf for tf, bias in biases.items() if bias not in ("NONE", expected)]
            context = ", ".join(f"{tf}={bias}" for tf, bias in biases.items())
            key = f"{symbol}|SMC|{setup['side']}|{setup['h1_time']}"
            if key in ALERTED:
                diagnostics.append(f"{symbol}: SMC-сигнал для цієї H1-свічки вже надсилався")
                continue
            message = make_message(symbol, setup, context)
            if conflicts:
                message += "\n⚠️ Конфлікт зі старшим таймфреймом: " + ", ".join(conflicts)
            if not SUBSCRIBERS:
                diagnostics.append(f"{symbol}: сигнал є, але немає підписників; надішли /start")
                continue
            delivered = False
            for chat_id in list(SUBSCRIBERS):
                try:
                    await app.bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
                    delivered = True
                    sent += 1
                except Exception:
                    log.exception("Telegram delivery failed for %s to chat %s", symbol, chat_id)
            if delivered:
                ALERTED.add(key)
                save_json(ALERTS_FILE, list(ALERTED)[-2000:])
                diagnostics.append(f"{symbol}: сигнал Smart Money надіслано")
        except Exception as exc:
            data_errors += 1
            log.exception("Scan failed for %s", symbol)
            diagnostics.append(f"{symbol}: помилка {type(exc).__name__}: {str(exc)[:140]}")
    if report_chat_id is not None:
        text = f"Сканування завершено. Сигнальних повідомлень: {sent}. Помилок даних/обробки: {data_errors}.\n\nДіагностика:\n"
        for line in diagnostics:
            piece = "• " + line + "\n"
            if len(text) + len(piece) > 3800:
                await app.bot.send_message(chat_id=report_chat_id, text=text)
                text = "Діагностика (продовження):\n"
            text += piece
        await app.bot.send_message(chat_id=report_chat_id, text=text)


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan(context.application)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    save_json(SUBSCRIBERS_FILE, sorted(SUBSCRIBERS))
    await update.effective_message.reply_text(
        f"Dmytro Gold Signals активний. Сканування кожні {INTERVAL} хв.\n"
        "Єдина стратегія: Smart Money (ліквідність + BOS/CHoCH + FVG + SNR), контекст H4/D1/W1.\n"
        "Команди: /status /symbols /scan /help. Автоматичних угод немає."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        f"Бот працює. Стратегія: лише Smart Money. Інструментів: {len(SYMBOLS)}. "
        f"Інтервал сканування: {INTERVAL} хв. Підписників: {len(SUBSCRIBERS)}.\n"
        "Входи перевіряються на закритих H1-свічках; H4/D1/W1 використовуються як контекст."
    )


async def symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Інструменти:\n" + "\n".join("• " + s for s in SYMBOLS))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("/start — підписатися\n/status — стан\n/symbols — інструменти\n/scan — сканувати зараз з діагностикою\n/help — команди")


async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Сканую 9 інструментів лише за Smart Money: ліквідність + BOS/CHoCH + FVG + SNR. Це може тривати до хвилини.")
    await scan(context.application, update.effective_chat.id)


async def post_init(app: Application):
    load_state()
    if app.job_queue is None:
        raise RuntimeError("У requirements.txt потрібен python-telegram-bot[job-queue]")
    app.job_queue.run_repeating(scheduled_scan, interval=INTERVAL * 60, first=15, name="market-scan")
    log.info("Bot started. Subscribers=%d, symbols=%d", len(SUBSCRIBERS), len(SYMBOLS))


def main():
    if not TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable")
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("symbols", symbols))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
