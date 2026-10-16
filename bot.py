import os
import json
import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Dmytro Gold Signals
# Context: W1 + D1, direction: H4, entry confirmation: H1
# Entry methods: SNR, RB-like reversal, FVG, Stochastic OB/OS, breakout + retest.
# Signals only. This bot does NOT place trades.

TOKEN = "".join(os.getenv("TELEGRAM_BOT_TOKEN", "").split())
SCAN_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
SUBSCRIBERS_FILE = "subscribers.json"

SYMBOLS = {
    "BTC/USD": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "EUR/JPY": "EURJPY=X",
    "AUD/USD": "AUDUSD=X",
    "XAG/USD": "SI=F",
    "EUR/AUD": "EURAUD=X",
    "USD/CHF": "CHF=X",
    "XAU/USD": "GC=F",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("dmytro-gold-signals")
SUBSCRIBERS = set()
LAST_ALERTED = set()


def load_subscribers():
    global SUBSCRIBERS
    try:
        with open(SUBSCRIBERS_FILE, "r", encoding="utf-8") as file:
            SUBSCRIBERS = {int(item) for item in json.load(file)}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        SUBSCRIBERS = set()


def save_subscribers():
    try:
        with open(SUBSCRIBERS_FILE, "w", encoding="utf-8") as file:
            json.dump(sorted(SUBSCRIBERS), file)
    except OSError:
        log.exception("Failed saving subscribers")


def download(ticker, period, interval):
    try:
        frame = yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=False, progress=False, threads=False
        )
        if frame is None or frame.empty:
            return pd.DataFrame()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(col).lower().replace(" ", "_") for col in frame.columns]
        required = ["open", "high", "low", "close"]
        if not all(col in frame.columns for col in required):
            return pd.DataFrame()
        return frame[required].dropna()
    except Exception:
        log.exception("Download failed: %s / %s / %s", ticker, period, interval)
        return pd.DataFrame()


def make_h4(h1):
    if h1.empty:
        return pd.DataFrame()
    return h1.resample("4h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def trend(frame):
    if len(frame) < 205:
        return "NEUTRAL"
    close = frame["close"]
    e50, e100, e200 = (ema(close, n).iloc[-1] for n in (50, 100, 200))
    price = float(close.iloc[-1])
    if price > e50 > e100 > e200:
        return "BULL"
    if price < e50 < e100 < e200:
        return "BEAR"
    return "NEUTRAL"


def stochastic(frame, k_period=14, smooth_k=3, d_period=3):
    low_min = frame["low"].rolling(k_period).min()
    high_max = frame["high"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, float("nan"))
    raw_k = 100 * (frame["close"] - low_min) / denom
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()
    return k, d


def latest_fvg(frame):
    # Only completed candles; scan recent three-candle patterns.
    data = frame.iloc[:-1]
    for i in range(len(data) - 1, max(1, len(data) - 35), -1):
        left, middle, right = data.iloc[i - 2], data.iloc[i - 1], data.iloc[i]
        if float(left["high"]) < float(right["low"]):
            return ("BULL FVG", float(left["high"]), float(right["low"]))
        if float(left["low"]) > float(right["high"]):
            return ("BEAR FVG", float(right["high"]), float(left["low"]))
    return None


def latest_snr(frame, lookback=30):
    data = frame.iloc[:-1].tail(lookback)
    if len(data) < 10:
        return None, None
    return float(data["low"].min()), float(data["high"].max())


def detect_setup(h1, h4, d1, w1):
    # Require sufficient data to calculate stable 200-period trend filters.
    if min(len(h1), len(h4), len(d1), len(w1)) < 205:
        return None, "Недостатньо історичних котирувань"

    wtrend, dtrend, h4trend = trend(w1), trend(d1), trend(h4)
    if "NEUTRAL" in (wtrend, dtrend, h4trend):
        return None, "Немає чіткого напрямку W1/D1/H4"
    if not (wtrend == dtrend == h4trend):
        return None, "Напрямки W1, D1 і H4 не узгоджені"

    # Evaluate only completed H1 candles.
    data = h1.iloc[:-1]
    if len(data) < 30:
        return None, "Недостатньо свічок H1"
    prev, cur = data.iloc[-2], data.iloc[-1]
    close = float(cur["close"])
    opn, high, low = float(cur["open"]), float(cur["high"]), float(cur["low"])
    prev_close = float(prev["close"])
    prev_open = float(prev["open"])
    prev_high = float(prev["high"])
    prev_low = float(prev["low"])

    k, d = stochastic(data)
    if len(k.dropna()) < 2 or len(d.dropna()) < 2:
        return None, "Stochastic ще не розрахований"
    k_prev, k_now = float(k.iloc[-2]), float(k.iloc[-1])
    d_prev, d_now = float(d.iloc[-2]), float(d.iloc[-1])

    support, resistance = latest_snr(h1)
    fvg = latest_fvg(h1)
    direction = "BUY" if h4trend == "BULL" else "SELL"
    reasons = []

    # Stochastic: actual exit from extreme zone plus K/D crossover.
    stoch_buy = k_prev < 20 and d_prev < 20 and k_now >= 20 and k_now > d_now and k_prev <= d_prev
    stoch_sell = k_prev > 80 and d_prev > 80 and k_now <= 80 and k_now < d_now and k_prev >= d_prev

    # SNR reaction: price tests the recent support/resistance and candle closes away.
    snr_buy = support is not None and low <= support * 1.002 and close > opn and close > support
    snr_sell = resistance is not None and high >= resistance * 0.998 and close < opn and close < resistance

    # FVG reaction: price returns into a recent imbalance and closes in its expected direction.
    fvg_buy = fvg is not None and fvg[0] == "BULL FVG" and low <= fvg[2] and close > opn and close >= fvg[1]
    fvg_sell = fvg is not None and fvg[0] == "BEAR FVG" and high >= fvg[1] and close < opn and close <= fvg[2]

    # RB-like reversal proxy: engulfing/break of prior candle extremum.
    # This is a candle-pattern approximation, not a formally identified institutional block.
    rb_buy = prev_close < prev_open and close > opn and close > prev_high
    rb_sell = prev_close > prev_open and close < opn and close < prev_low

    # Breakout and retest: prior candle breaks a recent level; current candle retests it.
    prior_window = data.iloc[-21:-2]
    breakout_buy = breakout_sell = False
    if len(prior_window) >= 10 and support is not None and resistance is not None:
        old_resistance = float(prior_window["high"].max())
        old_support = float(prior_window["low"].min())
        breakout_buy = prev_close > old_resistance and low <= old_resistance * 1.002 and close > old_resistance and close > opn
        breakout_sell = prev_close < old_support and high >= old_support * 0.998 and close < old_support and close < opn

    if direction == "BUY":
        if snr_buy:
            reasons.append("SNR: підтверджена реакція від підтримки")
        if fvg_buy:
            reasons.append("FVG: реакція в бичачій зоні")
        if rb_buy:
            reasons.append("RB: бичачий розворотний патерн")
        if stoch_buy:
            reasons.append("Stochastic: вихід із перепроданості + перетин %K/%D")
        if breakout_buy:
            reasons.append("Пробій опору + ретест")
        # H1 confirmation must be present; no requirement for breakout in every setup.
        if not reasons:
            return None, "Умови входу H1 не підтверджені"
        sl = min(low, support if support is not None else low) - max(abs(close) * 0.001, (high - low) * 0.5)
        risk = close - sl
        if risk <= 0:
            return None, "Неправильний розрахунок SL"
        targets = [close + risk, close + 2 * risk, close + 3 * risk]
    else:
        if snr_sell:
            reasons.append("SNR: підтверджена реакція від опору")
        if fvg_sell:
            reasons.append("FVG: реакція в ведмежій зоні")
        if rb_sell:
            reasons.append("RB: ведмежий розворотний патерн")
        if stoch_sell:
            reasons.append("Stochastic: вихід із перекупленості + перетин %K/%D")
        if breakout_sell:
            reasons.append("Пробій підтримки + ретест")
        if not reasons:
            return None, "Умови входу H1 не підтверджені"
        sl = max(high, resistance if resistance is not None else high) + max(abs(close) * 0.001, (high - low) * 0.5)
        risk = sl - close
        if risk <= 0:
            return None, "Неправильний розрахунок SL"
        targets = [close - risk, close - 2 * risk, close - 3 * risk]

    return {
        "direction": direction, "entry": close, "sl": sl, "targets": targets,
        "reasons": reasons, "wtrend": wtrend, "dtrend": dtrend, "h4trend": h4trend,
        "k": k_now, "d": d_now, "candle_time": str(data.index[-1]),
    }, "Сигнал знайдено"


def price_fmt(symbol, price):
    if symbol == "BTC/USD":
        return f"{price:,.2f}"
    if symbol in ("XAU/USD", "XAG/USD"):
        return f"{price:,.3f}"
    if "JPY" in symbol:
        return f"{price:.3f}"
    if abs(price) < 10:
        return f"{price:.5f}"
    return f"{price:.4f}"


def format_signal(symbol, setup):
    icon = "🟢" if setup["direction"] == "BUY" else "🔴"
    reasons = "\n".join(f"• {reason}" for reason in setup["reasons"])
    return (
        f"{icon} <b>{setup['direction']} — {symbol}</b>\n"
        f"━━━━━━━━━━━━━━\n"
        f"<b>Entry:</b> {price_fmt(symbol, setup['entry'])}\n"
        f"<b>SL:</b> {price_fmt(symbol, setup['sl'])}\n"
        f"<b>TP1:</b> {price_fmt(symbol, setup['targets'][0])}\n"
        f"<b>TP2:</b> {price_fmt(symbol, setup['targets'][1])}\n"
        f"<b>TP3:</b> {price_fmt(symbol, setup['targets'][2])}\n\n"
        f"<b>Причини входу:</b>\n{reasons}\n\n"
        f"<b>Контекст:</b> W1 {setup['wtrend']} | D1 {setup['dtrend']} | H4 {setup['h4trend']}\n"
        f"<b>Stochastic H1:</b> %K {setup['k']:.1f} / %D {setup['d']:.1f}\n"
        f"<b>Закрита свічка H1:</b> {setup['candle_time']}\n\n"
        "⚠️ Технічний сигнал, не гарантія прибутку. Перевір ризик і спред."
    )


async def scan_markets(app, report_chat_id=None):
    found = 0
    diagnostics = []
    for symbol, ticker in SYMBOLS.items():
        try:
            h1 = download(ticker, "60d", "1h")
            if h1.empty:
                diagnostics.append(f"{symbol}: немає котирувань H1")
                continue
            h4 = make_h4(h1)
            d1 = download(ticker, "2y", "1d")
            w1 = download(ticker, "5y", "1wk")
            if h4.empty or d1.empty or w1.empty:
                diagnostics.append(f"{symbol}: бракує даних одного з таймфреймів")
                continue
            setup, reason = detect_setup(h1, h4, d1, w1)
            if setup is None:
                diagnostics.append(f"{symbol}: {reason}")
                continue

            key = f"{symbol}|{setup['direction']}|{setup['candle_time']}"
            if key in LAST_ALERTED:
                diagnostics.append(f"{symbol}: сигнал цієї свічки вже надсилався")
                continue
            LAST_ALERTED.add(key)
            message = format_signal(symbol, setup)
            for chat_id in list(SUBSCRIBERS):
                try:
                    await app.bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
                    found += 1
                except Exception:
                    log.exception("Could not send to chat %s", chat_id)
        except Exception:
            log.exception("Scan failed for %s", symbol)
            diagnostics.append(f"{symbol}: помилка аналізу; дивись Railway logs")

    if report_chat_id is not None:
        summary = "Сканування завершено.\n"
        summary += f"Сигналів надіслано: {found}\n\n"
        summary += "Причини відсутності сигналів:\n" + "\n".join(f"• {x}" for x in diagnostics[:9])
        await app.bot.send_message(chat_id=report_chat_id, text=summary)


async def scan_job(context: ContextTypes.DEFAULT_TYPE):
    await scan_markets(context.application)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)
    save_subscribers()
    await update.effective_message.reply_text(
        "Dmytro Gold Signals підключений.\n"
        f"Автосканування: кожні {SCAN_MINUTES} хв.\n"
        "W1/D1 контекст → H4 напрямок → H1 підтвердження.\n"
        "Входи: SNR, RB-патерн, FVG, Stochastic OB/OS, пробій + ретест.\n"
        "Команди: /status /symbols /scan /help"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "/start — підписатися на сигнали\n"
        "/status — стан бота\n"
        "/symbols — список інструментів\n"
        "/scan — перевірити ринок зараз і показати діагностику\n"
        "/help — допомога"
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Бот запущений.\n"
        f"Період сканування: {SCAN_MINUTES} хв.\n"
        f"Підписників у пам'яті: {len(SUBSCRIBERS)}.\n"
        f"Інструментів: {len(SYMBOLS)}.\n"
        "Джерело котирувань: Yahoo Finance.\n"
        "Автоматичне відкриття угод: вимкнено."
    )


async def symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Інструменти:\n" + "\n".join(f"• {symbol}" for symbol in SYMBOLS))


async def scan_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Сканую ринок. Це може зайняти кілька хвилин.")
    await scan_markets(context.application, report_chat_id=update.effective_chat.id)


async def post_init(app: Application):
    load_subscribers()
    if app.job_queue is None:
        raise RuntimeError('JobQueue is unavailable. Add "python-telegram-bot[job-queue]" to requirements.txt')
    app.job_queue.run_repeating(
        scan_job, interval=SCAN_MINUTES * 60, first=15, name="market-scan"
    )
    log.info("Bot started; subscribers=%s; scan interval=%s minutes", len(SUBSCRIBERS), SCAN_MINUTES)


def main():
    if not TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable")
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("symbols", symbols))
    app.add_handler(CommandHandler("scan", scan_now))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
