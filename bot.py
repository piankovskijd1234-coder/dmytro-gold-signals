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
SYMBOLS = {
    "BTC/USD": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "EUR/JPY": "EURJPY=X",
    "AUD/USD": "AUDUSD=X",
    "XAG/USD": "SI=F",  # Silver futures proxy; may differ from broker spot CFD
    "EUR/AUD": "EURAUD=X",
    "USD/CHF": "CHF=X",
    "XAU/USD": "GC=F",  # Gold futures proxy; may differ from broker spot CFD
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmytro-gold-signals")
SUBSCRIBERS = set()
# Prevent duplicate alert for the same symbol, side and completed H1 candle.
ALERTED = set()

def load_alerts():
    global ALERTED
    try:
        with open(ALERTS_FILE, "r", encoding="utf-8") as f:
            ALERTED = set(json.load(f))
    except (OSError, ValueError, TypeError):
        ALERTED = set()

def save_alerts():
    try:
        # Bound file size: keep the latest 2000 keys.
        with open(ALERTS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(ALERTED)[-2000:], f)
    except OSError:
        log.exception("Cannot save alert history")


def load_subscribers():
    global SUBSCRIBERS
    try:
        with open(SUBSCRIBERS_FILE, "r", encoding="utf-8") as f:
            SUBSCRIBERS = {int(x) for x in json.load(f)}
    except (OSError, ValueError, TypeError):
        SUBSCRIBERS = set()


def save_subscribers():
    try:
        with open(SUBSCRIBERS_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(SUBSCRIBERS), f)
    except OSError:
        log.exception("Cannot save subscribers")


def download(ticker, period, interval):
    try:
        frame = yf.download(
            ticker, period=period, interval=interval,
            progress=False, threads=False, auto_adjust=False,
        )
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


def ema_bias(frame):
    """Trend context from already-closed candles only."""
    if frame.empty or len(frame) < 50:
        return "NEUTRAL"
    close = frame["close"]
    e20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    e50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    price = float(close.iloc[-1])
    if price > e20 > e50:
        return "BULL"
    if price < e20 < e50:
        return "BEAR"
    return "NEUTRAL"


def detect_setups(frame, label, lookback=30):
    """Return SNR/FVG/RB reactions on completed candles only."""
    result = {"BUY": [], "SELL": []}
    if frame.empty or len(frame) < 25:
        return result
    closed = frame.iloc[:-1]
    if len(closed) < 22:
        return result
    cur, prev = closed.iloc[-1], closed.iloc[-2]
    op, cl, hi, lo = map(float, (cur.open, cur.close, cur.high, cur.low))
    pop, pcl, phi, plo = map(float, (prev.open, prev.close, prev.high, prev.low))

    # Support/resistance reaction: current completed candle reacts at a prior range edge.
    prior = closed.iloc[-(lookback + 1):-1]
    if len(prior) >= 10:
        support = float(prior.low.min())
        resistance = float(prior.high.max())
        tolerance = max(abs(cl) * 0.001, (resistance - support) * 0.04)
        if lo <= support + tolerance and cl > op and cl > support:
            result["BUY"].append(f"{label} SNR: відбій від підтримки")
        if hi >= resistance - tolerance and cl < op and cl < resistance:
            result["SELL"].append(f"{label} SNR: відбій від опору")

    # Fair Value Gap (three-candle imbalance) with a reaction on the latest closed candle.
    for i in range(len(closed) - 1, max(2, len(closed) - 35), -1):
        left, right = closed.iloc[i - 2], closed.iloc[i]
        left_high, left_low = float(left.high), float(left.low)
        right_high, right_low = float(right.high), float(right.low)
        if left_high < right_low and lo <= right_low and cl > op and cl >= left_high:
            result["BUY"].append(f"{label} FVG: реакція в bullish FVG")
            break
        if left_low > right_high and hi >= right_high and cl < op and cl <= left_low:
            result["SELL"].append(f"{label} FVG: реакція в bearish FVG")
            break

    # RB proxy: reversal candle closes beyond the previous candle's extreme.
    if pcl < pop and cl > op and cl > phi:
        result["BUY"].append(f"{label} RB: бичаче поглинання/злам локального максимуму")
    if pcl > pop and cl < op and cl < plo:
        result["SELL"].append(f"{label} RB: ведмеже поглинання/злам локального мінімуму")
    return result


def analyze(h1, h4, d1, w1):
    """W1/D1 context; H4/H1 SNR-RB-FVG; closed H1 candle confirms. No M15/Stochastic."""
    if len(h1) < 40 or len(h4) < 30 or len(d1) < 30 or len(w1) < 20:
        return None, "Недостатньо історичних даних на одному з таймфреймів"

    # Yahoo may include the currently forming bar. Use completed bars for decisions.
    h1c, h4c, d1c, w1c = h1.iloc[:-1], h4.iloc[:-1], d1.iloc[:-1], w1.iloc[:-1]
    if min(len(h1c), len(h4c), len(d1c), len(w1c)) < 20:
        return None, "Недостатньо закритих свічок для аналізу"

    last, prev = h1c.iloc[-1], h1c.iloc[-2]
    op, cl = float(last.open), float(last.close)
    hi, lo = float(last.high), float(last.low)
    prev_hi, prev_lo, prev_close = float(prev.high), float(prev.low), float(prev.close)
    candle_range = max(hi - lo, abs(cl) * 0.0003)
    body = abs(cl - op)
    # H1 confirmation: close direction plus either momentum through previous close
    # or a meaningful candle body. This avoids requiring every indicator at once.
    bullish = cl > op and (cl > prev_close or body >= candle_range * 0.45)
    bearish = cl < op and (cl < prev_close or body >= candle_range * 0.45)
    if not bullish and not bearish:
        return None, "H1: остання закрита свічка не підтвердила напрямок"
    side = "BUY" if bullish else "SELL"

    # Look for a reaction/zone on H1 or H4 in the last few closed candles,
    # not only on the exact final candle. This reduces missed setups while keeping confirmation on H1.
    def recent_setup_reasons(closed, label, bars=4):
        out = []
        if len(closed) < 25:
            return out
        recent = closed.tail(bars)
        prior = closed.iloc[:-bars]
        if len(prior) < 10:
            return out
        support = float(prior.tail(30).low.min())
        resistance = float(prior.tail(30).high.max())
        width = max(resistance - support, abs(float(closed.close.iloc[-1])) * 0.0005)
        tol = max(abs(float(closed.close.iloc[-1])) * 0.0008, width * 0.06)
        if side == "BUY" and (float(recent.low.min()) <= support + tol) and cl > op:
            out.append(f"{label} SNR: тест/відбій підтримки")
        if side == "SELL" and (float(recent.high.max()) >= resistance - tol) and cl < op:
            out.append(f"{label} SNR: тест/відбій опору")
        # Three-candle FVG zones; check recent candles for a revisit.
        scan = closed.tail(35)
        for i in range(max(2, len(scan) - 12), len(scan)):
            left, mid, right = scan.iloc[i-2], scan.iloc[i-1], scan.iloc[i]
            if side == "BUY":
                gap_low, gap_high = float(left.high), float(right.low)
                if gap_low < gap_high and float(recent.low.min()) <= gap_high and float(recent.high.max()) >= gap_low:
                    out.append(f"{label} FVG: ціна тестувала bullish FVG")
                    break
            else:
                gap_low, gap_high = float(right.high), float(left.low)
                if gap_low < gap_high and float(recent.high.max()) >= gap_low and float(recent.low.min()) <= gap_high:
                    out.append(f"{label} FVG: ціна тестувала bearish FVG")
                    break
        # RB proxy: recent strong reversal candle breaks prior candle extreme.
        for i in range(max(1, len(recent)-3), len(recent)):
            cur, before = recent.iloc[i], closed.iloc[-len(recent)+i-1]
            if side == "BUY" and float(cur.close) > float(cur.open) and float(cur.close) > float(before.high):
                out.append(f"{label} RB: бичачий злам локального максимуму")
                break
            if side == "SELL" and float(cur.close) < float(cur.open) and float(cur.close) < float(before.low):
                out.append(f"{label} RB: ведмежий злам локального мінімуму")
                break
        return out

    reasons = recent_setup_reasons(h1c, "H1") + recent_setup_reasons(h4c, "H4")
    if not reasons:
        return None, f"H1 підтвердив {side}, але за останні закриті свічки немає реакції SNR/RB/FVG на H1/H4"

    w_bias, d_bias, h4_bias = ema_bias(w1c), ema_bias(d1c), ema_bias(h4c)
    expected = "BULL" if side == "BUY" else "BEAR"
    # Strict daily direction filter: no signal against D1, and no signal if D1 is neutral.
    if d_bias != expected:
        return None, f"D1-фільтр: сигнал {side} відхилено, бо D1={d_bias}; потрібен D1={expected}"
    warnings = []
    for tf, bias in (("W1", w_bias), ("H4", h4_bias)):
        if bias not in ("NEUTRAL", expected):
            warnings.append(f"УВАГА: {tf} проти сигналу ({bias})")

    entry = cl
    recent_h1 = h1c.tail(5)
    if side == "BUY":
        sl = float(recent_h1.low.min()) - candle_range * 0.15
        risk = entry - sl
        if risk <= 0:
            return None, "Не вдалося розрахувати стоп-лос"
        targets = [entry + risk, entry + 2 * risk, entry + 3 * risk]
    else:
        sl = float(recent_h1.high.max()) + candle_range * 0.15
        risk = sl - entry
        if risk <= 0:
            return None, "Не вдалося розрахувати стоп-лос"
        targets = [entry - risk, entry - 2 * risk, entry - 3 * risk]

    reasons.insert(0, "H1: напрямок підтверджено закритою свічкою")
    reasons.extend(warnings)
    return {
        "side": side, "entry": entry, "sl": sl, "tp": targets,
        "reasons": reasons,
        "context": f"W1={w_bias}, D1={d_bias}, H4={h4_bias}",
        "h1": str(h1c.index[-1]),
    }, "Signal"

def fmt(symbol, value):
    if symbol == "BTC/USD":
        return f"{value:,.2f}"
    if symbol in ("XAU/USD", "XAG/USD"):
        return f"{value:,.3f}"
    if "JPY" in symbol:
        return f"{value:.3f}"
    return f"{value:.5f}" if abs(value) < 10 else f"{value:.4f}"


def make_message(symbol, signal):
    icon = "🟢" if signal["side"] == "BUY" else "🔴"
    reasons = "\n".join("• " + r for r in signal["reasons"])
    return (
        f"{icon} <b>{signal['side']} — {symbol}</b>\n"
        f"Entry (H1): {fmt(symbol, signal['entry'])}\n"
        f"SL: {fmt(symbol, signal['sl'])}\n"
        f"TP1: {fmt(symbol, signal['tp'][0])}\n"
        f"TP2: {fmt(symbol, signal['tp'][1])}\n"
        f"TP3: {fmt(symbol, signal['tp'][2])}\n\n"
        f"Причини:\n{reasons}\n"
        f"Контекст: {signal['context']}\n"
        f"Підтвердження: закрита свічка H1\n"
        f"Свічка H1: {signal['h1']}\n\n"
        "⚠️ Не гарантія прибутку. Перевір котирування, спред і ризик. XAU/XAG використовують ф’ючерсні проксі Yahoo (GC=F/SI=F), ціна може відрізнятися від CFD брокера."
    )


async def scan(app, report_chat_id=None):
    sent = 0
    diagnostics = []
    for symbol, ticker in SYMBOLS.items():
        try:
            h1 = download(ticker, "60d", "1h")
            d1 = download(ticker, "2y", "1d")
            w1 = download(ticker, "5y", "1wk")
            if h1.empty or d1.empty or w1.empty:
                diagnostics.append(f"{symbol}: Yahoo Finance не повернув достатньо даних")
                continue
            # Build H4 from H1 data; label buckets in UTC.
            h4 = h1.resample("4h", label="left", closed="left").agg({
                "open": "first", "high": "max", "low": "min", "close": "last"
            }).dropna()
            signal, reason = analyze(h1, h4, d1, w1)
            if not signal:
                diagnostics.append(f"{symbol}: {reason}")
                continue
            key = f"{symbol}|{signal['side']}|{signal['h1']}"
            if key in ALERTED:
                diagnostics.append(f"{symbol}: цей сигнал уже надсилався для закритої H1-свічки")
                continue
            delivered = False
            for chat_id in list(SUBSCRIBERS):
                try:
                    await app.bot.send_message(chat_id=chat_id, text=make_message(symbol, signal), parse_mode="HTML")
                    delivered = True
                    sent += 1
                except Exception:
                    log.exception("Message send failed for chat %s", chat_id)
            if delivered:
                ALERTED.add(key)
                save_alerts()
            else:
                diagnostics.append(f"{symbol}: сигнал знайдено, але не доставлено; перевір /start і журнали Railway")
        except Exception:
            log.exception("Scan failed for %s", symbol)
            diagnostics.append(f"{symbol}: помилка; перевір Railway logs")
    if report_chat_id is not None:
        # Telegram messages have a 4096 character limit; split diagnostic output safely.
        header = f"Сканування завершено. Надіслано повідомлень із сигналами: {sent}\n\nДіагностика:\n"
        chunk = header
        for line in diagnostics:
            addition = "• " + line + "\n"
            if len(chunk) + len(addition) > 3800:
                await app.bot.send_message(chat_id=report_chat_id, text=chunk)
                chunk = "Діагностика (продовження):\n"
            chunk += addition
        await app.bot.send_message(chat_id=report_chat_id, text=chunk)


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan(context.application)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    save_subscribers()
    await update.effective_message.reply_text(
        f"Dmytro Gold Signals активний. Сканування кожні {INTERVAL} хв.\n"
        "Логіка: обов’язковий напрямок D1 → H4/H1 SNR, RB або FVG → підтвердження закритою H1-свічкою.\n"
        "M15 і Stochastic повністю прибрані. Автоматичні угоди не відкриваються.\n"
        "Команди: /status /symbols /scan /help"
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        f"Бот працює. Інтервал: {INTERVAL} хв. Підписників: {len(SUBSCRIBERS)}.\n"
        "Обов’язковий напрямок D1 + підтвердження закритою H1-свічкою; M15 і Stochastic вимкнені; автоматичної торгівлі немає. Дані: Yahoo Finance; золото/срібло — ф’ючерсні проксі."
    )


async def symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Інструменти:\n" + "\n".join("• " + s for s in SYMBOLS))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "/start — підписатися\n/status — стан бота\n/symbols — список інструментів\n"
        "/scan — сканувати зараз із діагностикою\n/help — команди"
    )


async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Сканую 9 інструментів. Напрямок D1 обов’язковий; підтвердження — закрита H1-свічка. M15 і Stochastic вимкнені.")
    await scan(context.application, update.effective_chat.id)


async def post_init(app: Application):
    load_subscribers()
    load_alerts()
    if app.job_queue is None:
        raise RuntimeError("У requirements.txt має бути python-telegram-bot[job-queue]")
    app.job_queue.run_repeating(scheduled_scan, interval=INTERVAL * 60, first=15, name="market-scan")
    log.info("Bot started. Subscribers=%d", len(SUBSCRIBERS))


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
