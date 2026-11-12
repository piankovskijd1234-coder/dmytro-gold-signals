import os
import json
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import pandas as pd
import yfinance as yf

TOKEN = "".join(os.getenv("TELEGRAM_BOT_TOKEN", "").split())
INTERVAL = max(5, int(os.getenv("SCAN_INTERVAL_MINUTES", "15")))
MIN_RR = max(2.0, float(os.getenv("MIN_RISK_REWARD", "2.0")))

SUBSCRIBERS_FILE = "subscribers.json"
ALERTS_FILE = "alerts.json"

# Yahoo Finance reference tickers; symbols/prices may differ from broker CFDs.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmytro-gold-signals")
SUBSCRIBERS = set()
ALERTED = set()


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
    global SUBSCRIBERS, ALERTED
    SUBSCRIBERS = {int(x) for x in load_json(SUBSCRIBERS_FILE, [])}
    ALERTED = set(load_json(ALERTS_FILE, []))


def download(ticker, period, interval):
    try:
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, threads=False, auto_adjust=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        needed = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in needed):
            return pd.DataFrame()
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df = df[needed + ["volume"]].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=needed)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()
    except Exception:
        log.exception("Download failed: %s %s", ticker, interval)
        return pd.DataFrame()


def resample_ohlcv(df, rule):
    if df.empty:
        return pd.DataFrame()
    return df.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"])


def ema_bias(df):
    if df.empty or len(df) < 55:
        return "NONE"
    c = df["close"].astype(float)
    e20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
    e50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    p = float(c.iloc[-1])
    if p > e20 > e50:
        return "BUY"
    if p < e20 < e50:
        return "SELL"
    return "NONE"


def swing_levels(df, lookback=20):
    """Simple confirmed range levels; approximation of strong high/low."""
    if df.empty or len(df) < lookback + 3:
        return None, None
    window = df.iloc[-(lookback + 1):-1]
    return float(window["high"].max()), float(window["low"].min())


def detect_htf_poi(df, bias, lookback=30):
    """
    Heuristic POI: most recent opposing candle before an impulse.
    This is a mechanical approximation, not discretionary institutional order flow.
    """
    if df.empty or len(df) < lookback + 5 or bias not in ("BUY", "SELL"):
        return None
    recent = df.iloc[-(lookback + 1):-1]
    atr = (
        pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1).rolling(14).mean()
    )
    for i in range(len(recent) - 2, -1, -1):
        candle = recent.iloc[i]
        nxt = recent.iloc[i + 1]
        impulse = abs(float(nxt["close"]) - float(nxt["open"]))
        a = float(atr.loc[nxt.name]) if nxt.name in atr.index and pd.notna(atr.loc[nxt.name]) else 0.0
        if a <= 0 or impulse < 1.2 * a:
            continue
        if bias == "BUY" and float(candle["close"]) < float(candle["open"]):
            return (float(candle["low"]), float(candle["high"]))
        if bias == "SELL" and float(candle["close"]) > float(candle["open"]):
            return (float(candle["low"]), float(candle["high"]))
    return None


def poi_touched(df, poi):
    if df.empty or poi is None:
        return False
    lo, hi = poi
    last = df.iloc[-1]
    return float(last["low"]) <= hi and float(last["high"]) >= lo


def order_flow_proxy(df, side):
    """OHLCV proxy only; not true exchange bid/ask delta."""
    if df.empty:
        return False, "даних немає"
    last = df.iloc[-1]
    rng = max(float(last["high"]) - float(last["low"]), 1e-12)
    close_pos = (float(last["close"]) - float(last["low"])) / rng
    if side == "BUY":
        candle_ok = float(last["close"]) > float(last["open"]) and close_pos >= 0.65
    else:
        candle_ok = float(last["close"]) < float(last["open"]) and close_pos <= 0.35
    vol = float(last.get("volume", 0.0))
    avg = float(df.iloc[-21:-1]["volume"].mean()) if len(df) > 21 else 0.0
    if vol > 0 and avg > 0:
        vol_ok = vol >= avg
        detail = f"обсяг {vol:.0f} / середній {avg:.0f}"
    else:
        body = abs(float(last["close"]) - float(last["open"])) / rng
        vol_ok = body >= 0.60
        detail = f"тіло свічки {body:.0%}; обсяг недоступний"
    return candle_ok and vol_ok, detail


def intraday_entry(h1, m15, direction, poi):
    """H1 BOS + retest, then M15 candle/volume confirmation."""
    if h1.empty or m15.empty or len(h1) < 30 or len(m15) < 30:
        return None, "Недостатньо історії H1/15M"

    # Require latest completed H1 candle to interact with HTF POI.
    if poi is None or not poi_touched(h1, poi):
        return None, "Ціна не тестує HTF POI"

    h = h1.copy()
    last_h1 = h.iloc[-1]
    prior = h.iloc[-21:-1]
    range_high = float(prior["high"].max())
    range_low = float(prior["low"].min())

    # A breakout and retest are required, not just a moving-average cross.
    if direction == "BUY":
        bos_seen = (h.iloc[-6:-1]["close"] > range_high).any()
        retest = float(last_h1["low"]) <= range_high and float(last_h1["close"]) > range_high
        if not (bos_seen and retest):
            return None, "Немає H1 BOS + Retest"
        mside_ok = float(m15.iloc[-1]["close"]) > float(m15.iloc[-1]["open"])
        swing_sl = float(h.iloc[-11:-1]["low"].min())
        sl = min(swing_sl, poi[0])
    else:
        bos_seen = (h.iloc[-6:-1]["close"] < range_low).any()
        retest = float(last_h1["high"]) >= range_low and float(last_h1["close"]) < range_low
        if not (bos_seen and retest):
            return None, "Немає H1 BOS + Retest"
        mside_ok = float(m15.iloc[-1]["close"]) < float(m15.iloc[-1]["open"])
        swing_sl = float(h.iloc[-11:-1]["high"].max())
        sl = max(swing_sl, poi[1])

    flow_ok, flow_text = order_flow_proxy(m15, direction)
    if not mside_ok or not flow_ok:
        return None, "15M не підтвердив напрямок / Order Flow"

    entry = float(m15.iloc[-1]["close"])
    risk = entry - sl if direction == "BUY" else sl - entry
    if risk <= 0:
        return None, "Некоректний SL відносно Entry"
    tp = entry + MIN_RR * risk if direction == "BUY" else entry - MIN_RR * risk
    return {
        "side": direction, "entry": entry, "sl": sl, "tp": tp,
        "flow": flow_text,
        "time": m15.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
    }, "Signal"


def fmt_price(symbol, value):
    if symbol in ("BTCUSD", "GER40"):
        return f"{value:,.2f}"
    if symbol == "XAGUSD":
        return f"{value:.3f}"
    if "JPY" in symbol:
        return f"{value:.3f}"
    return f"{value:.5f}" if abs(value) < 10 else f"{value:.4f}"


def format_signal(symbol, sig, weekly_bias, daily_bias, daily_poi, h4_poi):
    arrow = "🟢" if sig["side"] == "BUY" else "🔴"
    return (
        f"{arrow} <b>{sig['side']} — {symbol}</b>\\n"
        "Модель: TDA Sync Intraday\\n"
        f"Weekly Bias: {weekly_bias} | Daily Bias: {daily_bias}\\n"
        f"Daily POI: {daily_poi[0]:.5f}–{daily_poi[1]:.5f}\\n"
        f"H4 POI: {h4_poi[0]:.5f}–{h4_poi[1]:.5f}\\n"
        "Підтвердження: H1 BOS + Retest → 15M Confirm\\n\\n"
        f"Entry: {fmt_price(symbol, sig['entry'])}\\n"
        f"SL: {fmt_price(symbol, sig['sl'])} (за Strong High/Low)\\n"
        f"TP: {fmt_price(symbol, sig['tp'])}\\n"
        f"RR: 1:{MIN_RR:.1f}\\n"
        f"Order Flow proxy: {sig['flow']}\\n"
        f"Закрита 15M свічка: {sig['time']}\\n\\n"
        "⚠️ POI/Strong High-Low визначаються алгоритмічно як наближення. "
        "Order Flow — OHLCV-оцінка, не справжня Bid/Ask Delta. "
        "Yahoo Finance може відрізнятися від CFD-брокера. Перевір сигнал у MT5."
    )


async def scan(app, report_chat=None):
    sent, errors = 0, 0
    notes = []
    for symbol, ticker in SYMBOLS.items():
        try:
            h1raw = download(ticker, "60d", "1h")
            m15raw = download(ticker, "60d", "15m")
            d1raw = download(ticker, "2y", "1d")
            w1raw = download(ticker, "5y", "1wk")
            if any(x.empty for x in (h1raw, m15raw, d1raw, w1raw)):
                errors += 1
                notes.append(f"{symbol}: немає даних для одного з TF")
                continue

            # Use only completed candles.
            h1 = h1raw.iloc[:-1]
            m15 = m15raw.iloc[:-1]
            d1 = d1raw.iloc[:-1]
            w1 = w1raw.iloc[:-1]
            h4 = resample_ohlcv(h1raw, "4h").iloc[:-1]

            wb = ema_bias(w1)
            db = ema_bias(d1)
            if wb not in ("BUY", "SELL") or db != wb:
                notes.append(f"{symbol}: Weekly/Daily Bias не синхронізовані")
                continue

            daily_poi = detect_htf_poi(d1, db)
            h4_poi = detect_htf_poi(h4, db)
            if daily_poi is None or h4_poi is None:
                notes.append(f"{symbol}: не знайдено алгоритмічний Daily/H4 POI")
                continue

            # Price must be within or testing the Daily/H4 area.
            if not (poi_touched(h1, daily_poi) and poi_touched(h1, h4_poi)):
                notes.append(f"{symbol}: ціна не в HTF POI")
                continue

            sig, reason = intraday_entry(h1, m15, db, h4_poi)
            if sig is None:
                notes.append(f"{symbol}: {reason}")
                continue

            key = f"{symbol}|{sig['side']}|{sig['time']}"
            if key in ALERTED:
                notes.append(f"{symbol}: сигнал цієї свічки вже надсилався")
                continue
            if not SUBSCRIBERS:
                notes.append(f"{symbol}: немає підписників — натисни /start")
                continue

            msg = format_signal(symbol, sig, wb, db, daily_poi, h4_poi)
            delivered = False
            for chat_id in list(SUBSCRIBERS):
                try:
                    await app.bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")
                    delivered = True
                    sent += 1
                except Exception:
                    log.exception("Telegram send failed for %s", symbol)
            if delivered:
                ALERTED.add(key)
                save_json(ALERTS_FILE, sorted(ALERTED)[-3000:])
                notes.append(f"{symbol}: сигнал надіслано")
        except Exception as exc:
            errors += 1
            log.exception("Scan failed for %s", symbol)
            notes.append(f"{symbol}: {type(exc).__name__}: {str(exc)[:100]}")

    if report_chat is not None:
        report = (
            f"Сканування завершено. Інструментів: {len(SYMBOLS)}. "
            f"Надісланих повідомлень: {sent}. Помилок: {errors}.\\n\\n"
            + "\\n".join("• " + n for n in notes)
        )
        for i in range(0, len(report), 3900):
            await app.bot.send_message(chat_id=report_chat, text=report[i:i + 3900])


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    await scan(context.application)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    save_json(SUBSCRIBERS_FILE, sorted(SUBSCRIBERS))
    await update.effective_message.reply_text(
        "Dmytro Gold Signals активний.\\n"
        f"Інструментів: {len(SYMBOLS)}; сканування кожні {INTERVAL} хв.\\n"
        "TDA Sync Intraday: Weekly/Daily Bias → Daily/H4 POI → H1 BOS+Retest → 15M Confirm.\\n"
        "Один TP; мінімальний RR 1:2. Бот не відкриває угоди."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        f"Бот активний. Інструментів: {len(SYMBOLS)}.\\n"
        "Модель: TDA Sync Intraday. Weekly/Daily Bias; Daily/H4 POI; "
        "H1 BOS+Retest; 15M Confirm. Один TP; мінімальний RR 1:2.\\n"
        f"Сканування: {INTERVAL} хв. Підписників: {len(SUBSCRIBERS)}.\\n"
        "POI/Order Flow — алгоритмічні наближення, не біржові дані."
    )


async def symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Інструменти:\\n" + "\\n".join("• " + s for s in SYMBOLS)
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "/start — підписатися\\n/status — стан\\n"
        "/symbols — список інструментів\\n/scan — сканувати зараз\\n/help — команди"
    )


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        f"Сканую {len(SYMBOLS)} інструментів за TDA Sync. Це може тривати кілька хвилин."
    )
    await scan(context.application, update.effective_chat.id)


async def post_init(app: Application):
    load_state()
    if app.job_queue is None:
        raise RuntimeError("Встанови python-telegram-bot[job-queue] у requirements.txt")
    app.job_queue.run_repeating(
        scheduled_scan, interval=INTERVAL * 60, first=20, name="tda-market-scan"
    )
    log.info("TDA bot started: symbols=%d subscribers=%d", len(SYMBOLS), len(SUBSCRIBERS))


def main():
    if not TOKEN:
        raise RuntimeError("Не задано TELEGRAM_BOT_TOKEN у Railway Variables")
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("symbols", symbols_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("help", help_command))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
