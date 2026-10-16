import os
import json
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import pandas as pd
import yfinance as yf

TOKEN = "".join(os.getenv("TELEGRAM_BOT_TOKEN", "").split())
INTERVAL = max(1, int(os.getenv("SCAN_INTERVAL_MINUTES", "5")))
SUBSCRIBERS_FILE = "subscribers.json"
SYMBOLS = {
    "BTC/USD": "BTC-USD", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
    "EUR/JPY": "EURJPY=X", "AUD/USD": "AUDUSD=X", "XAG/USD": "SI=F",
    "EUR/AUD": "EURAUD=X", "USD/CHF": "CHF=X", "XAU/USD": "GC=F",
}
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmytro-gold-signals")
SUBSCRIBERS = set()
ALERTED = set()


def load_subscribers():
    global SUBSCRIBERS
    try:
        with open(SUBSCRIBERS_FILE, encoding="utf-8") as f:
            SUBSCRIBERS = {int(x) for x in json.load(f)}
    except Exception:
        SUBSCRIBERS = set()


def save_subscribers():
    try:
        with open(SUBSCRIBERS_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(SUBSCRIBERS), f)
    except Exception:
        log.exception("Cannot save subscribers")


def download(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        return df[["open", "high", "low", "close"]].dropna()
    except Exception:
        log.exception("Quote download failed: %s %s", ticker, interval)
        return pd.DataFrame()


def trend(df):
    if len(df) < 205:
        return "NEUTRAL"
    c = df["close"]
    e50, e100, e200 = [float(c.ewm(span=n, adjust=False).mean().iloc[-1]) for n in (50, 100, 200)]
    p = float(c.iloc[-1])
    if p > e50 > e100 > e200:
        return "BULL"
    if p < e50 < e100 < e200:
        return "BEAR"
    return "NEUTRAL"



def snr(df, n=30):
    d = df.iloc[:-1].tail(n)
    if len(d) < 10:
        return None, None
    return float(d.low.min()), float(d.high.max())


def fvg(df):
    d = df.iloc[:-1]
    for i in range(len(d)-1, max(1, len(d)-40), -1):
        left, right = d.iloc[i-2], d.iloc[i]
        if float(left.high) < float(right.low):
            return ("BULL", float(left.high), float(right.low))
        if float(left.low) > float(right.high):
            return ("BEAR", float(right.high), float(left.low))
    return None


def timeframe_setups(frame, label):
    """Detect approximate SNR, FVG and RB-style reactions on a completed candle."""
    out = {"BUY": [], "SELL": []}
    if frame.empty or len(frame) < 25:
        return out
    data = frame.iloc[:-1]  # don't use the candle still forming
    if len(data) < 20:
        return out
    cur, prev = data.iloc[-1], data.iloc[-2]
    op, cl, hi, lo = map(float, (cur.open, cur.close, cur.high, cur.low))
    pop, pcl, phi, plo = map(float, (prev.open, prev.close, prev.high, prev.low))

    # SNR zone: approximate recent support/resistance from the prior 20 completed bars.
    lookback = data.iloc[-21:-1]
    if len(lookback) >= 10:
        support = float(lookback.low.min())
        resistance = float(lookback.high.max())
        if lo <= support * 1.002 and cl > op and cl > support:
            out["BUY"].append(f"{label} SNR: bullish reaction near support")
        if hi >= resistance * 0.998 and cl < op and cl < resistance:
            out["SELL"].append(f"{label} SNR: bearish reaction near resistance")

    # FVG: most recent three-candle imbalance and current price reaction into it.
    gap = None
    for i in range(len(data)-1, max(1, len(data)-40), -1):
        left, right = data.iloc[i-2], data.iloc[i]
        if float(left.high) < float(right.low):
            gap = ("BUY", float(left.high), float(right.low))
            break
        if float(left.low) > float(right.high):
            gap = ("SELL", float(right.high), float(left.low))
            break
    if gap:
        if gap[0] == "BUY" and lo <= gap[2] and cl > op and cl >= gap[1]:
            out["BUY"].append(f"{label} FVG: bullish gap reaction")
        if gap[0] == "SELL" and hi >= gap[1] and cl < op and cl <= gap[2]:
            out["SELL"].append(f"{label} FVG: bearish gap reaction")

    # RB is a heuristic reversal-block proxy based on a reversal candle taking prior high/low.
    if pcl < pop and cl > op and cl > phi:
        out["BUY"].append(f"{label} RB: bullish reversal pattern")
    if pcl > pop and cl < op and cl < plo:
        out["SELL"].append(f"{label} RB: bearish reversal pattern")
    return out


def analyze(h1, h4, d1, w1):
    """W1/D1 context, H4 direction, H1 setup and H1 candle confirmation. No Stochastic/M15."""
    if min(len(h4), len(d1), len(w1)) < 205 or len(h1) < 35:
        return None, "Недостатньо історичних даних для W1/D1/H4/H1"

    all_tf = {
        "W1": timeframe_setups(w1, "W1"),
        "D1": timeframe_setups(d1, "D1"),
        "H4": timeframe_setups(h4, "H4"),
        "H1": timeframe_setups(h1, "H1"),
    }
    tw, td, th = trend(w1), trend(d1), trend(h4)
    if th == "NEUTRAL" and len(h4) >= 55:
        h4close = float(h4["close"].iloc[-2])
        h4ema50 = float(h4["close"].ewm(span=50, adjust=False).mean().iloc[-2])
        if h4close > h4ema50:
            th = "BULL"
        elif h4close < h4ema50:
            th = "BEAR"
    if th == "NEUTRAL":
        return None, f"H4 нейтральний; W1={tw}, D1={td}, H4={th}; SNR/RB/FVG перевірено"

    side = "BUY" if th == "BULL" else "SELL"
    reasons = []
    for tf in ("W1", "D1", "H4", "H1"):
        reasons.extend(all_tf[tf][side])
    if not reasons:
        return None, f"{side}: немає реакції SNR/RB/FVG на W1/D1/H4/H1"

    # H1 confirmation uses the latest COMPLETED H1 candle, never the live candle.
    h = h1.iloc[:-1]
    if len(h) < 6:
        return None, "Недостатньо закритих свічок H1"
    cur, prev = h.iloc[-1], h.iloc[-2]
    op, cl, prev_close = float(cur.open), float(cur.close), float(prev.close)
    h1_setups = all_tf["H1"][side]
    if side == "BUY":
        candle_confirm = cl > op and cl > prev_close
    else:
        candle_confirm = cl < op and cl < prev_close

    if not candle_confirm and not h1_setups:
        return None, f"{side}: є зона, але H1 ще не підтвердив напрямок закритою свічкою або реакцією SNR/RB/FVG"
    if candle_confirm:
        reasons.append("H1: підтвердження закритою свічкою в напрямку сигналу")
    if h1_setups:
        reasons.extend(h1_setups)

    entry = cl
    h1_high, h1_low = float(h.iloc[-1].high), float(h.iloc[-1].low)
    if side == "BUY":
        structure = min(h1_low, float(h.tail(5).low.min()))
        sl = structure - max(abs(entry) * 0.001, abs(h1_high-h1_low) * 0.25)
        risk = entry - sl
        if risk <= 0:
            return None, "Не вдалося коректно розрахувати SL"
        targets = [entry+risk, entry+2*risk, entry+3*risk]
    else:
        structure = max(h1_high, float(h.tail(5).high.max()))
        sl = structure + max(abs(entry) * 0.001, abs(h1_high-h1_low) * 0.25)
        risk = sl - entry
        if risk <= 0:
            return None, "Не вдалося коректно розрахувати SL"
        targets = [entry-risk, entry-2*risk, entry-3*risk]

    return {
        "side": side, "entry": entry, "sl": sl, "tp": targets,
        "reasons": reasons, "context": f"W1={tw}, D1={td}, H4={th}",
        "h1": str(h.index[-1]),
    }, "Signal"

def fmt(sym, x):
    if sym == "BTC/USD": return f"{x:,.2f}"
    if sym in ("XAU/USD","XAG/USD"): return f"{x:,.3f}"
    if "JPY" in sym: return f"{x:.3f}"
    return f"{x:.5f}" if abs(x)<10 else f"{x:.4f}"


def message(sym, s):
    icon = "🟢" if s["side"] == "BUY" else "🔴"
    why = "\n".join("• "+r for r in s["reasons"])
    return (f"{icon} <b>{s['side']} — {sym}</b>\n"
            f"Entry H1: {fmt(sym,s['entry'])}\nSL: {fmt(sym,s['sl'])}\n"
            f"TP1: {fmt(sym,s['tp'][0])}\nTP2: {fmt(sym,s['tp'][1])}\nTP3: {fmt(sym,s['tp'][2])}\n\n"
            f"Причини:\n{why}\nКонтекст: {s['context']}\nПідтвердження: H1 закрита свічка\nH1: {s['h1']}\n"
            "⚠️ Сигнал не гарантує прибутку; перевір спред і ризик.")


async def scan(app, report_id=None):
    sent, diagnostics = 0, []
    for sym, ticker in SYMBOLS.items():
        try:
            h1 = download(ticker, "60d", "1h")
            d1 = download(ticker, "2y", "1d")
            w1 = download(ticker, "5y", "1wk")
            if any(x.empty for x in (h1,d1,w1)):
                diagnostics.append(f"{sym}: немає даних від Yahoo Finance")
                continue
            h4 = h1.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
            setup, reason = analyze(h1,h4,d1,w1)
            if not setup:
                diagnostics.append(f"{sym}: {reason}")
                continue
            key = f"{sym}|{setup['side']}|{setup['h1']}"
            if key in ALERTED:
                diagnostics.append(f"{sym}: сигнал для цієї свічки вже надсилався")
                continue
            ALERTED.add(key)
            for chat in list(SUBSCRIBERS):
                try:
                    await app.bot.send_message(chat_id=chat,text=message(sym,setup),parse_mode="HTML")
                    sent += 1
                except Exception:
                    log.exception("Message send failed")
        except Exception:
            log.exception("Scan failed for %s", sym)
            diagnostics.append(f"{sym}: помилка сканування; дивись Railway logs")
    if report_id is not None:
        await app.bot.send_message(chat_id=report_id,text=f"Сканування завершено. Надіслано сигналів: {sent}\n\nДіагностика:\n"+"\n".join("• "+x for x in diagnostics))


async def job(ctx: ContextTypes.DEFAULT_TYPE):
    await scan(ctx.application)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    save_subscribers()
    await update.effective_message.reply_text(
        f"Бот активний. Сканування кожні {INTERVAL} хв.\n"
        "W1/D1 контекст → H4 напрямок → H1 зона та підтвердження.\n"
        "Входи: SNR, RB-патерн, FVG; без Stochastic і M15.\n"
        "Команди: /status /symbols /scan /help")


async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        f"Запущений. Інтервал {INTERVAL} хв. Підписників: {len(SUBSCRIBERS)}.\n"
        "W1/D1 → H4 → H1 підтвердження. Stochastic вимкнений; автоматичні угоди вимкнені.")


async def symbols(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Інструменти:\n"+"\n".join("• "+s for s in SYMBOLS))


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("/start — підписатися\n/status — стан\n/symbols — список\n/scan — сканування з діагностикою\n/help — допомога")


async def scan_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Сканую 9 інструментів; це може зайняти кілька хвилин.")
    await scan(ctx.application, update.effective_chat.id)


async def post_init(app: Application):
    load_subscribers()
    if app.job_queue is None:
        raise RuntimeError('Додай python-telegram-bot[job-queue] до requirements.txt')
    app.job_queue.run_repeating(job, interval=INTERVAL*60, first=15, name="market-scan")
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
