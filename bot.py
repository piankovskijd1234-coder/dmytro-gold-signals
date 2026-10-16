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


def stochastic(df):
    lo, hi = df["low"].rolling(14).min(), df["high"].rolling(14).max()
    raw = 100 * (df["close"] - lo) / (hi - lo).replace(0, float("nan"))
    k = raw.rolling(3).mean()
    return k, k.rolling(3).mean()


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


def analyze(h1, m15, h4, d1, w1):
    if min(len(h4), len(d1), len(w1)) < 205 or len(h1) < 35 or len(m15) < 40:
        return None, "Недостатньо історії для W1/D1/H4/H1/M15"
    tw, td, th = trend(w1), trend(d1), trend(h4)
    if th == "NEUTRAL":
        return None, f"H4 нейтральний (W1={tw}, D1={td}, H4={th})"
    side = "BUY" if th == "BULL" else "SELL"
    h = h1.iloc[:-1]
    a, b = h.iloc[-1], h.iloc[-2]
    op, cl, hi, lo = map(float, (a.open, a.close, a.high, a.low))
    pop, pcl, phi, plo = map(float, (b.open, b.close, b.high, b.low))
    support, resistance = snr(h1)
    gap = fvg(h1)
    reasons = []
    if side == "BUY":
        if support and lo <= support * 1.002 and cl > op and cl > support:
            reasons.append("SNR support reaction")
        if gap and gap[0] == "BULL" and lo <= gap[2] and cl > op and cl >= gap[1]:
            reasons.append("Bullish FVG reaction")
        if pcl < pop and cl > op and cl > phi:
            reasons.append("RB-style bullish reversal")
    else:
        if resistance and hi >= resistance * .998 and cl < op and cl < resistance:
            reasons.append("SNR resistance reaction")
        if gap and gap[0] == "BEAR" and hi >= gap[1] and cl < op and cl <= gap[2]:
            reasons.append("Bearish FVG reaction")
        if pcl > pop and cl < op and cl < plo:
            reasons.append("RB-style bearish reversal")
    prev = h.iloc[-22:-2]
    if len(prev) >= 10:
        r, s = float(prev.high.max()), float(prev.low.min())
        if side == "BUY" and pcl > r and lo <= r * 1.002 and cl > r and cl > op:
            reasons.append("Breakout + retest")
        if side == "SELL" and pcl < s and hi >= s * .998 and cl < s and cl < op:
            reasons.append("Breakdown + retest")
    # Final confirmation on the last fully closed M15 candle.
    m = m15.iloc[:-1]
    x, y = m.iloc[-1], m.iloc[-2]
    mop, mcl, mpcl = float(x.open), float(x.close), float(y.close)
    k, d = stochastic(m)
    stoch_reason = None
    if len(k.dropna()) >= 2 and len(d.dropna()) >= 2:
        kp, kn, dp, dn = float(k.iloc[-2]), float(k.iloc[-1]), float(d.iloc[-2]), float(d.iloc[-1])
        if kp < 20 and dp < 20 and kn >= 20 and kp <= dp and kn > dn:
            stoch_reason = ("BUY", "Stochastic exit from oversold + K/D cross")
        elif kp > 80 and dp > 80 and kn <= 80 and kp >= dp and kn < dn:
            stoch_reason = ("SELL", "Stochastic exit from overbought + K/D cross")
    if stoch_reason and stoch_reason[0] == side:
        reasons.append(stoch_reason[1])
    if not reasons:
        return None, f"{side}: H1 не знайшов підтвердження SNR/RB/FVG/Stochastic"
    if side == "BUY" and not (mcl > mop and mcl > mpcl):
        return None, "H1 зона знайдена, але немає бичачого підтвердження M15"
    if side == "SELL" and not (mcl < mop and mcl < mpcl):
        return None, "H1 зона знайдена, але немає ведмежого підтвердження M15"
    entry = mcl
    if side == "BUY":
        sl = min(lo, float(h.tail(5).low.min())) - max(abs(entry)*.001, abs(hi-lo)*.25)
        risk = entry-sl
        if risk <= 0: return None, "SL invalid"
        targets = [entry+risk, entry+2*risk, entry+3*risk]
    else:
        sl = max(hi, float(h.tail(5).high.max())) + max(abs(entry)*.001, abs(hi-lo)*.25)
        risk = sl-entry
        if risk <= 0: return None, "SL invalid"
        targets = [entry-risk, entry-2*risk, entry-3*risk]
    return {"side":side,"entry":entry,"sl":sl,"tp":targets,"reasons":reasons,
            "context":f"W1={tw}, D1={td}, H4={th}","h1":str(h.index[-1]),"m15":str(m.index[-1])}, "Signal"


def fmt(sym, x):
    if sym == "BTC/USD": return f"{x:,.2f}"
    if sym in ("XAU/USD","XAG/USD"): return f"{x:,.3f}"
    if "JPY" in sym: return f"{x:.3f}"
    return f"{x:.5f}" if abs(x)<10 else f"{x:.4f}"


def message(sym, s):
    icon = "🟢" if s["side"] == "BUY" else "🔴"
    why = "\n".join("• "+r for r in s["reasons"])
    return (f"{icon} <b>{s['side']} — {sym}</b>\n"
            f"Entry M15: {fmt(sym,s['entry'])}\nSL: {fmt(sym,s['sl'])}\n"
            f"TP1: {fmt(sym,s['tp'][0])}\nTP2: {fmt(sym,s['tp'][1])}\nTP3: {fmt(sym,s['tp'][2])}\n\n"
            f"Причини:\n{why}\nКонтекст: {s['context']}\nH1: {s['h1']}\nM15: {s['m15']}\n"
            "⚠️ Сигнал не гарантує прибутку; перевір спред і ризик.")


async def scan(app, report_id=None):
    sent, diagnostics = 0, []
    for sym, ticker in SYMBOLS.items():
        try:
            h1 = download(ticker, "60d", "1h")
            m15 = download(ticker, "60d", "15m")
            d1 = download(ticker, "2y", "1d")
            w1 = download(ticker, "5y", "1wk")
            if any(x.empty for x in (h1,m15,d1,w1)):
                diagnostics.append(f"{sym}: немає даних від Yahoo Finance")
                continue
            h4 = h1.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
            setup, reason = analyze(h1,m15,h4,d1,w1)
            if not setup:
                diagnostics.append(f"{sym}: {reason}")
                continue
            key = f"{sym}|{setup['side']}|{setup['m15']}"
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
        "W1/D1 контекст → H4 напрямок → H1 зона → M15 підтвердження.\n"
        "Входи: SNR, RB-патерн, FVG, Stochastic OB/OS, пробій + ретест.\n"
        "Команди: /status /symbols /scan /help")


async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        f"Запущений. Інтервал {INTERVAL} хв. Підписників: {len(SUBSCRIBERS)}.\n"
        "W1/D1 → H4 → H1 → M15. Автоматичні угоди вимкнені.")


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
