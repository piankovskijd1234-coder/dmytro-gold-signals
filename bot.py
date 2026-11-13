import os, logging
from datetime import datetime, timezone, timedelta
import pandas as pd
import yfinance as yf
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"), format="%(asctime)s %(levelname)s %(message)s")
log=logging.getLogger("tda-pro")
TOKEN=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
FINNHUB=os.getenv("FINNHUB_API_KEY","").strip()
SYMBOLS={"BTCUSD":"BTC-USD","EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","EURJPY":"EURJPY=X","AUDUSD":"AUDUSD=X","XAGUSD":"XAGUSD=X","EURAUD":"EURAUD=X","USDCHF":"CHF=X","GBPAUD":"GBPAUD=X","GER40":"^GDAXI","XAUUSD":"GC=F"}
TF={"1wk":("1wk","5y"),"1d":("1d","2y"),"4h":("1h","730d"),"1h":("1h","730d"),"15m":("15m","60d"),"5m":("5m","60d"),"1m":("1m","7d")}

def fetch(sym, tf):
    interval,period=TF[tf]
    d=yf.download(SYMBOLS[sym],period=period,interval=interval,progress=False,auto_adjust=False,threads=False)
    if d is None or d.empty: return pd.DataFrame()
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    d=d.rename(columns=str.title)
    cols=["Open","High","Low","Close","Volume"]
    if not all(x in d.columns for x in cols): return pd.DataFrame()
    d=d[cols].dropna()
    if tf=="4h": d=d.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    return d

def closed(d): return d.iloc[:-1] if len(d)>3 else d
def bias(d):
    if len(d)<55:return "NEUTRAL"
    c=d.Close; e20=c.ewm(span=20,adjust=False).mean().iloc[-1]; e50=c.ewm(span=50,adjust=False).mean().iloc[-1]
    return "BULL" if c.iloc[-1]>e20>e50 else "BEAR" if c.iloc[-1]<e20<e50 else "NEUTRAL"
def price(x): return f"{x:.5f}" if abs(x)<100 else f"{x:.2f}"

def h1_bos_retest(d):
    if len(d)<30:return None,"мало H1 даних"
    d=closed(d); cur=d.iloc[-1]; prev=d.iloc[-14:-2]
    hi=float(prev.High.max()); lo=float(prev.Low.min())
    direction="BUY" if cur.Close>hi and cur.Close>cur.Open else "SELL" if cur.Close<lo and cur.Close<cur.Open else None
    if not direction:return None,"немає H1 BOS"
    recent=d.iloc[-4:]
    if direction=="BUY":
        ok=((recent.Low<=hi+(cur.High-cur.Low)*.2)&(recent.Close>=hi)).any()
        zone=(hi,hi+(cur.High-cur.Low)*.2)
    else:
        ok=((recent.High>=lo-(cur.High-cur.Low)*.2)&(recent.Close<=lo)).any()
        zone=(lo-(cur.High-cur.Low)*.2,lo)
    return (direction,zone) if ok else (None,"BOS є, ретест не підтверджений")

def m15_confirm(d,side):
    if len(d)<5:return False
    d=closed(d); c=d.iloc[-1]; p=d.iloc[-2]
    return c.Close>c.Open and c.Close>p.High if side=="BUY" else c.Close<c.Open and c.Close<p.Low

def scalp(d5,d1,side):
    if len(d5)<25 or len(d1)<5:return None
    d=closed(d5); impulse=d.iloc[-8:-2]; recent=d.iloc[-3:]; hi=float(impulse.High.max()); lo=float(impulse.Low.min()); r=hi-lo
    if r<=0:return None
    c=d.iloc[-1]; m=closed(d1).iloc[-1]
    if side=="BUY":
        zl,zh=hi-.62*r,hi-.50*r
        touched=((recent.Low<=zh)&(recent.Low>=zl)).any()
        confirm=c.Close>c.Open and m.Close>m.Open
        sl=float(min(recent.Low.min(),lo)); tp=hi
    else:
        zl,zh=lo+.50*r,lo+.62*r
        touched=((recent.High>=zl)&(recent.High<=zh)).any()
        confirm=c.Close<c.Open and m.Close<m.Open
        sl=float(max(recent.High.max(),hi)); tp=lo
    entry=float(c.Close); risk=abs(entry-sl); rr=abs(tp-entry)/risk if risk else 0
    return {"zone":(zl,zh),"entry":entry,"sl":sl,"tp":tp,"rr":rr} if touched and confirm and rr>=.7 else None

def analyze(sym):
    try:
        ds={t:fetch(sym,t) for t in ("1wk","1d","4h","1h","15m")}
        if any(x.empty for x in ds.values()):return [],f"{sym}: немає даних на одному з ТФ"
        w,d,h=bias(ds["1wk"]),bias(ds["1d"]),bias(ds["4h"])
        if w==d==h and w in ("BULL","BEAR"): grade,side="A+","BUY" if w=="BULL" else "SELL"
        elif d==h and d in ("BULL","BEAR") and w=="NEUTRAL":grade,side="A","BUY" if d=="BULL" else "SELL"
        else:return [],f"{sym}: пропуск — W1 {w}, D1 {d}, H4 {h} не узгоджені"
        setup,msg=h1_bos_retest(ds["1h"])
        if not setup or setup[0]!=side:return [],f"{sym}: {msg if not setup else 'H1 проти HTF'}"
        if not m15_confirm(ds["15m"],side):return [],f"{sym}: немає M15 підтвердження"
        h1=closed(ds["1h"]); entry=float(h1.Close.iloc[-1])
        sl=float(h1.Low.iloc[-8:].min()) if side=="BUY" else float(h1.High.iloc[-8:].max())
        tp=float(h1.High.iloc[-30:-1].max()) if side=="BUY" else float(h1.Low.iloc[-30:-1].min())
        risk=abs(entry-sl); rr=abs(tp-entry)/risk if risk else 0
        if rr<.8:return [],f"{sym}: RR {rr:.2f} нижче 0.8"
        out=[f"🟢 {sym} | INTRADAY {side} | {grade}",f"Контекст W1 {w} / D1 {d} / H4 {h}", "Підтвердження: H1 BOS + retest, M15 confirm",f"Вхід-орієнтир: {price(entry)}",f"SL: {price(sl)}",f"TP1: {price(tp)} | RR ≈ 1:{rr:.2f}","Перевір спред і котирування у брокера."]
        d5,d1=fetch(sym,"5m"),fetch(sym,"1m")
        if not d5.empty and not d1.empty:
            sc=scalp(d5,d1,side)
            if sc:out += ["",f"⚡ {sym} | SCALP {side}",f"Зона повернення: {price(sc['zone'][0])} – {price(sc['zone'][1])}",f"Вхід після M5/M1 підтвердження: {price(sc['entry'])}",f"SL: {price(sc['sl'])}",f"TP: {price(sc['tp'])} | RR ≈ 1:{sc['rr']:.2f}","Перевір спред у брокера."]
        return out,f"{sym}: умови виконані ({grade})"
    except Exception as e:
        log.exception("scan %s",sym); return [],f"{sym}: помилка {type(e).__name__}"

def news_text():
    if not FINNHUB:return "Finnhub не підключений. Додай FINNHUB_API_KEY у Railway → Variables."
    now=datetime.now(timezone.utc); params={"from":now.strftime("%Y-%m-%d"),"to":(now+timedelta(days=2)).strftime("%Y-%m-%d"),"token":FINNHUB}
    try:
        r=requests.get("https://finnhub.io/api/v1/calendar/economic",params=params,timeout=15); r.raise_for_status()
        events=r.json().get("economicCalendar",[])
        events=[e for e in events if str(e.get("impact","")).lower() in ("high","3")]
        if not events:return "На найближчі 48 годин немає повернутих Finnhub подій високого впливу."
        lines=["📰 ВАЖЛИВІ НОВИНИ (час API):"]
        for e in events[:12]:lines.append(f"• {e.get('time','—')} | {e.get('country','')} | {e.get('event','')} | факт {e.get('actual','—')} / прогноз {e.get('estimate','—')} / попер. {e.get('prev','—')}")
        lines.append("Сюрприз даних не гарантує напрямок ціни.")
        return "\n".join(lines)
    except Exception as e:log.exception("Finnhub"); return f"Помилка Finnhub: {type(e).__name__}; перевір API-доступ і Railway Logs."

async def start(u,c):
    await u.message.reply_text("🤖 Dmytro Gold Signals — TDA PRO\n/scan — сканування\n/status — стан\n/news — календар новин\n\nINTRADAY: W1/D1/H4 → H1 BOS+Retest → M15.\nSCALP: повернення в зону → підтвердження M5/M1.\nЦе аналітичні сигнали, не автоматичне відкриття угод.")
async def status(u,c):
    await u.message.reply_text(f"🟢 TDA PRO запущений\nA+/A: W1/D1/H4 alignment; H1 BOS+Retest; M15 confirmation.\nОкремий SCALP: pullback zone + M5/M1 candle confirm.\nІнструментів: {len(SYMBOLS)}\nFinnhub: {'ключ заданий' if FINNHUB else 'не підключений'}\nДані: Yahoo Finance, можуть запізнюватися/відрізнятися від брокера. Реальний footprint/order flow та спред не отримуються.")
async def scan_cmd(u,c):
    m=await u.message.reply_text("🔎 Сканую ринки…")
    signals=[]; diag=[]
    for s in SYMBOLS:
        found,reason=analyze(s)
        if found:signals.extend(found)
        diag.append(reason)
    text="\n\n".join(signals) if signals else "Сигналів немає — фільтр не пропускає слабкі входи.\n\n"+"\n".join(diag)
    await m.edit_text(text[:3900])
async def news_cmd(u,c):await u.message.reply_text(news_text()[:3900])
async def stop_cmd(u,c):await u.message.reply_text("Для повної зупинки вимкни Worker у Railway. Команда не вимикає серверний процес.")
def main():
    if not TOKEN:raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")
    app=Application.builder().token(TOKEN).build()
    for cmd,fn in [("start",start),("status",status),("scan",scan_cmd),("news",news_cmd),("stop",stop_cmd)]:app.add_handler(CommandHandler(cmd,fn))
    log.info("TDA PRO started");app.run_polling(drop_pending_updates=True)
if __name__=="__main__":main()
