import os, logging, asyncio
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
# Live in-memory subscriptions/alerts. They reset if Railway restarts.
SUBSCRIBERS=set()
ACTIVE_ALERTS={}  # key -> alert dictionary
SENT_ALERTS=set()
LAST_SCAN=datetime.min.replace(tzinfo=timezone.utc)

def fetch(sym,tf):
    interval,period=TF[tf]
    d=yf.download(SYMBOLS[sym],period=period,interval=interval,progress=False,auto_adjust=False,threads=False)
    if d is None or d.empty:return pd.DataFrame()
    if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
    d=d.rename(columns=str.title)
    cols=["Open","High","Low","Close","Volume"]
    if not all(x in d.columns for x in cols):return pd.DataFrame()
    d=d[cols].dropna()
    if tf=="4h":d=d.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    return d

def closed(d):return d.iloc[:-1] if len(d)>3 else d
def bias(d):
    if len(d)<55:return "NEUTRAL"
    c=d.Close;e20=c.ewm(span=20,adjust=False).mean().iloc[-1];e50=c.ewm(span=50,adjust=False).mean().iloc[-1]
    return "BULL" if c.iloc[-1]>e20>e50 else "BEAR" if c.iloc[-1]<e20<e50 else "NEUTRAL"
def fmt(x):return f"{x:.5f}" if abs(x)<100 else f"{x:.2f}"

def structure_h1(d):
    if len(d)<35:return None
    d=closed(d);found=None
    # Search BOS in last 12 closed candles, not only the latest candle.
    for i in range(max(15,len(d)-12),len(d)):
        candle=d.iloc[i];prior=d.iloc[max(0,i-12):i]
        if len(prior)<10:continue
        hi=float(prior.High.max());lo=float(prior.Low.min())
        if candle.Close>hi and candle.Close>candle.Open:
            found=("BUY",hi,float(candle.High-candle.Low),i)
        elif candle.Close<lo and candle.Close<candle.Open:
            found=("SELL",lo,float(candle.High-candle.Low),i)
    if not found:return None
    side,level,bar_range,index=found
    after=d.iloc[index+1:]
    if len(after)==0:return None
    recent=after.iloc[-6:]
    pad=max(bar_range*.20,abs(level)*0.00002)
    if side=="BUY":
        zone=(level,level+pad)
        touched=((recent.Low<=zone[1])&(recent.Close>=zone[0])).any()
    else:
        zone=(level-pad,level)
        touched=((recent.High>=zone[0])&(recent.Close<=zone[1])).any()
    return {"side":side,"zone":zone,"level":level,"touched":bool(touched)}

def m15_confirm(d,side):
    if len(d)<5:return False
    d=closed(d);c=d.iloc[-1];p=d.iloc[-2]
    return c.Close>c.Open and c.Close>p.High if side=="BUY" else c.Close<c.Open and c.Close<p.Low

def make_intraday(sym):
    frames={t:fetch(sym,t) for t in ("1wk","1d","4h","1h","15m")}
    if any(x.empty for x in frames.values()):return None,"немає даних на одному з ТФ"
    w,d,h=[bias(frames[t]) for t in ("1wk","1d","4h")]
    if w==d==h and w in ("BULL","BEAR"):grade,side="A+","BUY" if w=="BULL" else "SELL"
    elif d==h and d in ("BULL","BEAR") and w=="NEUTRAL":grade,side="A","BUY" if d=="BULL" else "SELL"
    else:return None,f"HTF не узгоджені: W1 {w}, D1 {d}, H4 {h}"
    st=structure_h1(frames["1h"])
    if not st or st["side"]!=side:return None,"немає відповідного H1 BOS/зони ретесту"
    if not st["touched"]:return None,"BOS знайдений, очікуємо ретест H1"
    if not m15_confirm(frames["15m"],side):return None,"очікуємо підтвердження M15"
    h=closed(frames["1h"]);entry=float(h.Close.iloc[-1])
    sl=float(h.Low.iloc[-8:].min()) if side=="BUY" else float(h.High.iloc[-8:].max())
    tp=float(h.High.iloc[-30:-1].max()) if side=="BUY" else float(h.Low.iloc[-30:-1].min())
    risk=abs(entry-sl);rr=abs(tp-entry)/risk if risk else 0
    if rr<.8:return None,"RR нижче 0.8"
    return {"symbol":sym,"mode":"INTRADAY","side":side,"grade":grade,"zone":st["zone"],"entry":entry,"sl":sl,"tp":tp,"rr":rr},None

def make_scalp(sym,side):
    d5,d1=fetch(sym,"5m"),fetch(sym,"1m")
    if d5.empty or d1.empty or len(d5)<25 or len(d1)<5:return None
    d=closed(d5);imp=d.iloc[-8:-2];hi=float(imp.High.max());lo=float(imp.Low.min());r=hi-lo
    if r<=0:return None
    if side=="BUY":zone=(hi-.62*r,hi-.50*r);sl=float(min(d.iloc[-3:].Low.min(),lo));tp=hi
    else:zone=(lo+.50*r,lo+.62*r);sl=float(max(d.iloc[-3:].High.max(),hi));tp=lo
    c=d.iloc[-1];m=closed(d1).iloc[-1]
    touched=(c.Low<=zone[1] and c.High>=zone[0])
    confirm=(c.Close>c.Open and m.Close>m.Open) if side=="BUY" else (c.Close<c.Open and m.Close<m.Open)
    entry=float(c.Close);risk=abs(entry-sl);rr=abs(tp-entry)/risk if risk else 0
    if touched and confirm and rr>=.7:
        return {"symbol":sym,"mode":"SCALP","side":side,"grade":"SCALP","zone":zone,"entry":entry,"sl":sl,"tp":tp,"rr":rr}
    return None

def current_price(sym):
    d=fetch(sym,"1m")
    return float(d.Close.iloc[-1]) if not d.empty else None

def fmt_signal(s):
    return (f"✅ {s['symbol']} | {s['mode']} {s['side']} | {s['grade']}\n"
            f"Зона: {fmt(s['zone'][0])} – {fmt(s['zone'][1])}\n"
            f"Орієнтир входу: {fmt(s['entry'])}\nSL: {fmt(s['sl'])}\nTP: {fmt(s['tp'])} | RR ≈ 1:{s['rr']:.2f}\n"
            "Перевір актуальну ціну та спред у брокера. Це не наказ відкривати угоду.")

async def scan_all():
    results=[];diagnostics=[]
    for sym in SYMBOLS:
        s,why=await asyncio.to_thread(make_intraday,sym)
        if s:
            results.append(s)
            scalp_s=await asyncio.to_thread(make_scalp,sym,s["side"])
            if scalp_s:results.append(scalp_s)
        else:diagnostics.append(f"{sym}: {why}")
    return results,diagnostics

async def start(u,c):
    SUBSCRIBERS.add(u.effective_chat.id)
    await u.message.reply_text("🤖 TDA PRO 2.0 активний.\nПопередження про наближення до зони надходитиме для INTRADAY і SCALP.\nКоманди: /scan /status /news /alerts_on /alerts_off\nВажливо: «за 5 хвилин» — оцінка за відстанню та недавнім рухом ціни, не гарантія часу входу.")
async def alerts_on(u,c):
    SUBSCRIBERS.add(u.effective_chat.id);await u.message.reply_text("🔔 Попередження увімкнені для INTRADAY і SCALP.")
async def alerts_off(u,c):
    SUBSCRIBERS.discard(u.effective_chat.id);await u.message.reply_text("🔕 Автоматичні попередження вимкнені.")
async def status(u,c):
    await u.message.reply_text(f"🟢 TDA PRO 2.0 працює\nІнструментів: {len(SYMBOLS)}\nАвтопопередження: {'увімкнені' if u.effective_chat.id in SUBSCRIBERS else 'вимкнені'}\nПеріод перевірки: 1 хвилина\nFinnhub: {'ключ заданий' if FINNHUB else 'не підключений'}\nДжерело: Yahoo Finance; може запізнюватися. Реальний footprint/order flow і спред не отримуються.")
async def scan_cmd(u,c):
    m=await u.message.reply_text("🔎 Сканую…")
    signals,diag=await scan_all()
    if signals:out="\n\n".join(fmt_signal(s) for s in signals)
    else:out="Сигналів немає.\n\n"+"\n".join(diag)
    await m.edit_text(out[:3900])
async def news_cmd(u,c):
    if not FINNHUB:await u.message.reply_text("Finnhub не підключений. Додай FINNHUB_API_KEY у Railway Variables.");return
    now=datetime.now(timezone.utc)
    try:
        r=requests.get("https://finnhub.io/api/v1/calendar/economic",params={"from":now.strftime("%Y-%m-%d"),"to":(now+timedelta(days=2)).strftime("%Y-%m-%d"),"token":FINNHUB},timeout=15);r.raise_for_status()
        ev=[e for e in r.json().get("economicCalendar",[]) if str(e.get("impact","")).lower() in ("high","3")]
        lines=["📰 Важливі новини (час API/UTC):"]+[f"• {e.get('time','—')} | {e.get('country','')} | {e.get('event','')} | факт {e.get('actual','—')} / прогноз {e.get('estimate','—')} / попер. {e.get('prev','—')}" for e in ev[:12]]
        lines.append("Сюрприз даних не гарантує напрямок ціни.");await u.message.reply_text("\n".join(lines)[:3900])
    except Exception as e:log.exception("Finnhub");await u.message.reply_text(f"Помилка Finnhub: {type(e).__name__}")
async def stop_cmd(u,c):await alerts_off(u,c)

async def alert_job(context:ContextTypes.DEFAULT_TYPE):
    # Refresh candidate setups every 5 min; monitor current price each minute.
    global LAST_SCAN
    now=datetime.now(timezone.utc)
    if not SUBSCRIBERS:return
    if (now-LAST_SCAN).total_seconds()>300:
        try:
            setups,_=await scan_all()
            ACTIVE_ALERTS.clear()
            for s in setups:
                key=f"{s['symbol']}:{s['mode']}:{s['side']}"
                ACTIVE_ALERTS[key]=s
            LAST_SCAN=now
        except Exception:log.exception("refresh setups")
    for key,s in list(ACTIVE_ALERTS.items()):
        try:
            p=await asyncio.to_thread(current_price,s["symbol"])
            if p is None:continue
            lo,hi=sorted(s["zone"]);dist=min(abs(p-lo),abs(p-hi))
            # Estimate time to zone from recent 1m absolute candle movement.
            d=await asyncio.to_thread(fetch,s["symbol"],"1m")
            if d.empty or len(d)<8:continue
            avg=float((d.High-d.Low).tail(5).mean())
            if avg<=0:continue
            eta=(dist/avg)*60.0
            alert_key=f"{key}:{s['entry']:.8f}"
            if 60<=eta<=360 and alert_key not in SENT_ALERTS:
                SENT_ALERTS.add(alert_key)
                msg=(f"⏳ ПОПЕРЕДЖЕННЯ ~5 ХВ ДО ЗОНИ\n{s['symbol']} | {s['mode']} {s['side']}\n"
                     f"Поточна ціна: {fmt(p)}\nЗона: {fmt(lo)} – {fmt(hi)}\n"
                     f"Оцінка часу за недавнім рухом: ~{max(1,round(eta/60))} хв.\n"
                     "Час не гарантований. Чекай фактичного повернення та підтвердження M5/M1.")
                for chat_id in list(SUBSCRIBERS):
                    await context.bot.send_message(chat_id=chat_id,text=msg)
            # If zone reached, expire pre-alert; final confirmation remains a manual /scan check.
            if lo<=p<=hi:ACTIVE_ALERTS.pop(key,None)
        except Exception:log.exception("alert monitor %s",key)

def main():
    if not TOKEN:raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
    app=Application.builder().token(TOKEN).build()
    for cmd,fn in [("start",start),("alerts_on",alerts_on),("alerts_off",alerts_off),("status",status),("scan",scan_cmd),("news",news_cmd),("stop",stop_cmd)]:
        app.add_handler(CommandHandler(cmd,fn))
    app.job_queue.run_repeating(alert_job,interval=60,first=15,name="zone-prealerts")
    log.info("TDA PRO 2.0 started");app.run_polling(drop_pending_updates=True)
if __name__=="__main__":main()
