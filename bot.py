import os, json, logging
from datetime import timezone
import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN="".join(os.getenv("TELEGRAM_BOT_TOKEN","").split())
INTERVAL=max(5,int(os.getenv("SCAN_INTERVAL_MINUTES","15")))
SUBSCRIBERS_FILE="subscribers.json"
SYMBOLS={"BTCUSD":"BTC-USD","EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","EURJPY":"EURJPY=X","AUDUSD":"AUDUSD=X","XAGUSD":"XAGUSD=X","EURAUD":"EURAUD=X","USDCHF":"CHF=X","GBPAUD":"GBPAUD=X"}
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log=logging.getLogger("dmytro-gold-signals")
subs=set()
sent_keys=set()

def load_subs():
    global subs
    try:
        with open(SUBSCRIBERS_FILE,encoding="utf-8") as f: subs={int(x) for x in json.load(f)}
    except Exception: subs=set()

def save_subs():
    try:
        with open(SUBSCRIBERS_FILE,"w",encoding="utf-8") as f: json.dump(sorted(subs),f)
    except Exception: log.exception("save subscribers failed")

def get_data(ticker,period,interval):
    try:
        d=yf.download(ticker,period=period,interval=interval,progress=False,auto_adjust=False,threads=False)
        if d is None or d.empty: return pd.DataFrame()
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        d.columns=[str(c).title() for c in d.columns]
        cols=["Open","High","Low","Close","Volume"]
        return d[cols].dropna() if all(c in d.columns for c in cols) else pd.DataFrame()
    except Exception:
        log.exception("download failed %s %s",ticker,interval); return pd.DataFrame()

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def atr(d):
    pc=d.Close.shift()
    tr=pd.concat([d.High-d.Low,(d.High-pc).abs(),(d.Low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(14).mean()
def rsi(s):
    delta=s.diff()
    up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    dn=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=up/dn.replace(0,float("nan"))
    return 100-100/(1+rs)
def bias(d):
    if len(d)<200: return "NONE"
    e50,e200=ema(d.Close,50).iloc[-1],ema(d.Close,200).iloc[-1]
    c=d.Close.iloc[-1]
    return "BUY" if c>e50>e200 else ("SELL" if c<e50<e200 else "NONE")

def analyze(symbol,ticker):
    h1=get_data(ticker,"60d","1h"); h4=get_data(ticker,"730d","4h"); d1=get_data(ticker,"5y","1d")
    if min(len(h1),len(h4),len(d1))<205: raise ValueError("Недостатньо історичних даних H1/H4/D1")
    b4,b1=bias(h4),bias(d1)
    if b4=="NONE" or b4!=b1: return None
    h=h1.iloc[:-1].copy()
    h["e20"]=ema(h.Close,20); h["e50"]=ema(h.Close,50); h["atr"]=atr(h); h["rsi"]=rsi(h.Close)
    h=h.dropna()
    if len(h)<30: return None
    c=h.iloc[-1]; prev=h.iloc[-21:-1]; recent=h.iloc[-6:-1]
    level=float(prev.High.max()) if b4=="BUY" else float(prev.Low.min())
    av=float(c.atr); close=float(c.Close); rv=float(c.rsi)
    if not av or pd.isna(av): return None
    if b4=="BUY":
        ok=float(recent.Close.max())>level and float(c.Low)<=level+.25*av and close>level and close>float(c.Open) and close>c.e20>c.e50 and 50<=rv<=72
        if not ok: return None
        entry=close; sl=min(float(recent.Low.min()),level-.35*av); risk=entry-sl
        if risk<=0:return None
        tp=entry+2*risk
        why="H4/D1 BUY; пробій структури вгору; ретест рівня втримано; EMA та RSI підтверджують рух."
    else:
        ok=float(recent.Close.min())<level and float(c.High)>=level-.25*av and close<level and close<float(c.Open) and close<c.e20<c.e50 and 28<=rv<=50
        if not ok:return None
        entry=close; sl=max(float(recent.High.max()),level+.35*av); risk=sl-entry
        if risk<=0:return None
        tp=entry-2*risk
        why="H4/D1 SELL; пробій структури вниз; ретест рівня відхилено; EMA та RSI підтверджують рух."
    ts=h.index[-1]
    if getattr(ts,"tzinfo",None) is None: ts=ts.replace(tzinfo=timezone.utc)
    ts=ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return dict(symbol=symbol,side=b4,entry=entry,sl=sl,tp=tp,rsi=rv,time=ts,why=why)

def price(x): return f"{x:.2f}" if abs(x)>=1000 else (f"{x:.4f}" if abs(x)>=10 else f"{x:.5f}")
def message(s):
    return (f"📊 SIGNAL — {s['symbol']}\\nТаймфрейм: H1; фільтр H4 + D1\\nНапрямок: {s['side']}\\n"
    f"Стратегія: Trend Pullback + BOS + Retest\\nEntry: {price(s['entry'])}\\nSL: {price(s['sl'])}\\nTP: {price(s['tp'])}\\n"
    f"Ризик/прибуток: 1:2\\nRSI: {s['rsi']:.1f}\\nОбґрунтування: {s['why']}\\nЗакрита H1-свічка: {s['time']}\\n"
    "⚠️ Дані Yahoo можуть бути затримані й відрізнятися від брокера. Перевір спред, новини та ризик.")

async def start(u:Update,c:ContextTypes.DEFAULT_TYPE):
    subs.add(u.effective_chat.id); save_subs()
    await u.message.reply_text("Бот активний. Одна стратегія: Trend Pullback + BOS + Retest. Команди: /status /symbols /scan")
async def status(u,c):
    await u.message.reply_text(f"Бот запущений. Інструментів: 9. Стратегія: Trend Pullback + BOS + Retest. Таймфрейми: H1 (вхід), H4 і D1 (фільтр). Один TP, R:R 1:2. Інтервал: {INTERVAL} хв.")
async def symbols(u,c): await u.message.reply_text("Інструменти: "+", ".join(SYMBOLS))
async def run_scan(chat_id,ctx):
    n=err=0
    for sym,tick in SYMBOLS.items():
        try:
            s=analyze(sym,tick)
            if s:
                key=f"{sym}:{s['side']}:{s['time']}"
                if key not in sent_keys:
                    await ctx.bot.send_message(chat_id=chat_id,text=message(s)); sent_keys.add(key); n+=1
        except Exception: err+=1; log.exception("scan error %s",sym)
    return n,err
async def scan(u,c):
    await u.message.reply_text("Сканую 9 інструментів. Це може тривати кілька хвилин.")
    n,e=await run_scan(u.effective_chat.id,c)
    await u.message.reply_text(f"Перевірку завершено. Нових сигналів: {n}. Помилок даних: {e}. Деталі — Railway Logs.")
async def scheduled(c):
    for chat in list(subs):
        try: await run_scan(chat,c)
        except Exception: log.exception("scheduled scan failed")
async def help_cmd(u,c): await u.message.reply_text("/start — підписка\\n/status — стан\\n/symbols — інструменти\\n/scan — ручне сканування")
def main():
    if not TOKEN: raise RuntimeError("Не задано TELEGRAM_BOT_TOKEN у Railway Variables")
    load_subs()
    app=Application.builder().token(TOKEN).build()
    for cmd,fn in [("start",start),("status",status),("symbols",symbols),("scan",scan),("help",help_cmd)]: app.add_handler(CommandHandler(cmd,fn))
    if app.job_queue is None: raise RuntimeError("JobQueue недоступний: перевір python-telegram-bot[job-queue]")
    app.job_queue.run_repeating(scheduled,interval=INTERVAL*60,first=30)
    log.info("Starting: %d symbols, interval %d min",len(SYMBOLS),INTERVAL)
    app.run_polling(drop_pending_updates=True)
if __name__=="__main__": main()
