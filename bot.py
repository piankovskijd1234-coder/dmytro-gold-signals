import os, json, logging, urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta
import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN="".join(os.getenv("TELEGRAM_BOT_TOKEN","").split())
INTERVAL=max(5,int(os.getenv("SCAN_INTERVAL_MINUTES","15")))
RR=max(0.5,float(os.getenv("RISK_REWARD","2.0")))
SUB_FILE="subscribers.json"
SYMBOLS={
 "BTCUSD":"BTC-USD","EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X",
 "EURJPY":"EURJPY=X","AUDUSD":"AUDUSD=X","XAGUSD":"XAGUSD=X",
 "EURAUD":"EURAUD=X","USDCHF":"CHF=X","GBPAUD":"GBPAUD=X"
}
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log=logging.getLogger("dmytro-gold-signals")
subs=set()
sent=set()

def load_subs():
 global subs
 try:
  with open(SUB_FILE,encoding="utf-8") as f: subs={int(x) for x in json.load(f)}
 except Exception: subs=set()
def save_subs():
 with open(SUB_FILE,"w",encoding="utf-8") as f: json.dump(sorted(subs),f)

def get_yahoo(ticker,period,interval):
 d=yf.download(ticker,period=period,interval=interval,progress=False,auto_adjust=False,threads=False)
 if d is None or d.empty: return pd.DataFrame()
 if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
 d.columns=[str(x).title() for x in d.columns]
 cols=["Open","High","Low","Close","Volume"]
 return d[cols].dropna() if all(x in d.columns for x in cols) else pd.DataFrame()

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
def trend(d):
 if len(d)<205:return 0
 e50,e200=ema(d.Close,50).iloc[-1],ema(d.Close,200).iloc[-1]
 c=d.Close.iloc[-1]
 return 1 if c>e50>e200 else (-1 if c<e50<e200 else 0)

def binance_real_delta():
 """Approximate aggressor delta from Binance public aggregate trades over last 60 min.
 Returns None if endpoint unavailable; not used for non-BTC instruments."""
 try:
  end=int(datetime.now(timezone.utc).timestamp()*1000)
  start=end-60*60*1000
  url="https://api.binance.com/api/v3/aggTrades?"+urllib.parse.urlencode(
      {"symbol":"BTCUSDT","startTime":start,"endTime":end,"limit":1000})
  req=urllib.request.Request(url,headers={"User-Agent":"DmytroGoldSignals/1.0"})
  with urllib.request.urlopen(req,timeout=10) as resp: rows=json.loads(resp.read().decode())
  buy=sell=0.0
  for row in rows:
   qty=float(row["q"])
   # m=True means buyer was maker, so seller was aggressor.
   if row.get("m",False): sell+=qty
   else: buy+=qty
  total=buy+sell
  if total<=0:return None
  return (buy-sell)/total, "BINANCE_AGGTRADES"
 except Exception:
  log.exception("Binance real delta unavailable")
  return None

def candle_delta(df,bars=5):
 d=df.iloc[-bars:]
 signed=0.0; total=0.0
 for _,r in d.iterrows():
  v=float(r.Volume) if pd.notna(r.Volume) else 0.0
  total+=v
  if r.Close>r.Open:signed+=v
  elif r.Close<r.Open:signed-=v
 if total<=0:return None
 return signed/total

def analyze(symbol,ticker):
 # H1 entry; H4 and D1 trend filter. Use closed candles only.
 h1=get_yahoo(ticker,"60d","1h")
 h4=get_yahoo(ticker,"730d","4h")
 d1=get_yahoo(ticker,"5y","1d")
 if min(len(h1),len(h4),len(d1))<205:
  raise ValueError("Недостатньо історичних даних H1/H4/D1")
 b4,b1=trend(h4),trend(d1)
 if not b4 or b4!=b1:return None
 h=h1.iloc[:-1].copy()
 h["E20"]=ema(h.Close,20);h["E50"]=ema(h.Close,50);h["ATR"]=atr(h);h["RSI"]=rsi(h.Close)
 h=h.dropna()
 if len(h)<50:return None
 c=h.iloc[-1]; prior=h.iloc[-21:-1]; recent=h.iloc[-6:-1]
 resistance=float(prior.High.max());support=float(prior.Low.min())
 a=float(c.ATR);close=float(c.Close);rv=float(c.RSI)
 if not a or pd.isna(a):return None
 direction=b4
 # BOS occurred in one of the previous five completed bars; current bar retests.
 if direction==1:
  bos=bool((recent.Close>resistance).any())
  retest=float(c.Low)<=resistance+0.25*a and close>resistance
  candle=close>float(c.Open) and close>float(c.E20)>float(c.E50) and 50<=rv<=72
  if not (bos and retest and candle):return None
  entry=close;sl=min(float(c.Low),resistance-0.25*a);risk=entry-sl
  if risk<=0:return None
  tp=entry+RR*risk;side="BUY"
 else:
  bos=bool((recent.Close<support).any())
  retest=float(c.High)>=support-0.25*a and close<support
  candle=close<float(c.Open) and close<float(c.E20)<float(c.E50) and 28<=rv<=50
  if not (bos and retest and candle):return None
  entry=close;sl=max(float(c.High),support+0.25*a);risk=sl-entry
  if risk<=0:return None
  tp=entry-RR*risk;side="SELL"
 # Order flow confirmation: BTC uses real Binance aggTrades delta where available.
 # Other symbols use signed candle-volume estimate; label clearly.
 if symbol=="BTCUSD":
  result=binance_real_delta()
  if result is not None: delta,source=result
  else:
   delta=candle_delta(h);source="ESTIMATED_CANDLE_VOLUME (Binance unavailable)"
 else:
  delta=candle_delta(h);source="ESTIMATED_CANDLE_VOLUME (not true Bid/Ask delta)"
 if delta is None:return None
 if (side=="BUY" and delta<=0) or (side=="SELL" and delta>=0):return None
 ts=h.index[-1]
 if getattr(ts,"tzinfo",None) is None:ts=ts.replace(tzinfo=timezone.utc)
 ts=ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
 return {"symbol":symbol,"side":side,"entry":entry,"sl":sl,"tp":tp,
         "rr":RR,"rsi":rv,"delta":delta,"delta_source":source,
         "time":ts,"bias":"BUY" if direction==1 else "SELL"}

def fmt(x):return f"{x:.2f}" if abs(x)>=1000 else (f"{x:.4f}" if abs(x)>=10 else f"{x:.5f}")
def message(s):
 return (f"📊 {s['side']} — {s['symbol']}\\nСтратегія: Order Flow + Trend Pullback + BOS + Retest\\n"
 f"Вхід: H1 | Тренд H4/D1: {s['bias']}\\nEntry: {fmt(s['entry'])}\\nSL: {fmt(s['sl'])}\\nTP: {fmt(s['tp'])}\\n"
 f"RR: 1:{s['rr']:.1f}\\nRSI: {s['rsi']:.1f}\\nDelta: {s['delta']:+.3f}\\n"
 f"Джерело Delta: {s['delta_source']}\\nЗакрита H1-свічка: {s['time']}\\n"
 "Умови: тренд H4/D1, BOS, ретест, свічкове підтвердження та позитивна/негативна Delta.\\n"
 "⚠️ Для більшості Forex/металів Delta приблизна. Навіть реальна Binance Delta тут рахується за доступними aggTrades, не є гарантією. Перевір котирування брокера та ризик.")

async def start(u,c):
 subs.add(u.effective_chat.id);save_subs()
 await u.message.reply_text("Бот активний. Єдина система: Order Flow + Trend Pullback + BOS + Retest. /status /symbols /scan")
async def status(u,c):
 await u.message.reply_text(f"Інструментів: 9. Стратегія: Order Flow + Trend Pullback + BOS + Retest. Вхід H1, тренд H4/D1. Один TP, RR 1:{RR:g}. Автосканування кожні {INTERVAL} хв.")
async def symbols(u,c):await u.message.reply_text("Інструменти: "+", ".join(SYMBOLS))
async def scan_chat(chat,ctx):
 n=e=0
 for sym,ticker in SYMBOLS.items():
  try:
   s=analyze(sym,ticker)
   if s:
    key=f"{sym}:{s['side']}:{s['time']}"
    if key not in sent:
     await ctx.bot.send_message(chat_id=chat,text=message(s));sent.add(key);n+=1
  except Exception:
   e+=1;log.exception("Scan error %s",sym)
 return n,e
async def scan(u,c):
 await u.message.reply_text("Сканую 9 інструментів. Зачекай кілька хвилин.")
 n,e=await scan_chat(u.effective_chat.id,c)
 await u.message.reply_text(f"Сканування завершено. Нових сигналів: {n}. Помилок даних: {e}. Деталі — Railway Logs.")
async def scheduled(c):
 for chat in list(subs):
  try:await scan_chat(chat,c)
  except Exception:log.exception("scheduled scan failed")
async def help_cmd(u,c):await u.message.reply_text("/start — підписка\\n/status — налаштування\\n/symbols — інструменти\\n/scan — ручне сканування")
def main():
 if not TOKEN:raise RuntimeError("Не задано TELEGRAM_BOT_TOKEN у Railway Variables")
 load_subs()
 app=Application.builder().token(TOKEN).build()
 for name,fn in [("start",start),("status",status),("symbols",symbols),("scan",scan),("help",help_cmd)]:app.add_handler(CommandHandler(name,fn))
 if app.job_queue is None:raise RuntimeError("JobQueue недоступний; перевір requirements.txt")
 app.job_queue.run_repeating(scheduled,interval=INTERVAL*60,first=30)
 log.info("Bot started: %d symbols; OrderFlow + BOS + Retest",len(SYMBOLS))
 app.run_polling(drop_pending_updates=True)
if __name__=="__main__":main()
