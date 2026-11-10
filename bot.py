import os, json, logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = ''.join(os.getenv('TELEGRAM_BOT_TOKEN', '').split())
SCAN_INTERVAL = max(60, int(os.getenv('SCAN_INTERVAL_SECONDS', '300')))
RR = max(0.5, float(os.getenv('RISK_REWARD', '2.0')))
SUBS_FILE, SENT_FILE = 'subscribers.json', 'sent_signals.json'
SYMBOLS = {'BTCUSD':'BTC-USD','EURUSD':'EURUSD=X','GBPUSD':'GBPUSD=X','EURJPY':'EURJPY=X','AUDUSD':'AUDUSD=X','XAGUSD':'XAGUSD=X','EURAUD':'EURAUD=X','USDCHF':'CHF=X','GBPAUD':'GBPAUD=X'}
TIMEFRAMES = ('H1','H4','D1','W1')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('gold-signals')

def load_json(path, default):
    try:
        with open(path, encoding='utf-8') as f: return json.load(f)
    except (OSError, json.JSONDecodeError): return default

def save_json(path, obj):
    tmp = path+'.tmp'
    with open(tmp,'w',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False)
    os.replace(tmp,path)

def clean(df):
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    cols=['Open','High','Low','Close','Volume']
    if not all(c in df.columns for c in cols): return pd.DataFrame()
    x=df[cols].copy()
    for c in cols: x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna(subset=['Open','High','Low','Close']).sort_index()

def fetch(ticker):
    h=clean(yf.download(ticker,period='60d',interval='1h',progress=False,threads=False,auto_adjust=False))
    d=clean(yf.download(ticker,period='5y',interval='1d',progress=False,threads=False,auto_adjust=False))
    if h.empty or d.empty: raise ValueError('Yahoo Finance повернув порожні дані')
    agg={'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
    return {'H1':h,'H4':h.resample('4h').agg(agg).dropna(),'D1':d,'W1':d.resample('W').agg(agg).dropna()}

def atr(df,n=14):
    pc=df.Close.shift(1)
    tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    v=tr.rolling(n).mean().iloc[-1]
    return float(v) if pd.notna(v) and v>0 else float(df.Close.iloc[-1])*.005

def rsi(s,n=14):
    delta=s.diff(); gain=delta.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+gain/loss.replace(0,np.nan))

def stoch(df,n=14):
    lo=df.Low.rolling(n).min(); hi=df.High.rolling(n).max(); k=100*(df.Close-lo)/(hi-lo).replace(0,np.nan)
    return k,k.rolling(3).mean()

def signal(sym,ticker,tf,strategy,side,df,reason,sl):
    entry=float(df.Close.iloc[-1]); risk=abs(entry-float(sl))
    if not np.isfinite(risk) or risk<=0: return None
    tp=entry+RR*risk if side=='BUY' else entry-RR*risk
    return dict(symbol=sym,ticker=ticker,tf=tf,strategy=strategy,side=side,entry=entry,sl=float(sl),tp=tp,reason=reason,candle=pd.Timestamp(df.index[-1]).isoformat())

def grid(sym,t,tf,d):
    if len(d)<45:return None
    a=atr(d); w=d.iloc[-41:-1]; top=float(w.High.max()); bot=float(w.Low.min()); width=top-bot
    if width<=2*a:return None
    p=(float(d.Close.iloc[-1])-bot)/width
    if p<=.15:return signal(sym,t,tf,'Grid Trading','BUY',d,'Ціна біля нижньої межі 40-свічкового діапазону',bot-.35*a)
    if p>=.85:return signal(sym,t,tf,'Grid Trading','SELL',d,'Ціна біля верхньої межі 40-свічкового діапазону',top+.35*a)

def obos(sym,t,tf,d):
    if len(d)<30:return None
    x=rsi(d.Close); p,c=x.iloc[-2],x.iloc[-1]; a=atr(d)
    if pd.notna(p) and pd.notna(c) and p<30<=c:return signal(sym,t,tf,'Overbought/Oversold','BUY',d,f'RSI повернувся вище 30 ({c:.1f})',float(d.Low.iloc[-6:].min())-.2*a)
    if pd.notna(p) and pd.notna(c) and p>70>=c:return signal(sym,t,tf,'Overbought/Oversold','SELL',d,f'RSI повернувся нижче 70 ({c:.1f})',float(d.High.iloc[-6:].max())+.2*a)

def ppr(sym,t,tf,d):
    if len(d)<35:return None
    x=d.iloc[-3:]; c1,c2,c3=x.iloc[0],x.iloc[1],x.iloc[2]; k,st=stoch(d); a=atr(d)
    sup=float(d.Low.iloc[-25:-3].min()); res=float(d.High.iloc[-25:-3].max())
    bull=c2.Low<c1.Low and c3.Close>c2.High and c3.Close>c3.Open and c2.Low<=sup+.35*a
    bear=c2.High>c1.High and c3.Close<c2.Low and c3.Close<c3.Open and c2.High>=res-.35*a
    k0,d0,k1,d1=k.iloc[-2],st.iloc[-2],k.iloc[-1],st.iloc[-1]
    if not all(pd.notna(v) for v in (k0,d0,k1,d1)):return None
    if bull and k0<=d0 and k1>d1 and k1<35:return signal(sym,t,tf,'PPR + Stochastic','BUY',d,'Бичачий 3-свічковий PPR біля підтримки + перетин Stochastic внизу',float(x.Low.min())-.2*a)
    if bear and k0>=d0 and k1<d1 and k1>65:return signal(sym,t,tf,'PPR + Stochastic','SELL',d,'Ведмежий 3-свічковий PPR біля опору + перетин Stochastic вгорі',float(x.High.max())+.2*a)

def fibonacci(sym,t,tf,d):
    if len(d)<210:return None
    e50=d.Close.ewm(span=50,adjust=False).mean().iloc[-1]; e200=d.Close.ewm(span=200,adjust=False).mean().iloc[-1]; a=atr(d); w=d.iloc[-61:-1]
    hi_i=w.High.idxmax(); lo_i=w.Low.idxmin(); hi=float(w.High.max()); lo=float(w.Low.min()); last=d.iloc[-1]
    if hi<=lo:return None
    span=hi-lo
    if e50>e200 and lo_i<hi_i:
        f1,f2=hi-.618*span,hi-.382*span
        if last.Low<=f2 and last.Low>=f1-.25*a and last.Close>last.Open and last.Close>f1:
            return signal(sym,t,tf,'Американські гірки (Fibonacci)','BUY',d,'Висхідний тренд; корекція в зону 38.2–61.8% і бичача реакція',min(float(last.Low),f1)-.25*a)
    if e50<e200 and hi_i<lo_i:
        f1,f2=lo+.382*span,lo+.618*span
        if last.High>=f1 and last.High<=f2+.25*a and last.Close<last.Open and last.Close<f2:
            return signal(sym,t,tf,'Американські гірки (Fibonacci)','SELL',d,'Низхідний тренд; корекція в зону 38.2–61.8% і ведмежа реакція',max(float(last.High),f2)+.25*a)

def triangle(sym,t,tf,d):
    if len(d)<35:return None
    w=d.iloc[-21:-1]; x=np.arange(len(w)); us,ui=np.polyfit(x,w.High.to_numpy(float),1); ls,li=np.polyfit(x,w.Low.to_numpy(float),1)
    if not (us<0 and ls>0):return None
    up=ui+us*len(w); low=li+ls*len(w); last=d.iloc[-1]; a=atr(d)
    if last.Close>up and last.Close>last.Open:return signal(sym,t,tf,'Вихід із трикутника','BUY',d,'Закриття свічки вище верхньої спадної межі трикутника',min(float(last.Low),up)-.3*a)
    if last.Close<low and last.Close<last.Open:return signal(sym,t,tf,'Вихід із трикутника','SELL',d,'Закриття свічки нижче нижньої висхідної межі трикутника',max(float(last.High),low)+.3*a)

STRATEGIES=(grid,obos,ppr,fibonacci,triangle)

def scan_symbol(sym,ticker):
    frames=fetch(ticker); found=[]; errors=[]
    for tf in ('H1','H4','D1','W1'):
        d=frames[tf].iloc[:-1].copy()
        if len(d)<30:errors.append(f'{tf}: мало свічок');continue
        for fn in STRATEGIES:
            try:
                s=fn(sym,ticker,tf,d)
                if s:found.append(s)
            except Exception as e:
                log.exception('%s %s %s',sym,tf,fn.__name__);errors.append(f'{tf}/{fn.__name__}: {type(e).__name__}')
    return found,errors

def format_signal(s):
    ts=pd.Timestamp(s['candle']); ts=ts.tz_localize('UTC') if ts.tzinfo is None else ts; ts=ts.tz_convert('UTC')
    return (f"📊 SIGNAL — {s['symbol']} ({s['ticker']})\nТаймфрейм: {s['tf']}\nНапрямок: {s['side']}\nСтратегія: {s['strategy']}\nEntry: {s['entry']:.6g}\nSL: {s['sl']:.6g}\nTP: {s['tp']:.6g}\nОбґрунтування: {s['reason']}\nЗакрита свічка: {ts:%Y-%m-%d %H:%M UTC}\n⚠️ Інформаційний сигнал. Перевір спред, новини та ризик; прибуток не гарантований.")

async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    subs=load_json(SUBS_FILE,[]); cid=update.effective_chat.id
    if cid not in subs:subs.append(cid);save_json(SUBS_FILE,subs)
    await update.message.reply_text('Бот активний. Smart Money вимкнено. Стратегії: Grid Trading, Overbought/Oversold, PPR + Stochastic, Американські гірки (Fibonacci), Вихід із трикутника. Команди: /status /symbols /scan /help')

async def status(update:Update,context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f'Бот запущений. Інструментів: {len(SYMBOLS)}. Стратегій: 5 (без Smart Money). Таймфрейми: H1, H4, D1, W1. Інтервал: {SCAN_INTERVAL} с. Одна ціль TP; стандартне співвідношення ризику до прибутку 1:{RR:g}.')

async def symbols(update:Update,context:ContextTypes.DEFAULT_TYPE): await update.message.reply_text('Інструменти: '+', '.join(SYMBOLS))
async def help_cmd(update:Update,context:ContextTypes.DEFAULT_TYPE): await update.message.reply_text('/start — підписка\n/status — стан\n/symbols — інструменти\n/scan — сканувати зараз\n/help — допомога')

async def do_scan(app,targets=None):
    targets=targets if targets is not None else load_json(SUBS_FILE,[]); sent=load_json(SENT_FILE,{}); count=0; errs=0
    for sym,ticker in SYMBOLS.items():
        try:
            signals,errors=scan_symbol(sym,ticker);errs+=len(errors)
            for s in signals:
                key='|'.join([s['symbol'],s['tf'],s['strategy'],s['side'],s['candle']])
                if key in sent:continue
                for cid in targets:
                    try:await app.bot.send_message(chat_id=int(cid),text=format_signal(s))
                    except Exception:log.exception('Не вдалося надіслати chat_id=%s',cid)
                sent[key]=datetime.now(timezone.utc).isoformat();count+=1
        except Exception:errs+=1;log.exception('Помилка інструмента %s',sym)
    if len(sent)>2000:sent=dict(list(sent.items())[-1500:])
    save_json(SENT_FILE,sent);log.info('Сканування завершено: сигналів=%d, помилок=%d',count,errs)
    return count,errs

async def scan_cmd(update:Update,context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Сканую 9 інструментів на H1, H4, D1, W1. Це може тривати кілька хвилин.')
    n,e=await do_scan(context.application,[update.effective_chat.id])
    await update.message.reply_text(f'Перевірку завершено. Нових сигналів: {n}. Помилок даних: {e}. Деталі — Railway Logs.')

async def scheduled(context:ContextTypes.DEFAULT_TYPE): await do_scan(context.application)
async def post_init(app):
    if app.job_queue:app.job_queue.run_repeating(scheduled,interval=SCAN_INTERVAL,first=20);log.info('Планувальник запущено: %s секунд',SCAN_INTERVAL)

def main():
    if not TOKEN:raise RuntimeError('Не задано TELEGRAM_BOT_TOKEN')
    app=Application.builder().token(TOKEN).post_init(post_init).build()
    for name,fn in [('start',start),('status',status),('symbols',symbols),('help',help_cmd),('scan',scan_cmd)]:app.add_handler(CommandHandler(name,fn))
    log.info('Запуск: Smart Money вимкнено; активні стратегії: Grid, OB/OS, PPR+Stochastic, Fibonacci, Triangle')
    app.run_polling(drop_pending_updates=True)

if __name__=='__main__':main()
