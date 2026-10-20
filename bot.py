import os, json, logging, asyncio
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO'), format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('signal-bot')
TOKEN = ''.join(os.getenv('TELEGRAM_BOT_TOKEN','').split())
SCAN_INTERVAL = max(60, int(os.getenv('SCAN_INTERVAL','300')))
MIN_SCORE = max(1, min(2, int(os.getenv('MIN_CONFIRMATIONS','1'))))
SUBSCRIBERS_FILE = Path('subscribers.json')
ALERTS_FILE = Path('alerts.json')
SYMBOLS = {
    'BTCUSD': 'BTC-USD', 'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X',
    'EURJPY': 'EURJPY=X', 'AUDUSD': 'AUDUSD=X', 'XAGUSD': 'XAGUSD=X',
    'EURAUD': 'EURAUD=X', 'USDCHF': 'USDCHF=X', 'GBPAUD': 'GBPAUD=X'
}
PERIODS = {'H1': ('60m', '730d'), 'H4': ('60m', '730d'), 'D1': ('1d', 'max'), 'W1': ('1wk', 'max')}
last_alerts = set()


def read_json(path, default):
    try:
        if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    except Exception: log.exception('Could not read %s', path)
    return default

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0).rolling(n).mean(); down=(-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100/(1 + up/down.replace(0, np.nan))
def atr(df, n=14):
    pc=df['Close'].shift(1)
    tr=pd.concat([df['High']-df['Low'], (df['High']-pc).abs(), (df['Low']-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def pivots(df, left=3, right=3):
    h=df['High'].to_numpy(); l=df['Low'].to_numpy(); ph=np.full(len(df), np.nan); pl=np.full(len(df), np.nan)
    for i in range(left, len(df)-right):
        if h[i] == np.max(h[i-left:i+right+1]): ph[i]=h[i]
        if l[i] == np.min(l[i-left:i+right+1]): pl[i]=l[i]
    return pd.Series(ph, index=df.index), pd.Series(pl, index=df.index)
def last_valid(s):
    x=s.dropna(); return float(x.iloc[-1]) if len(x) else np.nan

def fetch_symbol(ticker):
    # Yahoo Finance supports 60m bars for a limited historical window. Resample to H4 locally.
    raw = yf.download(ticker, period='730d', interval='60m', auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty: raise RuntimeError('Yahoo Finance returned no intraday candles')
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    raw = raw[['Open','High','Low','Close','Volume']].dropna().sort_index()
    raw = raw[~raw.index.duplicated(keep='last')]
    # Exclude latest potentially incomplete H1 candle; resample OHLCV into H4.
    now = pd.Timestamp.now(tz='UTC')
    idx = pd.to_datetime(raw.index, utc=True)
    raw.index = idx
    if len(raw) and (now - raw.index[-1]).total_seconds() < 3600: raw = raw.iloc[:-1]
    h1 = raw.copy()
    h4 = h1.resample('4h', label='left', closed='left').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    # Keep only completed H4 bars.
    if len(h4) and (now - h4.index[-1]).total_seconds() < 4*3600: h4 = h4.iloc[:-1]
    d1 = yf.download(ticker, period='max', interval='1d', auto_adjust=False, progress=False, threads=False)
    w1 = yf.download(ticker, period='max', interval='1wk', auto_adjust=False, progress=False, threads=False)
    def clean(df):
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df[['Open','High','Low','Close','Volume']].dropna().sort_index()
    d1, w1 = clean(d1), clean(w1)
    # Daily/weekly provider may include current candle; exclude if still in progress.
    if len(d1) and (now - pd.to_datetime(d1.index[-1], utc=True)).total_seconds() < 86400: d1=d1.iloc[:-1]
    if len(w1) and (now - pd.to_datetime(w1.index[-1], utc=True)).total_seconds() < 7*86400: w1=w1.iloc[:-1]
    frames={'H1':h1, 'H4':h4, 'D1':d1, 'W1':w1}
    for tf, df in frames.items():
        if len(df) < 205: raise RuntimeError(f'Insufficient {tf} candles: {len(df)}')
    return frames

def fvg_status(df, lookback=120):
    """Return the most recent three-candle FVG and whether later candles filled it.

    Bullish gap: High[i-2] < Low[i]. Its lower edge is High[i-2].
    Bearish gap: Low[i-2] > High[i]. Its upper edge is Low[i-2].
    A gap is fully filled only if a later candle trades through its far edge.
    """
    if len(df) < 3:
        return 'no recent FVG', None
    first = max(2, len(df) - lookback)
    # Search newest to oldest, but validate fill using candles after the gap candle.
    for i in range(len(df) - 1, first - 1, -1):
        a, gap_candle = df.iloc[i-2], df.iloc[i]
        later = df.iloc[i+1:]
        if float(gap_candle['Low']) > float(a['High']):
            lo, hi = float(a['High']), float(gap_candle['Low'])
            if len(later) and (later['Low'] <= lo).any():
                status = 'bullish FVG fully filled'
            elif len(later) and (later['Low'] < hi).any():
                status = 'bullish FVG partially filled'
            else:
                status = 'bullish FVG open'
            return status, {'direction': 'BUY', 'low': lo, 'high': hi, 'filled': status.endswith('fully filled')}
        if float(gap_candle['High']) < float(a['Low']):
            lo, hi = float(gap_candle['High']), float(a['Low'])
            if len(later) and (later['High'] >= hi).any():
                status = 'bearish FVG fully filled'
            elif len(later) and (later['High'] > lo).any():
                status = 'bearish FVG partially filled'
            else:
                status = 'bearish FVG open'
            return status, {'direction': 'SELL', 'low': lo, 'high': hi, 'filled': status.endswith('fully filled')}
    return 'no recent FVG', None

def trendline_break_retest(df, atr_value):
    """Signal only after a candle closes beyond a pivot trendline and a later candle retests it.

    Uses confirmed pivots, a fitted line through recent swing points, and the last
    closed candle as the retest-response candle. This is a transparent heuristic.
    """
    if len(df) < 40 or not np.isfinite(atr_value) or atr_value <= 0:
        return 'NONE', 'not enough candles/ATR'
    highs, lows = pivots(df, left=3, right=3)
    n = len(df)
    tolerance = max(atr_value * 0.25, abs(float(df['Close'].iloc[-1])) * 0.0002)
    # Inspect recent confirmed swing points, excluding the latest 3 bars because pivots need right-side confirmation.
    high_pts = [(i, float(highs.iloc[i])) for i in range(max(0, n-100), n-8) if np.isfinite(highs.iloc[i])]
    low_pts = [(i, float(lows.iloc[i])) for i in range(max(0, n-100), n-8) if np.isfinite(lows.iloc[i])]
    close = df['Close'].astype(float).to_numpy()
    hi = df['High'].astype(float).to_numpy()
    lo = df['Low'].astype(float).to_numpy()

    def check(points, want):
        # want='BUY': descending swing-high resistance; want='SELL': ascending swing-low support.
        if len(points) < 2:
            return False
        pts = points[-4:]
        xs = np.array([x for x, _ in pts], dtype=float)
        ys = np.array([y for _, y in pts], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        if want == 'BUY' and slope >= 0: return False
        if want == 'SELL' and slope <= 0: return False
        # Find a genuine close beyond the line, then a later retest; the latest candle must reject the line.
        line = slope * np.arange(n) + intercept
        for br in range(n-2, max(int(xs[-1])+1, n-13), -1):
            if br < 1: continue
            broke = close[br] > line[br] + tolerance*0.15 if want == 'BUY' else close[br] < line[br] - tolerance*0.15
            if not broke: continue
            for rt in range(br+1, n):
                touched = lo[rt] <= line[rt] + tolerance if want == 'BUY' else hi[rt] >= line[rt] - tolerance
                held = close[rt] > line[rt] if want == 'BUY' else close[rt] < line[rt]
                if touched and held and rt == n-1:
                    return True
        return False

    if check(high_pts, 'BUY'):
        return 'BUY', 'close above descending trendline + retest held'
    if check(low_pts, 'SELL'):
        return 'SELL', 'close below ascending trendline + retest rejected'
    return 'NONE', 'no confirmed close-and-retest setup'


def analyze(frames):
    # Only two strategies are enabled: Smart Money and trendline break + retest.
    h1_raw = frames['H1']
    bar_time = pd.Timestamp(h1_raw.index[-1]).isoformat()
    h1, h4, d1, w1 = (frames[k].reset_index(drop=True) for k in ('H1', 'H4', 'D1', 'W1'))
    close = h1['Close'].astype(float)
    c = float(close.iloc[-1])
    av = float(atr(h1).iloc[-1])
    if not np.isfinite(av) or av <= 0:
        av = float((h1['High'] - h1['Low']).tail(20).mean())
    sig = {'Smart Money': 'NONE', 'Пробій трендової + ретест': 'NONE'}
    details = {'Smart Money': 'no confirmed setup', 'Пробій трендової + ретест': ''}

    # Smart Money proxy: confirmed swing break (BOS/CHOCH), liquidity sweep/reclaim,
    # and an unfilled FVG in the same direction. An opposing open FVG is not itself a signal.
    phs, pls = pivots(h1, left=3, right=3)
    valid_hi = [(i, float(v)) for i, v in enumerate(phs) if np.isfinite(v) and i < len(h1)-3]
    valid_lo = [(i, float(v)) for i, v in enumerate(pls) if np.isfinite(v) and i < len(h1)-3]
    p_hi = valid_hi[-1][1] if valid_hi else np.nan
    p_lo = valid_lo[-1][1] if valid_lo else np.nan
    fvg, gap = fvg_status(h1)
    prev = h1.iloc[-2]
    recent = h1.iloc[-21:-1]
    swept_low = len(recent) and float(prev['Low']) < float(recent['Low'].min()) and c > float(prev['Low'])
    swept_high = len(recent) and float(prev['High']) > float(recent['High'].max()) and c < float(prev['High'])
    bos_buy = np.isfinite(p_hi) and c > p_hi
    bos_sell = np.isfinite(p_lo) and c < p_lo
    gap_buy = bool(gap and gap['direction'] == 'BUY' and not gap['filled'])
    gap_sell = bool(gap and gap['direction'] == 'SELL' and not gap['filled'])
    # Require structure/liquidity evidence and same-direction FVG context to reduce arbitrary signals.
    if bos_buy and (gap_buy or swept_low):
        sig['Smart Money'] = 'BUY'
        details['Smart Money'] = 'BOS bullish' + (' + bullish FVG open' if gap_buy else ' + sell-side liquidity sweep/reclaim')
    elif bos_sell and (gap_sell or swept_high):
        sig['Smart Money'] = 'SELL'
        details['Smart Money'] = 'BOS bearish' + (' + bearish FVG open' if gap_sell else ' + buy-side liquidity sweep/reclaim')
    elif swept_low and gap_buy:
        sig['Smart Money'] = 'BUY'
        details['Smart Money'] = 'sell-side liquidity sweep/reclaim + bullish FVG open (CHOCH proxy)'
    elif swept_high and gap_sell:
        sig['Smart Money'] = 'SELL'
        details['Smart Money'] = 'buy-side liquidity sweep/reclaim + bearish FVG open (CHOCH proxy)'
    else:
        details['Smart Money'] = 'no matching BOS/CHOCH + liquidity/FVG setup'

    tl_sig, tl_detail = trendline_break_retest(h1, av)
    sig['Пробій трендової + ретест'] = tl_sig
    details['Пробій трендової + ретест'] = tl_detail

    trend = []
    for df in (h4, d1, w1):
        x = df['Close'].astype(float)
        e50x, e200x = ema(x, 50).iloc[-1], ema(x, 200).iloc[-1]
        trend.append('BUY' if x.iloc[-1] > e50x > e200x else 'SELL' if x.iloc[-1] < e50x < e200x else 'NONE')
    return sig, c, av, fvg, trend, p_hi, p_lo, bar_time, details

def price_precision(symbol):
    if symbol.startswith('BTC'): return 2
    if 'JPY' in symbol: return 3
    if symbol=='XAGUSD': return 3
    return 5

def generate_messages(symbol, ticker, frames):
    sig, entry, av, fvg, trend, ph, pl, bar, details = analyze(frames)
    active = {k:v for k,v in sig.items() if v in ('BUY','SELL')}
    if not active: return []
    bydir = {'BUY':[k for k,v in active.items() if v=='BUY'], 'SELL':[k for k,v in active.items() if v=='SELL']}
    out=[]; p=price_precision(symbol)
    for direction, names in bydir.items():
        if not names or len(names) < MIN_SCORE: continue
        risk=max(1.5*av, abs(entry)*0.0005)
        sl,tp1,tp2=(entry-risk,entry+risk,entry+2*risk) if direction=='BUY' else (entry+risk,entry-risk,entry-2*risk)
        key=f'{symbol}|{direction}|{bar}|{",".join(sorted(names))}'
        if key in last_alerts: continue
        opposite='SELL' if direction=='BUY' else 'BUY'
        conflict=(f'⚠️ Протилежний напрямок {opposite}: {", ".join(bydir[opposite])}. Сигнали незалежні; не змішувати.\n' if bydir[opposite] else '')
        why='\n'.join(f'• {name}: {details.get(name, "")}' for name in names)
        msg=(f'📊 SIGNAL — {symbol} ({ticker})\nНапрямок: {direction}\nСтратегії: {", ".join(names)}\n'
             f'Entry: {entry:.{p}f}\nSL: {sl:.{p}f}\nTP1: {tp1:.{p}f}\nTP2: {tp2:.{p}f}\n'
             f'Підтвердження: {len(names)}/2 активні стратегії\nОбґрунтування:\n{why}\n'
             f'FVG: {fvg}\nBOS/CHOCH: pivot-break proxy; high={ph:.5g}, low={pl:.5g}\n'
             f'Тренд H4/D1/W1: {trend}\n{conflict}Закрита H1-свічка (UTC): {bar}\n'
             '⚠️ Сигнал інформаційний. Перевір спред, новини та ризик; прибуток не гарантований.')
        out.append((key,msg))
    return out

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subs=read_json(SUBSCRIBERS_FILE,[]); uid=update.effective_chat.id
    if uid not in subs: subs.append(uid); write_json(SUBSCRIBERS_FILE,subs)
    await update.message.reply_text('Бот активний. Стратегії: Smart Money та пробій трендової з ретестом. Дані: Yahoo Finance, не MT5. Команди: /status /symbols /scan')
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f'Бот запущений. Інструментів: {len(SYMBOLS)}. Стратегії: Smart Money + пробій трендової з ретестом. Інтервал: {SCAN_INTERVAL} с. Мінімум стратегій в одному напрямку: {MIN_SCORE}/2.')
async def cmd_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Інструменти: '+', '.join(SYMBOLS.keys()))
async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Запускаю перевірку 9 інструментів. Це може тривати до хвилини.')
    result = await scan_once(context.application, force_chat=update.effective_chat.id)
    await update.message.reply_text(f"Перевірку завершено. Сигнальних повідомлень: {result['signals']}. Помилок даних: {result['errors']}. Якщо сигналів немає — поточні умови не спрацювали; подробиці дивись у Railway Logs.")

async def scan_once(app, force_chat=None):
    subs=read_json(SUBSCRIBERS_FILE,[])
    if force_chat and force_chat not in subs: subs.append(force_chat); write_json(SUBSCRIBERS_FILE,subs)
    stats = {'signals': 0, 'errors': 0}
    for symbol,ticker in SYMBOLS.items():
        try:
            frames=await asyncio.to_thread(fetch_symbol,ticker)
            symbol_signal_count = 0
            for key,msg in generate_messages(symbol,ticker,frames):
                # Send to all registered chats; store dedup key only after at least one send succeeds.
                sent=False
                for chat in subs:
                    try: await app.bot.send_message(chat_id=chat,text=msg); sent=True
                    except Exception: log.exception('Telegram send failed for chat %s',chat)
                if sent:
                    last_alerts.add(key)
                    stats['signals'] += 1
                    symbol_signal_count += 1
            if symbol_signal_count == 0:
                log.info('No new signal sent for %s (no setup or duplicate alert)', symbol)
        except Exception as e:
            stats['errors'] += 1
            log.exception('Scan failed for %s (%s): %s',symbol,ticker,e)
    return stats

async def periodic(context: ContextTypes.DEFAULT_TYPE):
    await scan_once(context.application)

def main():
    if not TOKEN: raise SystemExit('Set TELEGRAM_BOT_TOKEN in Railway Variables')
    app=Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start',cmd_start)); app.add_handler(CommandHandler('status',cmd_status))
    app.add_handler(CommandHandler('symbols',cmd_symbols)); app.add_handler(CommandHandler('scan',cmd_scan))
    app.job_queue.run_repeating(periodic, interval=SCAN_INTERVAL, first=15)
    log.info('Starting signal bot; enabled_strategies=Smart Money,Trendline Break+Retest symbols=%s interval=%ss min confirmations=%s',list(SYMBOLS),SCAN_INTERVAL,MIN_SCORE)
    app.run_polling(drop_pending_updates=True)
if __name__=='__main__': main()
