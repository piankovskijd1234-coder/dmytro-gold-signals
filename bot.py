import os
import json
import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = "".join(os.getenv("TELEGRAM_BOT_TOKEN", "").split())
INTERVAL = max(5, int(os.getenv("SCAN_INTERVAL_MINUTES", "5")))  # New-entry scans; active trades are checked separately every minute.
SUBSCRIBERS_FILE = "subscribers.json"
ALERTS_FILE = "alerts.json"
ACTIVE_FILE = "active_signals.json"
SYMBOLS = {
    "BTC/USD": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "EUR/JPY": "EURJPY=X",
    "AUD/USD": "AUDUSD=X",
    "XAG/USD": "XAGUSD=X",
    "EUR/AUD": "EURAUD=X",
    "USD/CHF": "CHF=X",
    "XAU/USD": "XAUUSD=X",
}
FALLBACK_TICKERS = {
    "XAU/USD": ["GC=F"],
    "XAG/USD": ["SI=F"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmytro-gold-signals")
SUBSCRIBERS = set()
# Prevent duplicate alert for the same symbol, side and completed H1 candle.
ALERTED = set()
ACTIVE_SIGNALS = {}

def load_alerts():
    global ALERTED
    try:
        with open(ALERTS_FILE, "r", encoding="utf-8") as f:
            ALERTED = set(json.load(f))
    except (OSError, ValueError, TypeError):
        ALERTED = set()
ACTIVE_SIGNALS = {}

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



def load_active_signals():
    global ACTIVE_SIGNALS
    try:
        with open(ACTIVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ACTIVE_SIGNALS = data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        ACTIVE_SIGNALS = {}


def save_active_signals():
    try:
        with open(ACTIVE_FILE, "w", encoding="utf-8") as f:
            json.dump(ACTIVE_SIGNALS, f, ensure_ascii=False)
    except OSError:
        log.exception("Cannot save active signal history")



def check_active_signal(symbol, h1):
    """Check only newly closed H1 candles. If TP and SL are both inside one candle,
    report ambiguity rather than falsely claiming a stop or target was first."""
    item = ACTIVE_SIGNALS.get(symbol)
    if not item or h1.empty:
        return None
    try:
        ts = pd.Timestamp(item.get("last_checked") or item.get("h1"))
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except Exception:
        ts = h1.index[-2] if len(h1) > 1 else h1.index[-1]
    # Exclude the currently forming candle; decisions use closed H1 bars only.
    closed = h1.iloc[:-1]
    bars = closed[closed.index > ts]
    if bars.empty:
        return "активний сигнал; нової закритої H1-свічки ще немає"

    events = []
    side = item["side"]
    entry, sl = float(item["entry"]), float(item["sl"])
    tp1, tp2 = float(item["tp1"]), float(item["tp2"])
    risk = abs(entry - sl)
    zone_low = float(item.get("zone_low", entry - risk * 0.15))
    zone_high = float(item.get("zone_high", entry + risk * 0.15))

    for timestamp, candle in bars.iterrows():
        high, low = float(candle.high), float(candle.low)
        op, cl = float(candle.open), float(candle.close)
        hit_sl = low <= sl if side == "BUY" else high >= sl
        hit_tp1 = high >= tp1 if side == "BUY" else low <= tp1
        hit_tp2 = high >= tp2 if side == "BUY" else low <= tp2

        # OHLC candles don't reveal the intrabar order. Never invent which level came first.
        if hit_sl and (hit_tp1 or hit_tp2):
            events.append(("AMBIGUOUS", timestamp))
            item["last_checked"] = str(timestamp)
            item["ambiguous_candle"] = str(timestamp)
            continue
        if hit_sl:
            events.append(("SL", timestamp))
            ACTIVE_SIGNALS.pop(symbol, None)
            break
        if hit_tp1 and not item.get("tp1_notified"):
            events.append(("TP1", timestamp))
            item["tp1_notified"] = True
        if hit_tp2:
            if not item.get("tp1_notified"):
                events.append(("TP1", timestamp))
                item["tp1_notified"] = True
            events.append(("TP2", timestamp))
            ACTIVE_SIGNALS.pop(symbol, None)
            break

        touched_entry_zone = low <= zone_high and high >= zone_low
        if touched_entry_zone and not item.get("entry_retest_notified"):
            direction_close = cl > op if side == "BUY" else cl < op
            closes_in_zone = zone_low <= cl <= zone_high
            if direction_close and closes_in_zone:
                events.append(("ENTRY_CONFIRMED", timestamp))
                item["entry_retest_notified"] = True
            elif (side == "BUY" and cl < zone_low - risk * 0.25) or (side == "SELL" and cl > zone_high + risk * 0.25):
                events.append(("CANCELLED", timestamp))
                ACTIVE_SIGNALS.pop(symbol, None)
                break
        item["last_checked"] = str(timestamp)

    save_active_signals()
    return events if events else "активний сигнал; чекаємо повернення в зону входу або TP/SL"


async def notify_trade_events(app, symbol, item, events):
    for event, timestamp in events:
        if event == "TP1":
            text = f"✅ TP1 досягнуто — {symbol} {item['side']}\nРівень: {fmt(symbol, float(item['tp1']))}\nСигнал активний, очікуємо TP2 або SL.\nСвічка: {timestamp}"
        elif event == "TP2":
            text = f"🎯 TP2 досягнуто — {symbol} {item['side']}\nРівень: {fmt(symbol, float(item['tp2']))}\nСигнал завершено.\nСвічка: {timestamp}"
        elif event == "AMBIGUOUS":
            text = (f"⚠️ НЕОДНОЗНАЧНА H1-СВІЧКА — {symbol} {item['side']}\n"
                    f"Діапазон свічки торкнувся і SL, і TP. За OHLC неможливо надійно визначити, що було першим.\n"
                    f"Бот не зараховує SL/TP за цією свічкою; перевір котирування брокера.\nСвічка: {timestamp}")
        elif event == "ENTRY_CONFIRMED":
            text = f"✅ ПОЧАТКОВИЙ СИГНАЛ ПІДТВЕРДЖЕНО ПІСЛЯ ПОВЕРНЕННЯ В ЗОНУ — {symbol} {item['side']}\nЗона: {fmt(symbol, float(item.get('zone_low', item['entry'])))}–{fmt(symbol, float(item.get('zone_high', item['entry'])))}\nH1 закрилася в зоні в напрямку початкового сигналу {item['side']}.\nСвічка: {timestamp}"
        elif event == "CANCELLED":
            text = f"❌ ПОЧАТКОВИЙ СИГНАЛ СКАСОВАНО — {symbol} {item['side']}\nПісля повернення до зони H1 закрилася суттєво проти початкового напрямку.\nСвічка: {timestamp}"
        else:
            text = f"🛑 STOP LOSS — {symbol} {item['side']}\nРівень: {fmt(symbol, float(item['sl']))}\nСигнал завершено.\nСвічка: {timestamp}"
        for chat_id in list(SUBSCRIBERS):
            try:
                await app.bot.send_message(chat_id=chat_id, text=text)
            except Exception:
                log.exception("Could not send %s event for %s to chat %s", event, symbol, chat_id)


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


def download_for_symbol(symbol, period, interval):
    """Try the preferred Yahoo ticker, then a documented fallback if data is missing."""
    primary = SYMBOLS[symbol]
    candidates = [primary] + FALLBACK_TICKERS.get(symbol, [])
    for ticker in candidates:
        frame = download(ticker, period, interval)
        if not frame.empty and len(frame) >= 20:
            if ticker != primary:
                log.warning("%s: using fallback Yahoo ticker %s; price may differ from broker spot/CFD", symbol, ticker)
            return frame, ticker
    return pd.DataFrame(), primary


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


def rsi_series(close, period=14):
    """RSI using Wilder-style exponential smoothing."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)
    return rsi


def detect_smc_rsi(h1_closed):
    """Practical SMC + RSI confluence using completed candles only."""
    if h1_closed.empty or len(h1_closed) < 60:
        return None, "Недостатньо H1-свічок для RSI та Smart Money"

    df = h1_closed.copy()
    df["rsi"] = rsi_series(df["close"])
    # Search the last 12 closed candles for a complete setup.
    start_i = max(25, len(df) - 12)
    found = None

    for i in range(start_i, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        if pd.isna(row.rsi) or pd.isna(prev.rsi):
            continue
        sell_rsi = float(prev.rsi) >= 70 and float(row.rsi) < 70
        buy_rsi = float(prev.rsi) <= 30 and float(row.rsi) > 30
        if not (buy_rsi or sell_rsi):
            continue
        side = "BUY" if buy_rsi else "SELL"

        # Liquidity sweep can occur up to five candles before RSI reversal.
        sweep_idx = None
        for k in range(i, max(4, i - 6), -1):
            candle = df.iloc[k]
            prior = df.iloc[max(0, k - 20):k]
            if len(prior) < 10:
                continue
            if side == "BUY":
                swept = float(candle.low) < float(prior.low.min()) and float(candle.close) > float(prior.low.min())
            else:
                swept = float(candle.high) > float(prior.high.max()) and float(candle.close) < float(prior.high.max())
            if swept:
                sweep_idx = k
                break
        if sweep_idx is None:
            continue

        # Structure shift must occur after the sweep, within five closed candles.
        structure_idx = None
        for j in range(sweep_idx + 1, min(i + 1, sweep_idx + 6)):
            candle = df.iloc[j]
            before = df.iloc[max(0, j - 5):j]
            if len(before) < 3:
                continue
            if side == "BUY" and float(candle.close) > float(before.high.max()):
                structure_idx = j
                break
            if side == "SELL" and float(candle.close) < float(before.low.min()):
                structure_idx = j
                break
        if structure_idx is None:
            continue

        # Nearby FVG or opposing-candle order-block proxy.
        zone_start = max(0, sweep_idx - 8)
        zone_end = min(len(df), structure_idx + 2)
        scan = df.iloc[zone_start:zone_end]
        fvg = ob = False
        for k in range(2, len(scan)):
            left, middle, right = scan.iloc[k - 2], scan.iloc[k - 1], scan.iloc[k]
            if side == "BUY":
                fvg |= float(left.high) < float(right.low)
                ob |= (float(middle.close) < float(middle.open) and
                       float(right.close) > float(right.open) and
                       float(right.close) > float(middle.high))
            else:
                fvg |= float(left.low) > float(right.high)
                ob |= (float(middle.close) > float(middle.open) and
                       float(right.close) < float(right.open) and
                       float(right.close) < float(middle.low))
        if not (fvg or ob):
            continue

        if side == "BUY":
            reasons = ["RSI: повернення вище 30 після перепроданості",
                       "SMC: зняття ліквідності під локальними мінімумами",
                       "SMC: бичачий CHoCH/BOS після зняття ліквідності",
                       "SMC: знайдено bullish FVG або Order Block поблизу"]
        else:
            reasons = ["RSI: повернення нижче 70 після перекупленості",
                       "SMC: зняття ліквідності над локальними максимумами",
                       "SMC: ведмежий CHoCH/BOS після зняття ліквідності",
                       "SMC: знайдено bearish FVG або Order Block поблизу"]
        found = {"side": side, "index": i, "rsi": float(row.rsi), "reasons": reasons}
        break

    if not found:
        return None, "Немає повного збігу в останніх 12 H1-свічках: RSI 70/30 + sweep + CHoCH/BOS + FVG/OB"
    return found, "Setup"

def detect_overbought_oversold(h1_closed):
    """Standalone RSI reversal strategy; uses completed H1 candles only."""
    if h1_closed.empty or len(h1_closed) < 30:
        return None, "Недостатньо H1-свічок для RSI"
    df = h1_closed.copy()
    df["rsi"] = rsi_series(df["close"])
    # Inspect the latest 24 closed candles, newest first.
    start = max(1, len(df) - 24)
    for i in range(len(df) - 1, start - 1, -1):
        cur, prev = df.iloc[i], df.iloc[i - 1]
        if pd.isna(cur.rsi) or pd.isna(prev.rsi):
            continue
        # BUY: RSI was oversold and crosses back above 30, with bullish close.
        if float(prev.rsi) <= 30 and float(cur.rsi) > 30 and float(cur.close) > float(cur.open):
            return {"side": "BUY", "index": i, "rsi": float(cur.rsi),
                    "reasons": ["Стратегія Overbought/Oversold", "RSI вийшов вище 30 із перепроданості", "Остання закрита H1-свічка бичача"]}, "Setup"
        # SELL: RSI was overbought and crosses back below 70, with bearish close.
        if float(prev.rsi) >= 70 and float(cur.rsi) < 70 and float(cur.close) < float(cur.open):
            return {"side": "SELL", "index": i, "rsi": float(cur.rsi),
                    "reasons": ["Стратегія Overbought/Oversold", "RSI опустився нижче 70 із перекупленості", "Остання закрита H1-свічка ведмежа"]}, "Setup"
    return None, "Немає RSI-перетину 30/70 з підтверджувальною свічкою за останні 24 H1-свічки"



def detect_daily_zone_setup(d1_closed, h4_closed, h1_closed):
    """Daily support/resistance reaction or break-and-retest confirmed on H4/H1."""
    if len(d1_closed) < 35 or len(h4_closed) < 10 or len(h1_closed) < 10:
        return None, "Недостатньо даних для Daily зон"
    # Use prior daily candles only; exclude the latest 3 days from level calculation
    # to reduce the chance that the current reaction defines its own level.
    prior = d1_closed.iloc[-63:-3] if len(d1_closed) >= 63 else d1_closed.iloc[:-3]
    if len(prior) < 20:
        return None, "Недостатньо історії Daily для зон"
    support = float(prior.low.quantile(0.08))
    resistance = float(prior.high.quantile(0.92))
    last_h1 = h1_closed.iloc[-1]
    prev_h1 = h1_closed.iloc[-2]
    last_h4 = h4_closed.iloc[-1]
    price = float(last_h1.close)
    # Approximate tolerance with daily range, capped to avoid excessively broad zones.
    daily_range = float((prior.high - prior.low).tail(20).median())
    tolerance = max(daily_range * 0.18, abs(price) * 0.0015)
    low, high = float(last_h1.low), float(last_h1.high)
    op, close = float(last_h1.open), float(last_h1.close)
    h4close = float(last_h4.close)
    # Support bounce BUY: H1 tags the daily support area and closes bullish.
    if low <= support + tolerance and close > op and close > support:
        return {"side":"BUY", "index":len(h1_closed)-1, "rsi":float(rsi_series(h1_closed.close).iloc[-1]),
                "reasons":[f"Daily підтримка: {support:.5f}", "H1 торкнулася зони підтримки та закрилася бичачою свічкою", "Стратегія Daily Support/Resistance"]}, "Daily support bounce"
    # Support break + retest -> SELL. Require H4 close below support and H1 retest from below.
    if h4close < support - tolerance and high >= support - tolerance and close < support and close < op:
        return {"side":"SELL", "index":len(h1_closed)-1, "rsi":float(rsi_series(h1_closed.close).iloc[-1]),
                "reasons":[f"Daily підтримку пробито: {support:.5f}", "H4 закрився нижче підтримки", "H1 зробила ретест знизу й закрилася ведмежою", "Стратегія Daily Break-and-Retest"]}, "Daily support breakdown retest"
    # Resistance rejection SELL.
    if high >= resistance - tolerance and close < op and close < resistance:
        return {"side":"SELL", "index":len(h1_closed)-1, "rsi":float(rsi_series(h1_closed.close).iloc[-1]),
                "reasons":[f"Daily опір: {resistance:.5f}", "H1 торкнулася зони опору та закрилася ведмежою свічкою", "Стратегія Daily Support/Resistance"]}, "Daily resistance rejection"
    # Resistance break + retest -> BUY.
    if h4close > resistance + tolerance and low <= resistance + tolerance and close > resistance and close > op:
        return {"side":"BUY", "index":len(h1_closed)-1, "rsi":float(rsi_series(h1_closed.close).iloc[-1]),
                "reasons":[f"Daily опір пробито: {resistance:.5f}", "H4 закрився вище опору", "H1 зробила ретест зверху й закрилася бичачою", "Стратегія Daily Break-and-Retest"]}, "Daily resistance breakout retest"
    return None, f"Daily рівні: підтримка {support:.5f}, опір {resistance:.5f}; реакції/ретесту поки немає"

def analyze(h1, h4, d1, w1):
    """SMC + RSI: W1/D1 context, H4 trend filter, H1 liquidity sweep and structure shift."""
    if len(h1) < 80 or len(h4) < 50 or len(d1) < 30 or len(w1) < 20:
        return None, "Недостатньо історичних даних на одному з таймфреймів"

    # Exclude each currently forming candle from every decision.
    h1c, h4c, d1c, w1c = h1.iloc[:-1], h4.iloc[:-1], d1.iloc[:-1], w1.iloc[:-1]
    if min(len(h1c), len(h4c), len(d1c), len(w1c)) < 20:
        return None, "Недостатньо закритих свічок для аналізу"

    smc_setup, smc_status = detect_smc_rsi(h1c)
    obos_setup, obos_status = detect_overbought_oversold(h1c)
    daily_setup, daily_status = detect_daily_zone_setup(d1c, h4c, h1c)
    # Three independent candidate sources: SMC+RSI, RSI reversal, or Daily zone reaction/retest.
    # Choose the most recent completed-H1 setup, and never combine opposing candidates.
    candidates = []
    if smc_setup:
        candidates.append((int(smc_setup["index"]), 3, smc_setup, "Smart Money + RSI"))
    if obos_setup:
        candidates.append((int(obos_setup["index"]), 2, obos_setup, "Overbought/Oversold"))
    if daily_setup:
        candidates.append((int(daily_setup["index"]), 1, daily_setup, "Daily Support/Resistance"))
    if not candidates:
        return None, f"Немає сигналу. SMC: {smc_status}. RSI: {obos_status}. Daily: {daily_status}"
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, setup, strategy_name = candidates[0]
    side = setup["side"]

    w_bias, d_bias, h4_bias = ema_bias(w1c), ema_bias(d1c), ema_bias(h4c)
    expected = "BULL" if side == "BUY" else "BEAR"
    if d_bias not in (expected, "NEUTRAL"):
        return None, f"D1-фільтр: {side} відхилено, бо D1={d_bias}"
    if d_bias == "NEUTRAL" and h4_bias != expected:
        return None, f"D1 нейтральний, а H4={h4_bias}; потрібне підтвердження H4={expected}"

    warnings = []
    if d_bias == "NEUTRAL":
        warnings.append("D1 нейтральний; напрямок підтверджено H4")
    for tf, bias in (("W1", w_bias), ("H4", h4_bias)):
        if bias not in ("NEUTRAL", expected):
            warnings.append(f"УВАГА: {tf} проти сигналу ({bias})")

    sig_i = setup["index"]
    sig_candle = h1c.iloc[sig_i]
    # Entry is current last closed H1 close so the bot doesn't send a stale historical entry.
    last = h1c.iloc[-1]
    entry = float(last.close)
    recent_h1 = h1c.tail(5)
    candle_range = max(float(last.high) - float(last.low), abs(entry) * 0.0003)
    if side == "BUY":
        sl = float(recent_h1.low.min()) - candle_range * 0.15
        risk = entry - sl
        if risk <= 0:
            return None, "Не вдалося розрахувати стоп-лос"
        targets = [entry + risk, entry + 2 * risk]
    else:
        sl = float(recent_h1.high.max()) + candle_range * 0.15
        risk = sl - entry
        if risk <= 0:
            return None, "Не вдалося розрахувати стоп-лос"
        targets = [entry - risk, entry - 2 * risk]

    reasons = [f"Стратегія: {strategy_name}"] + setup["reasons"] + [f"RSI H1: {setup['rsi']:.1f}", "Контекст: W1/D1/H4 фільтр"] + warnings
    return {
        "side": side, "entry": entry, "sl": sl, "tp": targets,
        "zone_low": entry - risk * 0.15, "zone_high": entry + risk * 0.15,
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
        f"Причини:\n{reasons}\n"
        f"Контекст: {signal['context']}\n"
        f"Підтвердження: закрита свічка H1\n"
        f"Свічка H1: {signal['h1']}\n\n"
        "⚠️ Не гарантія прибутку. Перевір котирування, спред і ризик. XAU/XAG беруться з Yahoo Finance (XAUUSD=X/XAGUSD=X); вони можуть відрізнятися від CFD брокера."
    )


async def scan(app, report_chat_id=None):
    sent = 0
    diagnostics = []
    for symbol, ticker in SYMBOLS.items():
        try:
            h1, used_ticker = download_for_symbol(symbol, "60d", "1h")
            d1, _ = download_for_symbol(symbol, "2y", "1d")
            w1, _ = download_for_symbol(symbol, "5y", "1wk")
            if h1.empty or d1.empty or w1.empty:
                diagnostics.append(f"{symbol}: Yahoo Finance не повернув достатньо даних (основний тикер {ticker}; fallback перевірено)")
                continue
            # Active signals are monitored by a separate one-minute job using 1m quotes.
            # Never generate a second signal for the same symbol while one is unresolved.
            if symbol in ACTIVE_SIGNALS:
                diagnostics.append(f"{symbol}: активний сигнал; TP/SL перевіряються окремо щохвилини")
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
            send_errors = []
            subscribers = list(SUBSCRIBERS)
            if not subscribers:
                diagnostics.append(f"{symbol}: сигнал знайдено, але немає підписників; надішли /start")
                continue
            message_text = make_message(symbol, signal)
            for chat_id in subscribers:
                try:
                    await app.bot.send_message(
                        chat_id=chat_id,
                        text=message_text,
                        parse_mode="HTML",
                    )
                    delivered = True
                    sent += 1
                except Exception as exc:
                    # Log the actual exception and surface a short actionable version
                    # in /scan diagnostics instead of hiding the cause.
                    log.exception("Telegram send failed for symbol=%s chat_id=%s", symbol, chat_id)
                    detail = f"{type(exc).__name__}: {str(exc)}".replace("\n", " ")
                    send_errors.append(detail[:220])
            if delivered:
                ALERTED.add(key)
                save_alerts()
                ACTIVE_SIGNALS[symbol] = {
                    "side": signal["side"],
                    "entry": float(signal["entry"]),
                    "sl": float(signal["sl"]),
                    "tp1": float(signal["tp"][0]),
                    "tp2": float(signal["tp"][1]),
                    "zone_low": float(signal.get("zone_low", signal["entry"])),
                    "zone_high": float(signal.get("zone_high", signal["entry"])),
                    "h1": signal["h1"],
                    "last_checked": signal["h1"],
                    "tp1_notified": False,
                    "tp2_notified": False,
                    "sl_notified": False,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                save_active_signals()
                if send_errors:
                    diagnostics.append(f"{symbol}: доставлено не всім; помилки: {'; '.join(send_errors)}")
            else:
                diagnostics.append(f"{symbol}: сигнал знайдено, але Telegram відхилив надсилання: {'; '.join(send_errors) or 'невідома помилка'}")
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


async def monitor_active_signals(context: ContextTypes.DEFAULT_TYPE):
    """Check active trades independently every minute using Yahoo 1-minute candles.

    Yahoo quotes are not guaranteed real-time and may be unavailable for some tickers.
    The function deliberately does not infer event order if a single candle spans both
    SL and TP. Signals remain active until an unambiguous event is observed.
    """
    app = context.application
    for symbol in list(ACTIVE_SIGNALS.keys()):
        item = ACTIVE_SIGNALS.get(symbol)
        if not item:
            continue
        try:
            frame, used_ticker = download_for_symbol(symbol, "1d", "1m")
            if frame.empty or len(frame) < 2:
                log.warning("No usable 1m quotes for active signal %s; TP/SL check skipped this minute", symbol)
                continue
            events = check_active_signal(symbol, frame)
            if isinstance(events, list) and events:
                await notify_trade_events(app, symbol, item, events)
        except Exception:
            log.exception("Active signal monitor failed for %s", symbol)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    save_subscribers()
    await update.effective_message.reply_text(
        f"Dmytro Gold Signals активний. Нові сигнали скануються кожні {INTERVAL} хв; активні TP/SL перевіряються окремо щохвилини.\n"
        "Логіка: RSI 70/30 + sweep ліквідності + CHoCH/BOS + FVG/OB; фільтр D1/H4.\n"
        "M15 і Stochastic прибрані. Повторний сигнал по інструменту не надсилається, доки попередній не завершиться на TP2 або SL. Автоматичних угод немає.\n"
        "Команди: /status /symbols /scan /help"
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        f"Бот працює. Інтервал: {INTERVAL} хв. Підписників: {len(SUBSCRIBERS)}. Якщо після перезапуску число знову 0, потрібне постійне сховище Railway для subscribers.json.\n"
        "Три незалежні пошуки: Smart Money + RSI; Overbought/Oversold; Daily Support/Resistance. Активні TP/SL перевіряються щохвилини за 1m Yahoo-котируваннями, якщо вони доступні. Автоматичної торгівлі немає. Yahoo Finance може відрізнятися від Vantage MT5 і не гарантує реального часу."
    )


async def symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Інструменти:\n" + "\n".join("• " + s for s in SYMBOLS))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "/start — підписатися\n/status — стан бота\n/symbols — список інструментів\n"
        "/scan — сканувати зараз із діагностикою\n/help — команди"
    )


async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Сканую 9 інструментів за трьома логіками: Smart Money + RSI, Overbought/Oversold, Daily підтримка/опір з відбоєм або пробоєм і ретестом. 2 тейки, сповіщення TP1/TP2/SL.")
    await scan(context.application, update.effective_chat.id)


async def post_init(app: Application):
    load_subscribers()
    load_alerts()
    load_active_signals()
    if app.job_queue is None:
        raise RuntimeError("У requirements.txt має бути python-telegram-bot[job-queue]")
    app.job_queue.run_repeating(scheduled_scan, interval=INTERVAL * 60, first=15, name="market-scan")
    app.job_queue.run_repeating(monitor_active_signals, interval=60, first=20, name="active-signal-monitor-1m")
    log.info("Bot started. Subscribers=%d; new-entry scan=%d min; active TP/SL monitor=60 sec", len(SUBSCRIBERS), INTERVAL)


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
