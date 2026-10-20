import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ============================================================
# TELEGRAM SIGNAL BOT
# Strategies:
# 1) Smart Money: W1 context, D1/H4/H1 FVG, liquidity sweep OR
#    structure close/hold, with a closed H1 confirmation.
# 2) Trendline breakout: breakout on D1 or H4, then H1 retest
#    and confirmation.
#
# Data source: Yahoo Finance (not MT5). Signals are informational.
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("signal-bot")

TOKEN = "".join(os.getenv("TELEGRAM_BOT_TOKEN", "").split())
SCAN_INTERVAL = max(60, int(os.getenv("SCAN_INTERVAL", "300")))
SUBSCRIBERS_FILE = Path("subscribers.json")
ALERTS_FILE = Path("alerts.json")

# Set MIN_CONFIRMATIONS=1 to allow either strategy to alert independently.
# Set it to 2 to require both strategies in the same direction.
MIN_CONFIRMATIONS = max(1, min(2, int(os.getenv("MIN_CONFIRMATIONS", "1"))))

SYMBOLS = {
    "BTCUSD": "BTC-USD",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "EURJPY": "EURJPY=X",
    "AUDUSD": "AUDUSD=X",
    "XAGUSD": "XAGUSD=X",
    "EURAUD": "EURAUD=X",
    "USDCHF": "USDCHF=X",
    "GBPAUD": "GBPAUD=X",
}

last_alerts = set()


def read_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Could not read %s", path)
    return default


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def atr(df, length=14):
    previous_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(length).mean()


def pivots(df, left=3, right=3):
    highs = df["High"].astype(float).to_numpy()
    lows = df["Low"].astype(float).to_numpy()
    pivot_highs = np.full(len(df), np.nan)
    pivot_lows = np.full(len(df), np.nan)

    for i in range(left, len(df) - right):
        high_window = highs[i - left : i + right + 1]
        low_window = lows[i - left : i + right + 1]
        if highs[i] == np.max(high_window):
            pivot_highs[i] = highs[i]
        if lows[i] == np.min(low_window):
            pivot_lows[i] = lows[i]

    return (
        pd.Series(pivot_highs, index=df.index),
        pd.Series(pivot_lows, index=df.index),
    )


def clean_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing Yahoo Finance columns: {missing}")
    out = df[required].dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index = pd.to_datetime(out.index, utc=True)
    return out


def download(ticker, period, interval):
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return clean_ohlcv(data)


def drop_incomplete_bar(df, timeframe):
    """Conservatively remove the last candle if it may still be forming."""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    last_time = pd.Timestamp(df.index[-1])
    durations = {"1h": 3600, "4h": 4 * 3600, "1d": 24 * 3600, "1wk": 7 * 86400}
    duration = durations[timeframe]
    if (now - last_time).total_seconds() < duration:
        return df.iloc[:-1]
    return df


def fetch_symbol(ticker):
    # Yahoo Finance intraday history is limited; create H4 candles from H1.
    h1 = download(ticker, "730d", "60m")
    h1 = drop_incomplete_bar(h1, "1h")
    if h1.empty:
        raise RuntimeError("Yahoo Finance returned no completed H1 candles")

    h4 = (
        h1.resample("4h", label="left", closed="left")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )
    h4 = drop_incomplete_bar(h4, "4h")

    d1 = drop_incomplete_bar(download(ticker, "max", "1d"), "1d")
    w1 = drop_incomplete_bar(download(ticker, "max", "1wk"), "1wk")

    frames = {"H1": h1, "H4": h4, "D1": d1, "W1": w1}
    for tf, frame in frames.items():
        if len(frame) < 205:
            raise RuntimeError(f"Insufficient {tf} candles: {len(frame)}")
    return frames


def fvg_zones(df, lookback=160):
    """
    Find three-candle Fair Value Gaps in chronological order.
    Bullish: High[i-2] < Low[i]
    Bearish: Low[i-2] > High[i]
    Each zone records whether subsequent candles partially/fully filled it.
    """
    zones = []
    if len(df) < 3:
        return zones

    start = max(2, len(df) - lookback)
    highs = df["High"].astype(float).to_numpy()
    lows = df["Low"].astype(float).to_numpy()

    for i in range(start, len(df)):
        left_high = highs[i - 2]
        left_low = lows[i - 2]
        gap_high = lows[i]
        gap_low = highs[i - 2]
        if gap_high > gap_low:
            later_lows = lows[i + 1 :]
            fully_filled = bool(len(later_lows) and np.any(later_lows <= gap_low))
            partially_filled = bool(
                len(later_lows) and np.any((later_lows < gap_high) & (later_lows > gap_low))
            )
            zones.append(
                {
                    "direction": "BUY",
                    "low": float(gap_low),
                    "high": float(gap_high),
                    "bar_index": i,
                    "filled": fully_filled,
                    "partial": partially_filled,
                }
            )

        # Bearish gap: current high below the low two candles earlier.
        if highs[i] < left_low:
            gap_low = highs[i]
            gap_high = lows[i - 2]
            later_highs = highs[i + 1 :]
            fully_filled = bool(len(later_highs) and np.any(later_highs >= gap_high))
            partially_filled = bool(
                len(later_highs) and np.any((later_highs > gap_low) & (later_highs < gap_high))
            )
            zones.append(
                {
                    "direction": "SELL",
                    "low": float(gap_low),
                    "high": float(gap_high),
                    "bar_index": i,
                    "filled": fully_filled,
                    "partial": partially_filled,
                }
            )
    return zones


def latest_open_fvg(df, direction=None, lookback=160):
    zones = fvg_zones(df, lookback=lookback)
    for zone in reversed(zones):
        if zone["filled"]:
            continue
        if direction and zone["direction"] != direction:
            continue
        return zone
    return None


def price_in_or_near_zone(price, zone, tolerance=0.0):
    if not zone:
        return False
    return zone["low"] - tolerance <= price <= zone["high"] + tolerance


def structure_levels(df):
    highs, lows = pivots(df, 3, 3)
    valid_highs = [
        (i, float(value))
        for i, value in enumerate(highs.to_numpy())
        if np.isfinite(value) and i < len(df) - 3
    ]
    valid_lows = [
        (i, float(value))
        for i, value in enumerate(lows.to_numpy())
        if np.isfinite(value) and i < len(df) - 3
    ]
    return (
        valid_highs[-1][1] if valid_highs else np.nan,
        valid_lows[-1][1] if valid_lows else np.nan,
    )


def higher_timeframe_bias(frames):
    """EMA trend context on W1, D1, H4; NONE means mixed/unclear."""
    result = {}
    for tf in ("W1", "D1", "H4"):
        close = frames[tf]["Close"].astype(float)
        e50 = float(ema(close, 50).iloc[-1])
        e200 = float(ema(close, 200).iloc[-1])
        last = float(close.iloc[-1])
        if last > e50 > e200:
            result[tf] = "BUY"
        elif last < e50 < e200:
            result[tf] = "SELL"
        else:
            result[tf] = "NONE"
    return result


def h1_confirmation(frames, direction, reference_zone=None):
    """
    Confirm on the last CLOSED H1 candle:
    - structure close beyond the latest confirmed swing, OR
    - liquidity sweep/reclaim, OR
    - FVG response with close in the intended direction.
    This is a heuristic, not a guarantee.
    """
    df = frames["H1"]
    if len(df) < 30:
        return False, "недостатньо закритих H1-свічок"

    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["Close"])
    high = float(last["High"])
    low = float(last["Low"])
    previous_close = float(prev["Close"])
    p_high, p_low = structure_levels(df.iloc[:-1])

    recent = df.iloc[-21:-1]
    swept_low = (
        len(recent) > 0
        and float(prev["Low"]) < float(recent["Low"].min())
        and close > float(prev["Low"])
        and close > previous_close
    )
    swept_high = (
        len(recent) > 0
        and float(prev["High"]) > float(recent["High"].max())
        and close < float(prev["High"])
        and close < previous_close
    )

    bullish_bos = np.isfinite(p_high) and close > p_high
    bearish_bos = np.isfinite(p_low) and close < p_low

    if direction == "BUY":
        if bullish_bos:
            return True, "H1 підтвердив бичачий BOS закриттям свічки"
        if swept_low:
            return True, "H1 зняв ліквідність знизу та повернувся вище"
        zone = latest_open_fvg(df, "BUY")
        if zone and price_in_or_near_zone(low, zone) and close > float(last["Open"]):
            return True, "H1 відреагував від бичачого FVG"
    else:
        if bearish_bos:
            return True, "H1 підтвердив ведмежий BOS закриттям свічки"
        if swept_high:
            return True, "H1 зняв ліквідність зверху та повернувся нижче"
        zone = latest_open_fvg(df, "SELL")
        if zone and price_in_or_near_zone(high, zone) and close < float(last["Open"]):
            return True, "H1 відреагував від ведмежого FVG"

    return False, "немає підтвердження на закритій H1-свічці"


def liquidity_or_close_setup(df, direction):
    """
    On a completed candle, accept either:
    1) sweep of recent liquidity and reclaim, or
    2) close beyond latest confirmed structure level (break/hold proxy).
    """
    if len(df) < 30:
        return False, "недостатньо свічок"

    last = df.iloc[-1]
    previous = df.iloc[-2]
    recent = df.iloc[-21:-1]
    p_high, p_low = structure_levels(df.iloc[:-1])
    close = float(last["Close"])

    swept_low = (
        direction == "BUY"
        and float(last["Low"]) < float(recent["Low"].min())
        and close > float(last["Low"])
        and close > float(previous["Close"])
    )
    swept_high = (
        direction == "SELL"
        and float(last["High"]) > float(recent["High"].max())
        and close < float(last["High"])
        and close < float(previous["Close"])
    )

    if direction == "BUY" and swept_low:
        return True, "зняття sell-side ліквідності та повернення ціни"
    if direction == "SELL" and swept_high:
        return True, "зняття buy-side ліквідності та повернення ціни"
    if direction == "BUY" and np.isfinite(p_high) and close > p_high:
        return True, "закріплення вище підтвердженого структурного максимуму"
    if direction == "SELL" and np.isfinite(p_low) and close < p_low:
        return True, "закріплення нижче підтвердженого структурного мінімуму"

    return False, "немає зняття ліквідності або закріплення за структурою"


def smart_money_signal(frames):
    """
    Smart Money:
    - W1/D1/H4 trend context is assessed.
    - An unfilled FVG on D1 or H4 provides a higher-timeframe zone/context.
    - On D1 or H4, a liquidity sweep OR structure close/hold is required.
    - H1 must independently confirm in the same direction.
    H1 FVG is also checked as a possible entry-response context.
    """
    bias = higher_timeframe_bias(frames)
    candidates = []

    for direction in ("BUY", "SELL"):
        aligned = [tf for tf, value in bias.items() if value == direction]
        # Do not require every higher timeframe to align, but reject a clear
        # opposite W1 trend. D1/H4 must have at least one matching context.
        if bias["W1"] not in ("NONE", direction):
            continue
        if not any(bias[tf] == direction for tf in ("D1", "H4")):
            continue

        d1_fvg = latest_open_fvg(frames["D1"], direction)
        h4_fvg = latest_open_fvg(frames["H4"], direction)
        h1_fvg = latest_open_fvg(frames["H1"], direction)

        setup_matches = []
        for tf in ("D1", "H4"):
            ok, why = liquidity_or_close_setup(frames[tf], direction)
            if ok:
                setup_matches.append((tf, why))

        # An FVG is context, not an automatic entry. Need a D1/H4 FVG
        # or a valid D1/H4 sweep/structure event.
        if not (d1_fvg or h4_fvg or setup_matches):
            continue
        if not setup_matches:
            continue

        confirmed, confirm_text = h1_confirmation(frames, direction, h1_fvg)
        if not confirmed:
            continue

        zone_text = []
        if d1_fvg:
            zone_text.append("FVG D1")
        if h4_fvg:
            zone_text.append("FVG H4")
        if h1_fvg:
            zone_text.append("FVG H1")
        if not zone_text:
            zone_text.append("FVG не знайдено; є структурне підтвердження")

        setup_text = "; ".join(f"{tf}: {why}" for tf, why in setup_matches)
        candidates.append(
            (
                direction,
                "Smart Money",
                f"{setup_text}; {', '.join(zone_text)}; {confirm_text}",
            )
        )

    # Avoid simultaneous BUY and SELL from the same strategy.
    if len(candidates) == 1:
        return candidates[0]
    return None


def trendline_break_retest_on_tf(df, atr_value, direction):
    """
    Heuristic trendline detection using recent confirmed pivot highs/lows.
    D1/H4 candle must close beyond the fitted line, and a later retest must
    be confirmed on that same timeframe. H1 confirmation is checked separately.
    """
    if len(df) < 50 or not np.isfinite(atr_value) or atr_value <= 0:
        return False, "недостатньо даних для трендової"

    highs, lows = pivots(df, 3, 3)
    n = len(df)
    start = max(0, n - 100)
    high_points = [(i, float(highs.iloc[i])) for i in range(start, n - 3) if np.isfinite(highs.iloc[i])]
    low_points = [(i, float(lows.iloc[i])) for i in range(start, n - 3) if np.isfinite(lows.iloc[i])]
    close = df["Close"].astype(float).to_numpy()
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    tolerance = max(float(atr_value) * 0.20, abs(float(close[-1])) * 0.0002)

    points = high_points[-4:] if direction == "BUY" else low_points[-4:]
    if len(points) < 2:
        return False, "немає достатньо підтверджених pivot-точок"

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    if direction == "BUY" and slope >= 0:
        return False, "немає низхідної трендової опору"
    if direction == "SELL" and slope <= 0:
        return False, "немає висхідної трендової підтримки"

    line = slope * np.arange(n) + intercept
    # Search a recent breakout candle, then a subsequent retest; latest bar
    # must be the retest-response candle.
    for br in range(max(int(xs[-1]) + 1, n - 12), n - 2):
        broke = (
            close[br] > line[br] + tolerance * 0.15
            if direction == "BUY"
            else close[br] < line[br] - tolerance * 0.15
        )
        if not broke:
            continue
        for rt in range(br + 1, n):
            touched = (
                low[rt] <= line[rt] + tolerance
                if direction == "BUY"
                else high[rt] >= line[rt] - tolerance
            )
            held = close[rt] > line[rt] if direction == "BUY" else close[rt] < line[rt]
            response = close[rt] > close[rt - 1] if direction == "BUY" else close[rt] < close[rt - 1]
            if rt == n - 1 and touched and held and response:
                return True, f"пробій трендової на {direction}, ретест і реакція на закритій свічці"
    return False, "немає підтвердженого пробою та ретесту"


def trendline_signal(frames):
    """Trendline breakout on D1 or H4, with a closed H1 confirmation."""
    found = []
    for tf in ("D1", "H4"):
        df = frames[tf]
        atr_value = float(atr(df).iloc[-1])
        if not np.isfinite(atr_value) or atr_value <= 0:
            atr_value = float((df["High"] - df["Low"]).tail(20).mean())

        for direction in ("BUY", "SELL"):
            ok, reason = trendline_break_retest_on_tf(df, atr_value, direction)
            if not ok:
                continue
            confirmed, confirm_text = h1_confirmation(frames, direction)
            if confirmed:
                found.append(
                    (
                        direction,
                        "Пробій трендової",
                        f"{tf}: {reason}; H1: {confirm_text}",
                    )
                )

    if len(found) == 1:
        return found[0]
    return None


def price_precision(symbol):
    if symbol.startswith("BTC"):
        return 2
    if "JPY" in symbol:
        return 3
    if symbol == "XAGUSD":
        return 3
    return 5


def build_signal_messages(symbol, ticker, frames):
    current = float(frames["H1"]["Close"].iloc[-1])
    bar_time = pd.Timestamp(frames["H1"].index[-1]).isoformat()
    atr_value = float(atr(frames["H1"]).iloc[-1])
    if not np.isfinite(atr_value) or atr_value <= 0:
        atr_value = float((frames["H1"]["High"] - frames["H1"]["Low"]).tail(20).mean())
    if not np.isfinite(atr_value) or atr_value <= 0:
        return []

    signals = []
    sm = smart_money_signal(frames)
    tl = trendline_signal(frames)
    if sm:
        signals.append(sm)
    if tl:
        signals.append(tl)

    by_direction = {
        "BUY": [s for s in signals if s[0] == "BUY"],
        "SELL": [s for s in signals if s[0] == "SELL"],
    }

    p = price_precision(symbol)
    output = []
    for direction in ("BUY", "SELL"):
        selected = by_direction[direction]
        if len(selected) < MIN_CONFIRMATIONS:
            continue

        # Volatility-based illustrative levels; not position sizing advice.
        risk_distance = max(1.5 * atr_value, abs(current) * 0.0005)
        if direction == "BUY":
            sl = current - risk_distance
            tp1 = current + risk_distance
            tp2 = current + 2 * risk_distance
        else:
            sl = current + risk_distance
            tp1 = current - risk_distance
            tp2 = current - 2 * risk_distance

        strategies = [s[1] for s in selected]
        details = "\n".join(f"• {name}: {reason}" for _, name, reason in selected)
        opposing = "SELL" if direction == "BUY" else "BUY"
        conflict_text = ""
        if by_direction[opposing]:
            conflict_text = (
                f"\n⚠️ Є також сигнал протилежного напрямку ({opposing}). "
                "Перевір графік вручну; не відкривай суперечливі угоди автоматично.\n"
            )

        key = f"{symbol}|{direction}|{bar_time}|{','.join(sorted(strategies))}"
        if key in last_alerts:
            continue

        bias = higher_timeframe_bias(frames)
        message = (
            f"📊 ТОРГОВИЙ СИГНАЛ — {symbol} ({ticker})\n"
            f"Напрямок: {direction}\n"
            f"Стратегії: {', '.join(strategies)}\n"
            f"Entry: {current:.{p}f}\n"
            f"SL: {sl:.{p}f}\n"
            f"TP1: {tp1:.{p}f}\n"
            f"TP2: {tp2:.{p}f}\n"
            f"Підтвердження: {len(selected)}/2 стратегій\n"
            f"Обґрунтування:\n{details}\n"
            f"Контекст EMA W1/D1/H4: {bias}\n"
            f"Остання закрита свічка H1 (UTC): {bar_time}\n"
            f"{conflict_text}"
            "⚠️ Інформаційний сигнал, не фінансова рекомендація. "
            "Перевір спред, новини, розмір позиції та ризик. Прибуток не гарантований."
        )
        output.append((key, message))
    return output


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribers = read_json(SUBSCRIBERS_FILE, [])
    if chat_id not in subscribers:
        subscribers.append(chat_id)
        write_json(SUBSCRIBERS_FILE, subscribers)

    await update.effective_message.reply_text(
        "Бот активний.\n"
        "Стратегії:\n"
        "1) Smart Money: W1/D1/H4, FVG на D1/H4/H1, зняття ліквідності "
        "або закріплення за структурою та підтвердження H1.\n"
        "2) Пробій трендової на D1 або H4, ретест і підтвердження H1.\n"
        "Джерело котирувань: Yahoo Finance, не MT5.\n"
        "Команди: /status /symbols /scan"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        f"Бот запущений.\n"
        f"Інструментів: {len(SYMBOLS)}\n"
        f"Інтервал сканування: {SCAN_INTERVAL} секунд\n"
        f"Мінімум стратегій для сигналу: {MIN_CONFIRMATIONS}/2\n"
        "Підтвердження виконується за закритими свічками H1."
    )


async def cmd_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Інструменти: " + ", ".join(SYMBOLS.keys()))


async def scan_once(app, force_chat=None):
    subscribers = read_json(SUBSCRIBERS_FILE, [])
    if force_chat and force_chat not in subscribers:
        subscribers.append(force_chat)
        write_json(SUBSCRIBERS_FILE, subscribers)

    stats = {"signals": 0, "errors": 0}
    for symbol, ticker in SYMBOLS.items():
        try:
            frames = await asyncio.to_thread(fetch_symbol, ticker)
            messages = build_signal_messages(symbol, ticker, frames)
            for key, message in messages:
                sent = False
                for chat_id in subscribers:
                    try:
                        await app.bot.send_message(chat_id=chat_id, text=message)
                        sent = True
                    except Exception:
                        log.exception("Telegram send failed for chat %s", chat_id)
                if sent:
                    last_alerts.add(key)
                    stats["signals"] += 1
            if not messages:
                log.info("No new confirmed signal for %s", symbol)
        except Exception:
            stats["errors"] += 1
            log.exception("Scan failed for %s (%s)", symbol, ticker)

    # Persist de-duplication state so restarts don't immediately repeat alerts.
    write_json(ALERTS_FILE, sorted(last_alerts)[-5000:])
    return stats


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        f"Запускаю перевірку {len(SYMBOLS)} інструментів. Це може тривати кілька хвилин."
    )
    result = await scan_once(context.application, force_chat=update.effective_chat.id)
    await update.effective_message.reply_text(
        f"Перевірку завершено.\n"
        f"Надісланих сигналів: {result['signals']}.\n"
        f"Помилок даних: {result['errors']}.\n"
        "Якщо сигналів немає, підтверджені умови входу не сформувалися. "
        "Докладні помилки дивись у Railway Logs."
    )


async def periodic(context: ContextTypes.DEFAULT_TYPE):
    await scan_once(context.application)


def main():
    if not TOKEN:
        raise SystemExit("Додай TELEGRAM_BOT_TOKEN у Railway Variables.")

    global last_alerts
    last_alerts = set(read_json(ALERTS_FILE, []))

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("symbols", cmd_symbols))
    app.add_handler(CommandHandler("scan", cmd_scan))

    app.job_queue.run_repeating(periodic, interval=SCAN_INTERVAL, first=15)
    log.info(
        "Starting bot: Smart Money + Trendline Break/Retest; symbols=%s; interval=%ss; min confirmations=%s",
        list(SYMBOLS),
        SCAN_INTERVAL,
        MIN_CONFIRMATIONS,
    )
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
