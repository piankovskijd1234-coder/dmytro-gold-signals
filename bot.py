import os
import json
import logging
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
TOKEN = "".join(os.getenv("TELEGRAM_BOT_TOKEN", "").split())
INTERVAL = max(5, int(os.getenv("SCAN_INTERVAL_MINUTES", "5")))
SUBSCRIBERS_FILE = "subscribers.json"
ALERTS_FILE = "alerts.json"
ACTIVE_FILE = "active_signals.json"
SYMBOLS = {
"BTCUSD": "BTC-USD",
"EURUSD": "EURUSD=X",
"GBPUSD": "GBPUSD=X",
"EURJPY": "EURJPY=X",
"AUDUSD": "AUDUSD=X",
"XAGUSD": "XAGUSD=X",
"EURAUD": "EURAUD=X",
"USDCHF": "CHF=X",
"GBPAUD": "GBPAUD=X",
}
FALLBACK_TICKERS = {"XAGUSD": ["SI=F"]}
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmytro-gold-signals")
SUBSCRIBERS = set()
ALERTED = set()
ACTIVE_SIGNALS = {}
def load_json(path, default):
try:
with open(path, "r", encoding="utf-8") as f:
return json.load(f)
except (OSError, ValueError, TypeError):
return default
def save_json(path, data):
try:
with open(path, "w", encoding="utf-8") as f:
json.dump(data, f, ensure_ascii=False)
except OSError:
log.exception("Cannot save %s", path)
def load_state():
global SUBSCRIBERS, ALERTED, ACTIVE_SIGNALS
SUBSCRIBERS = {int(x) for x in load_json(SUBSCRIBERS_FILE, [])}
ALERTED = set(load_json(ALERTS_FILE, []))
ACTIVE_SIGNALS = load_json(ACTIVE_FILE, {})
def download(ticker, period, interval):
try:
frame = yf.download(ticker, period=period, interval=interval, progress=False,
threads=False, auto_adjust=False)
if frame is None or frame.empty:
return pd.DataFrame()
if isinstance(frame.columns, pd.MultiIndex):
frame.columns = frame.columns.get_level_values(0)
frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
required = ["open", "high", "low", "close"]
if not all(c in frame.columns for c in required):
return pd.DataFrame()
keep = required + (["volume"] if "volume" in frame.columns else [])
frame = frame[keep].apply(pd.to_numeric, errors="coerce").dropna(subset=required)
if "volume" not in frame.columns:
frame["volume"] = 0.0
if frame.index.tz is None:
frame.index = frame.index.tz_localize("UTC")
else:
frame.index = frame.index.tz_convert("UTC")
return frame.sort_index()
except Exception:
log.exception("Quote download failed: %s %s", ticker, interval)
return pd.DataFrame()
def download_symbol(symbol, period, interval):
candidates = [SYMBOLS[symbol]] + FALLBACK_TICKERS.get(symbol, [])
for ticker in candidates:
data = download(ticker, period, interval)
if len(data) >= 30:
if ticker != SYMBOLS[symbol]:
log.warning("%s uses fallback %s; price may differ from broker", symbol, ticker)
return data, ticker
return pd.DataFrame(), candidates[0]
def trend_bias(frame):
if frame.empty or len(frame) < 50:
return "NONE"
close = frame.close.astype(float)
e20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
e50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
price = close.iloc[-1]
if price > e20 > e50:
return "BUY"
if price < e20 < e50:
return "SELL"
return "NONE"
def _fvg_at(df, i, side):
"""Return an FVG zone formed at i, or None. i is the third candle of the pattern."""
if i < 2:
return None
left, right = df.iloc[i - 2], df.iloc[i]
if side == "BUY" and float(left.high) < float(right.low):
return (float(left.high), float(right.low))
if side == "SELL" and float(left.low) > float(right.high):
return (float(right.high), float(left.low))
return None
def _fvg_status(df, formed_i, zone, side):
"""Assess subsequent candles; returns open/partial/filled."""
low, high = zone
later = df.iloc[formed_i + 1:]
if later.empty:
return "open"
if side == "BUY":
deepest = float(later.low.min())
if deepest <= low:
return "filled"
if deepest < high:
return "partially filled"
else:
highest = float(later.high.max())
if highest >= high:
return "filled"
if highest > low:
return "partially filled"
return "open"
def _order_flow_proxy(df, side):
"""OHLCV proxy only; not true bid/ask delta or footprint order flow."""
last = df.iloc[-1]
candle_range = max(float(last.high) - float(last.low), 1e-12)
close_location = (float(last.close) - float(last.low)) / candle_range
directional = (
float(last.close) > float(last.open) and close_location >= 0.65
if side == "BUY" else
float(last.close) < float(last.open) and close_location <= 0.35
)
vol = float(last.get("volume", 0.0))
prior_vol = df.iloc[-21:-1]["volume"].astype(float)
avg_vol = float(prior_vol.mean()) if len(prior_vol) else 0.0
if avg_vol > 0 and vol > 0:
volume_ok = vol >= avg_vol * 1.10
detail = f"Ð¾Ð±ÑÑÐ³ {vol:.0f} / ÑÐµÑÐµÐ´Ð½ÑÐ¹ {avg_vol:.0f}"
else:
body_ratio = abs(float(last.close) - float(last.open)) / candle_range
volume_ok = body_ratio >= 0.60
detail = f"ÑÑÐ»Ð¾ ÑÐ²ÑÑÐºÐ¸ {body_ratio:.0%}; Ð¾Ð±ÑÑÐ³ Ð½ÐµÐ´Ð¾ÑÑÑÐ¿Ð½Ð¸Ð¹"
return directional and volume_ok, detail
def detect_smc(h1_closed):
"""H1: liquidity sweep -> BOS/CHoCH -> FVG/imbalance -> RB/retest -> SNR -> OF proxy."""
if h1_closed.empty or len(h1_closed) < 60:
return None, "ÐÐµÐ´Ð¾ÑÑÐ°ÑÐ½ÑÐ¾ Ð·Ð°ÐºÑÐ¸ÑÐ¸Ñ H1-ÑÐ²ÑÑÐ¾Ðº (Ð¿Ð¾ÑÑÑÐ±Ð½Ð¾ ÑÐ¾Ð½Ð°Ð¹Ð¼ÐµÐ½ÑÐµ 60)"
df = h1_closed.copy().reset_index(drop=False)
first = max(25, len(df) - 16)
candidates = []

for sweep_i in range(first, len(df) - 3):
    candle = df.iloc[sweep_i]
    prior = df.iloc[max(0, sweep_i - 20):sweep_i]
    if len(prior) < 12:
        continue
    prev_low, prev_high = float(prior.low.min()), float(prior.high.max())
    sweep_buy = float(candle.low) < prev_low and float(candle.close) > prev_low
    sweep_sell = float(candle.high) > prev_high and float(candle.close) < prev_high

    for side, swept in (("BUY", sweep_buy), ("SELL", sweep_sell)):
        if not swept:
            continue
        structure_i, broken_level = None, None
        for j in range(sweep_i + 1, min(sweep_i + 7, len(df) - 1)):
            before = df.iloc[max(0, j - 5):j]
            if len(before) < 3:
                continue
            if side == "BUY" and float(df.iloc[j].close) > float(before.high.max()):
                structure_i, broken_level = j, float(before.high.max())
                break
            if side == "SELL" and float(df.iloc[j].close) < float(before.low.min()):
                structure_i, broken_level = j, float(before.low.min())
                break
        if structure_i is None:
            continue

        for fvg_i in range(max(2, structure_i - 2), min(len(df) - 1, structure_i + 4)):
            zone = _fvg_at(df, fvg_i, side)
            if zone is None:
                continue
            state = _fvg_status(df, fvg_i, zone, side)
            if state == "filled":
                continue

            last = df.iloc[-1]
            zlow, zhigh = zone
            fvg_touch = float(last.low) <= zhigh and float(last.high) >= zlow
            tolerance = max(abs(broken_level) * 0.00025, 1e-8)
            rb_touch = (
                abs(float(last.low) - broken_level) <= tolerance
                if side == "BUY" else
                abs(float(last.high) - broken_level) <= tolerance
            )
            directional_close = (
                float(last.close) > float(last.open)
                if side == "BUY" else
                float(last.close) < float(last.open)
            )
            if not (fvg_touch or rb_touch) or not directional_close:
                continue

            # SNR confluence: nearby range edge or broken-level retest.
            prior20 = df.iloc[-21:-1]
            support, resistance = float(prior20.low.min()), float(prior20.high.max())
            snr_width = max((resistance - support) * 0.08, abs(float(last.close)) * 0.0002)
            if side == "BUY":
                snr_ok = abs(float(last.low) - support) <= snr_width or abs(float(last.close) - broken_level) <= snr_width
                snr_label = "Ð¿ÑÐ´ÑÑÐ¸Ð¼ÐºÐ° / ÑÐµÑÐµÑÑ Ð¿ÑÐ¾Ð±Ð¸ÑÐ¾Ð³Ð¾ Ð¾Ð¿Ð¾ÑÑ"
            else:
                snr_ok = abs(float(last.high) - resistance) <= snr_width or abs(float(last.close) - broken_level) <= snr_width
                snr_label = "Ð¾Ð¿ÑÑ / ÑÐµÑÐµÑÑ Ð¿ÑÐ¾Ð±Ð¸ÑÐ¾Ñ Ð¿ÑÐ´ÑÑÐ¸Ð¼ÐºÐ¸"
            if not snr_ok:
                continue

            of_ok, of_detail = _order_flow_proxy(df, side)
            if not of_ok:
                continue

            candidates.append({
                "side": side, "fvg_i": fvg_i, "zone": zone, "fvg_state": state,
                "rb_touch": rb_touch, "snr": snr_label, "order_flow": of_detail,
            })

if not candidates:
    return None, (
        "ÐÐµÐ¼Ð°Ñ Ð¿Ð¾Ð²Ð½Ð¾Ð³Ð¾ Ð½Ð°Ð±Ð¾ÑÑ: sweep + BOS/CHoCH + FVG/ÑÐ¼Ð±Ð°Ð»Ð°Ð½Ñ + RB/ÑÐµÑÑ + "
        "SNR + Ð¿ÑÐ´ÑÐ²ÐµÑÐ´Ð¶ÐµÐ½Ð½Ñ Order Flow. Ð¡Ð¸Ð³Ð½Ð°Ð» Ð½Ðµ ÑÐ¾ÑÑÑÑÐ¼Ð¾."
    )

setup = max(candidates, key=lambda x: x["fvg_i"])
side, last = setup["side"], df.iloc[-1]
entry = float(last.close)
recent = df.iloc[-6:]
candle_range = max(float(last.high) - float(last.low), abs(entry) * 0.0003)

if side == "BUY":
    sl = min(float(recent.low.min()), setup["zone"][0]) - candle_range * 0.1
    risk = entry - sl
    if risk <= 0:
        return None, "ÐÐµ Ð²Ð´Ð°Ð»Ð¾ÑÑ ÑÐ¾Ð·ÑÐ°ÑÑÐ²Ð°ÑÐ¸ SL Ð´Ð»Ñ BUY"
    tp1 = entry + risk
else:
    sl = max(float(recent.high.max()), setup["zone"][1]) + candle_range * 0.1
    risk = sl - entry
    if risk <= 0:
        return None, "ÐÐµ Ð²Ð´Ð°Ð»Ð¾ÑÑ ÑÐ¾Ð·ÑÐ°ÑÑÐ²Ð°ÑÐ¸ SL Ð´Ð»Ñ SELL"
    tp1 = entry - risk

candle_time = pd.Timestamp(df.iloc[-1][df.columns[0]]).isoformat()
return {
    "side": side, "entry": entry, "sl": sl, "tp1": tp1,
    "fvg_state": setup["fvg_state"], "fvg_zone": setup["zone"],
    "sweep": "Ð·Ð½ÑÑÑÑ Ð»ÑÐºÐ²ÑÐ´Ð½Ð¾ÑÑÑ Ð½Ð¸Ð¶ÑÐµ Ð¼ÑÐ½ÑÐ¼ÑÐ¼ÑÐ²" if side == "BUY" else "Ð·Ð½ÑÑÑÑ Ð»ÑÐºÐ²ÑÐ´Ð½Ð¾ÑÑÑ Ð²Ð¸ÑÐµ Ð¼Ð°ÐºÑÐ¸Ð¼ÑÐ¼ÑÐ²",
    "structure": "BOS/CHoCH bullish" if side == "BUY" else "BOS/CHoCH bearish",
    "snr": setup["snr"],
    "rb": "ÑÐµÑÐµÑÑ Ð¿ÑÐ¾Ð±Ð¸ÑÐ¾Ð³Ð¾ ÑÑÐ²Ð½Ñ (RB)" if setup["rb_touch"] else "ÑÐµÑÑ FVG/ÑÐ¼Ð±Ð°Ð»Ð°Ð½ÑÑ",
    "order_flow": setup["order_flow"],
    "h1_time": candle_time,
}, "Signal"
def fmt(symbol, value):
if symbol == "BTCUSD":
return f"{value:,.2f}"
if symbol == "XAGUSD":
return f"{value:,.3f}"
if "JPY" in symbol:
return f"{value:.3f}"
return f"{value:.5f}" if abs(value) < 10 else f"{value:.4f}"
def make_message(symbol, sig, context):
icon = "ð¢" if sig["side"] == "BUY" else "ð´"
low, high = sig["fvg_zone"]
return (
f"{icon} <b>SMART MONEY â {symbol}</b>\n"
f"ÐÐ°Ð¿ÑÑÐ¼Ð¾Ðº: <b>{sig['side']}</b>\n"
f"Entry: {fmt(symbol, sig['entry'])}\n"
f"SL: {fmt(symbol, sig['sl'])}\n"
f"TP1: {fmt(symbol, sig['tp1'])}\n\n"
f"ÐÑÐ´ÑÐ²ÐµÑÐ´Ð¶ÐµÐ½Ð½Ñ ÑÑÑÐ°ÑÐµÐ³ÑÑ H1:\n"
f"â¢ {sig['sweep']}\nâ¢ {sig['structure']}\n"
f"â¢ FVG/ÑÐ¼Ð±Ð°Ð»Ð°Ð½Ñ: {sig['fvg_state']} ({fmt(symbol, low)}â{fmt(symbol, high)})\n"
f"â¢ RB / ÑÐµÑÑ: {sig['rb']}\n"
f"â¢ SNR: {sig['snr']}\n"
f"â¢ Order Flow (OHLCV-Ð¿ÑÐ¾ÐºÑÑ): {sig['order_flow']}\n"
f"ÐÐ¾Ð½ÑÐµÐºÑÑ W1/D1/H4: {context}\n"
f"ÐÐ°ÐºÑÐ¸ÑÐ° H1-ÑÐ²ÑÑÐºÐ° (UTC): {sig['h1_time']}\n\n"
"â ï¸ ÐÐ½ÑÐ¾ÑÐ¼Ð°ÑÑÐ¹Ð½Ð¸Ð¹ ÑÐ¸Ð³Ð½Ð°Ð», Ð½Ðµ Ð³Ð°ÑÐ°Ð½ÑÑÑ Ð¿ÑÐ¸Ð±ÑÑÐºÑ. ÐÐµÑÐµÐ²ÑÑ ÐºÐ¾ÑÐ¸ÑÑÐ²Ð°Ð½Ð½Ñ, ÑÐ¿ÑÐµÐ´ Ñ ÑÐ¸Ð·Ð¸Ðº. Yahoo Finance Ð¼Ð¾Ð¶Ðµ Ð²ÑÐ´ÑÑÐ·Ð½ÑÑÐ¸ÑÑ Ð²ÑÐ´ ÑÑÐ½ CFD Ð±ÑÐ¾ÐºÐµÑÐ°."
)
async def scan(app, report_chat_id=None):
sent = 0
data_errors = 0
diagnostics = []
for symbol in SYMBOLS:
try:
h1, ticker = download_symbol(symbol, "60d", "1h")
d1, _ = download_symbol(symbol, "2y", "1d")
w1, _ = download_symbol(symbol, "5y", "1wk")
if h1.empty or d1.empty or w1.empty:
data_errors += 1
diagnostics.append(f"{symbol}: Yahoo Finance Ð½Ðµ Ð¿Ð¾Ð²ÐµÑÐ½ÑÐ² Ð´Ð¾ÑÑÐ°ÑÐ½ÑÐ¾ Ð´Ð°Ð½Ð¸Ñ (ÑÐ¸ÐºÐµÑ {ticker})")
log.warning("Missing data for %s ticker=%s", symbol, ticker)
continue
# Only use fully closed candles; discard current H1/H4/D1/W1 bars.
h1c, d1c, w1c = h1.iloc[:-1], d1.iloc[:-1], w1.iloc[:-1]
h4 = h1.resample("4h", label="left", closed="left").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
h4c = h4.iloc[:-1]
setup, reason = detect_smc(h1c)
if not setup:
diagnostics.append(f"{symbol}: {reason}")
continue
biases = {"H4": trend_bias(h4c), "D1": trend_bias(d1c), "W1": trend_bias(w1c)}
expected = setup["side"]
conflicts = [tf for tf, bias in biases.items() if bias not in ("NONE", expected)]
context = ", ".join(f"{tf}={bias}" for tf, bias in biases.items())
key = f"{symbol}|SMC|{setup['side']}|{setup['h1_time']}"
if key in ALERTED:
diagnostics.append(f"{symbol}: SMC-ÑÐ¸Ð³Ð½Ð°Ð» Ð´Ð»Ñ ÑÑÑÑ H1-ÑÐ²ÑÑÐºÐ¸ Ð²Ð¶Ðµ Ð½Ð°Ð´ÑÐ¸Ð»Ð°Ð²ÑÑ")
continue
message = make_message(symbol, setup, context)
if conflicts:
message += "\nâ ï¸ ÐÐ¾Ð½ÑÐ»ÑÐºÑ Ð·Ñ ÑÑÐ°ÑÑÐ¸Ð¼ ÑÐ°Ð¹Ð¼ÑÑÐµÐ¹Ð¼Ð¾Ð¼: " + ", ".join(conflicts)
if not SUBSCRIBERS:
diagnostics.append(f"{symbol}: ÑÐ¸Ð³Ð½Ð°Ð» Ñ, Ð°Ð»Ðµ Ð½ÐµÐ¼Ð°Ñ Ð¿ÑÐ´Ð¿Ð¸ÑÐ½Ð¸ÐºÑÐ²; Ð½Ð°Ð´ÑÑÐ»Ð¸ /start")
continue
delivered = False
for chat_id in list(SUBSCRIBERS):
try:
await app.bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
delivered = True
sent += 1
except Exception:
log.exception("Telegram delivery failed for %s to chat %s", symbol, chat_id)
if delivered:
ALERTED.add(key)
save_json(ALERTS_FILE, list(ALERTED)[-2000:])
diagnostics.append(f"{symbol}: ÑÐ¸Ð³Ð½Ð°Ð» Smart Money Ð½Ð°Ð´ÑÑÐ»Ð°Ð½Ð¾")
except Exception as exc:
data_errors += 1
log.exception("Scan failed for %s", symbol)
diagnostics.append(f"{symbol}: Ð¿Ð¾Ð¼Ð¸Ð»ÐºÐ° {type(exc).name}: {str(exc)[:140]}")
if report_chat_id is not None:
text = f"Ð¡ÐºÐ°Ð½ÑÐ²Ð°Ð½Ð½Ñ Ð·Ð°Ð²ÐµÑÑÐµÐ½Ð¾. Ð¡Ð¸Ð³Ð½Ð°Ð»ÑÐ½Ð¸Ñ Ð¿Ð¾Ð²ÑÐ´Ð¾Ð¼Ð»ÐµÐ½Ñ: {sent}. ÐÐ¾Ð¼Ð¸Ð»Ð¾Ðº Ð´Ð°Ð½Ð¸Ñ/Ð¾Ð±ÑÐ¾Ð±ÐºÐ¸: {data_errors}.\n\nÐÑÐ°Ð³Ð½Ð¾ÑÑÐ¸ÐºÐ°:\n"
for line in diagnostics:
piece = "â¢ " + line + "\n"
if len(text) + len(piece) > 3800:
await app.bot.send_message(chat_id=report_chat_id, text=text)
text = "ÐÑÐ°Ð³Ð½Ð¾ÑÑÐ¸ÐºÐ° (Ð¿ÑÐ¾Ð´Ð¾Ð²Ð¶ÐµÐ½Ð½Ñ):\n"
text += piece
await app.bot.send_message(chat_id=report_chat_id, text=text)
async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
await scan(context.application)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
SUBSCRIBERS.add(update.effective_chat.id)
save_json(SUBSCRIBERS_FILE, sorted(SUBSCRIBERS))
await update.effective_message.reply_text(
f"Dmytro Gold Signals Ð°ÐºÑÐ¸Ð²Ð½Ð¸Ð¹. Ð¡ÐºÐ°Ð½ÑÐ²Ð°Ð½Ð½Ñ ÐºÐ¾Ð¶Ð½Ñ {INTERVAL} ÑÐ².\n"
"Ð¡ÑÑÐ°ÑÐµÐ³ÑÑ H1: RB + SNR + FVG/ÑÐ¼Ð±Ð°Ð»Ð°Ð½Ñ + ÑÐµÑÑ Ð·Ð¾Ð½Ð¸ + Order Flow-Ð¿ÑÐ¾ÐºÑÑ. ÐÐ¸ÑÐµ TP1.\n"
"ÐÐ¾Ð¼Ð°Ð½Ð´Ð¸: /status /symbols /scan /help. ÐÐ²ÑÐ¾Ð¼Ð°ÑÐ¸ÑÐ½Ð¸Ñ ÑÐ³Ð¾Ð´ Ð½ÐµÐ¼Ð°Ñ."
)
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
await update.effective_message.reply_text(
f"ÐÐ¾Ñ Ð¿ÑÐ°ÑÑÑ. Ð¡ÑÑÐ°ÑÐµÐ³ÑÑ H1: RB + SNR + FVG/ÑÐ¼Ð±Ð°Ð»Ð°Ð½Ñ + ÑÐµÑÑ + Order Flow-Ð¿ÑÐ¾ÐºÑÑ; Ð»Ð¸ÑÐµ TP1. ÐÐ½ÑÑÑÑÐ¼ÐµÐ½ÑÑÐ²: {len(SYMBOLS)}. "
f"ÐÐ½ÑÐµÑÐ²Ð°Ð» ÑÐºÐ°Ð½ÑÐ²Ð°Ð½Ð½Ñ: {INTERVAL} ÑÐ². ÐÑÐ´Ð¿Ð¸ÑÐ½Ð¸ÐºÑÐ²: {len(SUBSCRIBERS)}.\n"
"ÐÑÐ¾Ð´Ð¸ Ð¿ÐµÑÐµÐ²ÑÑÑÑÑÑÑÑ Ð½Ð° Ð·Ð°ÐºÑÐ¸ÑÐ¸Ñ H1-ÑÐ²ÑÑÐºÐ°Ñ; H4/D1/W1 Ð²Ð¸ÐºÐ¾ÑÐ¸ÑÑÐ¾Ð²ÑÑÑÑÑÑ ÑÐº ÐºÐ¾Ð½ÑÐµÐºÑÑ."
)
async def symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
await update.effective_message.reply_text("ÐÐ½ÑÑÑÑÐ¼ÐµÐ½ÑÐ¸:\n" + "\n".join("â¢ " + s for s in SYMBOLS))
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
await update.effective_message.reply_text("/start â Ð¿ÑÐ´Ð¿Ð¸ÑÐ°ÑÐ¸ÑÑ\n/status â ÑÑÐ°Ð½\n/symbols â ÑÐ½ÑÑÑÑÐ¼ÐµÐ½ÑÐ¸\n/scan â ÑÐºÐ°Ð½ÑÐ²Ð°ÑÐ¸ Ð·Ð°ÑÐ°Ð· Ð· Ð´ÑÐ°Ð³Ð½Ð¾ÑÑÐ¸ÐºÐ¾Ñ\n/help â ÐºÐ¾Ð¼Ð°Ð½Ð´Ð¸")
async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
await update.effective_message.reply_text("Ð¡ÐºÐ°Ð½ÑÑ 9 ÑÐ½ÑÑÑÑÐ¼ÐµÐ½ÑÑÐ² Ð·Ð° H1: RB + SNR + FVG/ÑÐ¼Ð±Ð°Ð»Ð°Ð½Ñ + ÑÐµÑÑ + Order Flow-Ð¿ÑÐ¾ÐºÑÑ. ÐÐ¸ÑÐµ TP1. Ð¦Ðµ Ð¼Ð¾Ð¶Ðµ ÑÑÐ¸Ð²Ð°ÑÐ¸ Ð´Ð¾ ÑÐ²Ð¸Ð»Ð¸Ð½Ð¸.")
await scan(context.application, update.effective_chat.id)
async def post_init(app: Application):
load_state()
if app.job_queue is None:
raise RuntimeError("Ð£ requirements.txt Ð¿Ð¾ÑÑÑÐ±ÐµÐ½ python-telegram-bot[job-queue]")
app.job_queue.run_repeating(scheduled_scan, interval=INTERVAL * 60, first=15, name="market-scan")
log.info("Bot started. Subscribers=%d, symbols=%d", len(SUBSCRIBERS), len(SYMBOLS))
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
if name == "main":
main()
