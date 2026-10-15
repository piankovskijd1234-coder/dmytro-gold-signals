import os
import logging
import asyncio
import hashlib
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
CHECK_MINUTES = 5
NEWS_LEAD_MINUTES = int(os.getenv("NEWS_LEAD_MINUTES", "30"))
NEWS_BLACKOUT_MINUTES = int(os.getenv("NEWS_BLACKOUT_MINUTES", "30"))
NEWS_LOOKAHEAD_HOURS = int(os.getenv("NEWS_LOOKAHEAD_HOURS", "48"))
SYMBOLS = {
    "BTC/USD": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "EUR/JPY": "EURJPY=X",
    "AUD/USD": "AUDUSD=X",
    "XAG/USD": "SI=F",
    "EUR/AUD": "EURAUD=X",
    "USD/CHF": "USDCHF=X",
    "XAU/USD": "GC=F",
}

SUBSCRIBERS = set()
LAST_SIGNAL = {}
NEWS_CACHE = {}
NEWS_PRE_ALERTED = set()
NEWS_RESULT_ALERTED = set()
LAST_NEWS_REFRESH = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmytro-gold-signals")


def utc_now():
    return datetime.now(timezone.utc)


def parse_event_time(value):
    if not value:
        return None
    try:
        # Finnhub economic calendar timestamps are commonly ISO-like UTC strings.
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def event_id(event):
    raw = "|".join([
        str(event.get("time", "")), str(event.get("country", "")),
        str(event.get("event", "")), str(event.get("estimate", "")),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def fetch_economic_calendar():
    """Fetch upcoming/recent economic events from Finnhub. Requires FINNHUB_API_KEY."""
    if not FINNHUB_API_KEY:
        log.warning("FINNHUB_API_KEY is not set; automatic economic calendar is disabled.")
        return []
    now = utc_now()
    start = (now - timedelta(hours=3)).date().isoformat()
    end = (now + timedelta(hours=NEWS_LOOKAHEAD_HOURS)).date().isoformat()
    url = "https://finnhub.io/api/v1/calendar/economic"
    try:
        response = requests.get(
            url,
            params={"from": start, "to": end, "token": FINNHUB_API_KEY},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("economicCalendar", []) if isinstance(payload, dict) else []
        events = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            impact = str(row.get("impact", "")).lower()
            country = str(row.get("country", "")).strip()
            dt = parse_event_time(row.get("time"))
            if not dt:
                continue
            # Focus on high-impact USD/EUR/GBP events, which matter most for these instruments.
            if impact not in ("high", "3", "3.0"):
                continue
            if country.upper() not in ("US", "UNITED STATES", "EUR", "EURO AREA", "EU", "GB", "UK", "UNITED KINGDOM"):
                continue
            item = dict(row)
            item["_dt"] = dt
            item["_id"] = event_id(row)
            events.append(item)
        log.info("Economic calendar refreshed: %s high-impact events", len(events))
        return events
    except Exception:
        log.exception("Finnhub economic calendar request failed")
        return []


def currency_scope(event):
    country = str(event.get("country", "")).upper()
    name = str(event.get("event", "")).lower()
    if country in ("US", "UNITED STATES") or any(x in name for x in (
        "fomc", "federal reserve", "nonfarm", "non-farm", "cpi", "pce", "ism",
        "jobless claims", "gdp", "retail sales", "payroll", "powell"
    )):
        return "USD"
    if country in ("GB", "UK", "UNITED KINGDOM") or any(x in name for x in (
        "bank of england", "boe", "uk cpi", "uk gdp", "claimant count", "bailey"
    )):
        return "GBP"
    if country in ("EUR", "EURO AREA", "EU") or any(x in name for x in (
        "ecb", "european central bank", "eurozone", "euro area", "lagarde"
    )):
        return "EUR"
    return "OTHER"


def expected_impact_text(event):
    """Scenario guide, not a directional prediction."""
    name = str(event.get("event", "")).lower()
    scope = currency_scope(event)
    if any(k in name for k in ("interest rate", "rate decision", "fomc", "monetary policy", "bank rate")):
        return (
            "Ставки/центробанк: жорсткіший тон або вища ставка за очікування може підтримати відповідну валюту "
            "та тиснути на золото; м’якший тон може дати протилежну реакцію. Реакція залежить від того, "
            "що вже закладено в ціну."
        )
    if any(k in name for k in ("cpi", "inflation", "pce price", "consumer prices")):
        return (
            "Інфляція: показник вище прогнозу може підсилити очікування жорсткішої політики та підтримати валюту; "
            "нижче прогнозу — послабити її. Це не гарантований рух."
        )
    if any(k in name for k in ("nonfarm", "non-farm", "payroll", "unemployment", "jobless", "employment", "claimant")):
        return (
            "Ринок праці: сильніші дані часто підтримують валюту через очікування ставок; слабші можуть її послабити. "
            "Важливі також перегляди попередніх значень і зарплати."
        )
    if any(k in name for k in ("gdp", "gross domestic product", "retail sales", "ism", "pmi")):
        return (
            "Зростання/ділова активність: результат вище прогнозу може підтримати валюту, нижче — послабити її; "
            "реакція залежить від впливу на очікування центрального банку."
        )
    if any(k in name for k in ("speech", "speaks", "testimony", "press conference")):
        return (
            "Виступ: ринок оцінюватиме коментарі про інфляцію, ставки та економіку. Напрямок заздалегідь невідомий; "
            "важливі формулювання та відхилення від очікувань."
        )
    return (
        "Напрямок не можна надійно визначити лише з назви події. Порівняй фактичне значення з прогнозом і попереднім "
        "значенням та дочекайся реакції ціни."
    )


def impact_map(event):
    scope = currency_scope(event)
    if scope == "USD":
        return (
            "XAU/USD: часто обернена реакція на долар/дохідності, але не завжди; "
            "EUR/USD і GBP/USD: можливий рух проти USD; BTC/USD: реакція залежить також від апетиту до ризику."
        )
    if scope == "EUR":
        return (
            "EUR/USD: прямий вплив на євро; XAU/USD: можливий непрямий вплив через USD та дохідності; "
            "GBP/USD: можливий перетік через EUR/GBP; BTC/USD: переважно непрямий вплив."
        )
    if scope == "GBP":
        return (
            "GBP/USD: прямий вплив на фунт; EUR/USD: можливий непрямий вплив через EUR/GBP та загальний USD; "
            "XAU/USD і BTC/USD: переважно непрямий вплив."
        )
    return "Вплив на XAU/USD, EUR/USD, GBP/USD і BTC/USD невизначений; перевір фактичну реакцію ціни."


def fmt_value(value):
    if value is None or str(value).strip() in ("", "None", "nan"):
        return "немає даних"
    return str(value)


def event_message(event, title="⚠️ ВАЖЛИВА НОВИНА"):
    dt = event["_dt"].strftime("%d.%m.%Y %H:%M UTC")
    name = event.get("event", "Економічна подія")
    country = event.get("country", "—")
    return (
        f"<b>{title}</b>\n"
        f"<b>Подія:</b> {name}\n"
        f"<b>Країна/регіон:</b> {country}\n"
        f"<b>Час:</b> {dt}\n"
        f"<b>Прогноз:</b> {fmt_value(event.get('estimate'))}\n"
        f"<b>Попереднє:</b> {fmt_value(event.get('prev'))}\n"
        f"<b>Факт:</b> {fmt_value(event.get('actual'))}\n\n"
        f"<b>Можливий сценарій:</b> {expected_impact_text(event)}\n\n"
        f"<b>Інструменти:</b> {impact_map(event)}\n\n"
        "⚠️ Це сценарний аналіз, не передбачення гарантованого напрямку. Перевір джерело та реакцію ринку."
    )


def news_is_in_blackout():
    now = utc_now()
    for event in NEWS_CACHE.values():
        dt = event["_dt"]
        # Pause new trading signals around the event, before and after.
        if dt - timedelta(minutes=NEWS_LEAD_MINUTES) <= now <= dt + timedelta(minutes=NEWS_BLACKOUT_MINUTES):
            return True, event
    return False, None


async def send_to_all(app, message):
    for chat_id in list(SUBSCRIBERS):
        try:
            await app.bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
        except Exception:
            log.exception("Telegram send failed for chat %s", chat_id)


async def process_news(app):
    global LAST_NEWS_REFRESH, NEWS_CACHE
    if not FINNHUB_API_KEY:
        return
    events = await asyncio.to_thread(fetch_economic_calendar)
    now = utc_now()
    # Keep events returned from the provider, including recent events with actual results.
    for event in events:
        NEWS_CACHE[event["_id"]] = event
    # Prune events older than 2 days to avoid unbounded memory use.
    NEWS_CACHE = {k: v for k, v in NEWS_CACHE.items() if v["_dt"] >= now - timedelta(days=2)}

    for key, event in list(NEWS_CACHE.items()):
        dt = event["_dt"]
        minutes_until = (dt - now).total_seconds() / 60
        # One alert at the chosen lead time.
        if 0 <= minutes_until <= NEWS_LEAD_MINUTES and key not in NEWS_PRE_ALERTED:
            await send_to_all(app, event_message(event, "⏰ ВАЖЛИВА НОВИНА — ЗА 30 ХВИЛИН"))
            NEWS_PRE_ALERTED.add(key)

        # After the event: send actual-versus-forecast once actual is populated.
        actual = event.get("actual")
        if dt <= now and actual not in (None, "") and key not in NEWS_RESULT_ALERTED:
            estimate = event.get("estimate")
            comparison = "Факт і прогноз доступні; звір значення та реакцію ринку."
            try:
                a, e = float(actual), float(estimate)
                comparison = (
                    f"Факт {'вище' if a > e else 'нижче' if a < e else 'дорівнює'} прогнозу "
                    f"({a:g} проти {e:g}). Сам по собі цей факт не гарантує напрямок."
                )
            except (ValueError, TypeError):
                pass
            msg = event_message(event, "📣 НОВИНА ВИЙШЛА — РЕЗУЛЬТАТ")
            msg += "\n\n<b>Факт проти прогнозу:</b> " + comparison
            await send_to_all(app, msg)
            NEWS_RESULT_ALERTED.add(key)
    LAST_NEWS_REFRESH = now


def history(ticker, period, interval):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close"]].dropna()
        return df.iloc[:-1] if len(df) > 2 else pd.DataFrame()
    except Exception:
        log.exception("Price download failed: %s", ticker)
        return pd.DataFrame()


def resample(df, rule):
    return df.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()


def indicators(df):
    if df.empty:
        return df
    x = df.copy()
    x["EMA20"] = x.Close.ewm(span=20, adjust=False).mean()
    x["EMA50"] = x.Close.ewm(span=50, adjust=False).mean()
    x["MA50"] = x.Close.rolling(50).mean()
    x["MA100"] = x.Close.rolling(100).mean()
    x["MA200"] = x.Close.rolling(200).mean()
    prev = x.Close.shift(1)
    tr = pd.concat([x.High-x.Low, (x.High-prev).abs(), (x.Low-prev).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.rolling(14).mean()
    delta = x.Close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    al = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = ag / al.replace(0, float("nan"))
    x["RSI"] = (100 - 100/(1+rs)).fillna(100)
    return x.dropna()


def trend(df):
    x = indicators(df)
    if len(x) < 55:
        return "NEUTRAL"
    a = x.iloc[-1]
    if a.Close > a.EMA20 > a.EMA50:
        return "BUY"
    if a.Close < a.EMA20 < a.EMA50:
        return "SELL"
    return "NEUTRAL"


def latest_fvg(df, side, lookback=30):
    start = max(2, len(df)-lookback)
    for i in range(len(df)-1, start-1, -1):
        if side == "BUY" and df.Low.iloc[i] > df.High.iloc[i-2]:
            return float(df.High.iloc[i-2]), float(df.Low.iloc[i])
        if side == "SELL" and df.High.iloc[i] < df.Low.iloc[i-2]:
            return float(df.High.iloc[i]), float(df.Low.iloc[i-2])
    return None


def analyze(label, ticker):
    daily = history(ticker, "2y", "1d")
    hourly = history(ticker, "60d", "1h")
    if len(daily) < 80 or len(hourly) < 150:
        return None, f"{label}: недостатньо історичних даних"
    weekly, h4 = resample(daily, "W-FRI"), resample(hourly, "4h")
    w, d, h = trend(weekly), trend(daily), trend(h4)
    if w not in ("BUY","SELL") or w != d or w != h:
        return None, f"{label}: напрямки 1W/1D/H4 не узгоджені"
    x = indicators(hourly)
    if len(x) < 55:
        return None, f"{label}: недостатньо даних H1"
    a, prev = x.iloc[-1], x.iloc[-2]
    close, atr, rsi = float(a.Close), float(a.ATR), float(a.RSI)
    if atr <= 0:
        return None, f"{label}: ATR недоступний"
    side = w
    if side == "BUY" and not 45 <= rsi <= 68:
        return None, f"{label}: RSI {rsi:.1f} поза фільтром BUY"
    if side == "SELL" and not 32 <= rsi <= 55:
        return None, f"{label}: RSI {rsi:.1f} поза фільтром SELL"

    # SNR-рівні беремо з попередньої ділянки, не з тих самих свічок пробою
    recent = x.iloc[-27:-6]
    if len(recent) < 10:
        return None, f"{label}: недостатньо даних для SNR"
    resistance = float(recent["High"].max())
    support = float(recent["Low"].min())
    fvg = latest_fvg(x, side)
    prior = x.iloc[-6:-1]

    if side == "BUY":
        zone_reaction = (float(a.Low) <= support + 0.2*atr and close > float(a.Open) and close > float(prev.Close))
        fvg_reaction = (fvg is not None and float(a.Low) <= fvg[1] and float(a.High) >= fvg[0] and close > float(a.Open))
        had_breakout = bool((prior["Close"] > resistance).any())
        retest = had_breakout and float(a.Low) <= resistance + 0.15*atr and close > resistance and close > float(a.Open)
        confirmed = zone_reaction or fvg_reaction or retest
        reason = "пробій + ретест SNR" if retest else "реакція від FVG" if fvg_reaction else "реакція від підтримки SNR"
        sl = min(float(a.Low), support) - 0.25*atr
        risk = close - sl
        tps = [close + r*risk for r in (1, 2, 3)]
    else:
        zone_reaction = (float(a.High) >= resistance - 0.2*atr and close < float(a.Open) and close < float(prev.Close))
        fvg_reaction = (fvg is not None and float(a.High) >= fvg[0] and float(a.Low) <= fvg[1] and close < float(a.Open))
        had_breakout = bool((prior["Close"] < support).any())
        retest = had_breakout and float(a.High) >= support - 0.15*atr and close < support and close < float(a.Open)
        confirmed = zone_reaction or fvg_reaction or retest
        reason = "пробій + ретест SNR" if retest else "реакція від FVG" if fvg_reaction else "реакція від опору SNR"
        sl = max(float(a.High), resistance) + 0.25*atr
        risk = sl - close
        tps = [close - r*risk for r in (1, 2, 3)]
    if not confirmed or risk <= 0:
        return None, f"{label}: немає підтвердження H1"
    decimals = 3 if label in ("EUR/JPY",) else 2 if label in ("XAU/USD","XAG/USD","BTC/USD") else 5
    ma50 = float(a.MA50) if pd.notna(a.MA50) else None
    ma100 = float(a.MA100) if pd.notna(a.MA100) else None
    ma200 = float(a.MA200) if pd.notna(a.MA200) else None
    return {"symbol":label,"side":side,"entry":close,"sl":sl,"tps":tps,"reason":reason,
            "rsi":rsi,"ma50":ma50,"ma100":ma100,"ma200":ma200,
            "time":x.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
            "decimals":decimals,"candle":str(x.index[-1])}, None


def format_signal(s):
    p = s["decimals"]
    side = "🟢 BUY" if s["side"] == "BUY" else "🔴 SELL"
    ma50 = f"{s['ma50']:.{p}f}" if s.get("ma50") is not None else "немає даних"
    ma100 = f"{s['ma100']:.{p}f}" if s.get("ma100") is not None else "немає даних"
    ma200 = f"{s['ma200']:.{p}f}" if s.get("ma200") is not None else "немає даних"
    return (
        f"📊 <b>{s['symbol']} — {side}</b>\n"
        f"Вхід: <code>{s['entry']:.{p}f}</code>\n"
        f"SL: <code>{s['sl']:.{p}f}</code>\n"
        f"TP1: <code>{s['tps'][0]:.{p}f}</code>\n"
        f"TP2: <code>{s['tps'][1]:.{p}f}</code>\n"
        f"TP3: <code>{s['tps'][2]:.{p}f}</code>\n"
        f"RSI(14) H1: <b>{s['rsi']:.1f}</b>\n\n"
        f"🔴 MA50: <code>{ma50}</code>\n"
        f"🔵 MA100: <code>{ma100}</code>\n"
        f"🟢 MA200: <code>{ma200}</code>\n\n"
        f"Підтвердження: {s['reason']}\n"
        f"Контекст: 1W → 1D → H4 → H1\n"
        f"Час: {s['time']}\n\n"
        "⚠️ Алгоритмічний сигнал, не гарантія прибутку."
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    await update.message.reply_text(
        "✅ Dmytro Gold Signals активний.\nАвтосигнали: BTC/USD, EUR/USD, GBP/USD, EUR/JPY, AUD/USD, XAG/USD, EUR/AUD, USD/CHF, XAU/USD.\n"
        "Стратегія: 1W→1D→H4→H1, RSI(14), MA50/100/200, FVG/SNR, ретест після пробою.\n"
        "Новини: повний календар і попередження за 30 хв працюють за наявності FINNHUB_API_KEY.\n"
        "Команди: /status /symbols /help"
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    blackout, event = news_is_in_blackout()
    news = "Календар новин підключений." if FINNHUB_API_KEY else "⚠️ FINNHUB_API_KEY не задано — автоматичні новини вимкнені."
    if blackout:
        news += f"\nПауза сигналів через новину: {event.get('event','подія')}."
    await update.message.reply_text(
        f"Бот запущений. Підписників у поточному запуску: {len(SUBSCRIBERS)}.\n"
        f"Сканування: кожні {CHECK_MINUTES} хв.\n{news}\n"
        "Ціни: Yahoo Finance; можуть мати затримку."
    )


async def symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Інструменти:\n" + "\n".join(SYMBOLS.keys()))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start — підписатися\n/status — стан\n/symbols — інструменти\n"
        "Новини потребують FINNHUB_API_KEY у Railway. Сповіщення за 30 хв, результат після публікації факту.\n"
        "Фільтр RSI H1: BUY 45–68, SELL 32–55. Мувінги: MA50 червоний, MA100 синій, MA200 зелений. Сигнали призупиняються навколо важливих подій."
    )


async def scan(context: ContextTypes.DEFAULT_TYPE):
    await process_news(context.application)
    blackout, event = news_is_in_blackout()
    if blackout:
        log.info("Signal scan paused around high-impact event: %s", event.get("event"))
        return
    for label, ticker in SYMBOLS.items():
        signal, note = await asyncio.to_thread(analyze, label, ticker)
        if note:
            log.info(note)
        if not signal:
            continue
        key = signal["side"] + "|" + signal["candle"] + "|" + signal["reason"]
        if LAST_SIGNAL.get(label) == key:
            continue
        LAST_SIGNAL[label] = key
        await send_to_all(context.application, format_signal(signal))


async def post_init(app):
    app.job_queue.run_repeating(scan, interval=CHECK_MINUTES*60, first=5, name="market_scan")
    log.info("Scheduled price/news scan every %s minutes", CHECK_MINUTES)


def main():
    if not TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable")
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("symbols", symbols))
    app.add_handler(CommandHandler("help", help_cmd))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
