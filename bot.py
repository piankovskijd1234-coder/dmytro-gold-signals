
import os
import logging
import asyncio
import yfinance as yf
import pandas as pd

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

SYMBOLS = {
    "XAU/USD": "GC=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "BTC/USD": "BTC-USD",
}

TIMEFRAME = "15m"
CHECK_MINUTES = 5
subscribers = set()
last_signals = {}


def get_signal(symbol, ticker):
    try:
        df = yf.Ticker(ticker).history(
            period="5d",
            interval=TIMEFRAME,
            auto_adjust=False,
        )

        if df.empty or len(df) < 60:
            return None

        df = df.dropna().copy()

        # Використовуємо тільки закриті свічки
        df = df.iloc[:-1].copy()

        if len(df) < 55:
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        df["EMA20"] = close.ewm(
            span=20, adjust=False
        ).mean()

        df["EMA50"] = close.ewm(
            span=50, adjust=False
        ).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(
            alpha=1 / 14, adjust=False
        ).mean()

        loss = (-delta.clip(upper=0)).ewm(
            alpha=1 / 14, adjust=False
        ).mean()

        rs = gain / loss.replace(0, float("nan"))
        df["RSI"] = 100 - (100 / (1 + rs))

        previous_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        df["ATR"] = tr.rolling(14).mean()

        current = df.iloc[-1]
        recent = df.iloc[-5:-1]

        if current[["EMA20", "EMA50", "RSI", "ATR"]].isna().any():
            return None

        entry = float(current["Close"])
        atr = float(current["ATR"])
        rsi = float(current["RSI"])

        if atr <= 0:
            return None

        # BUY: висхідний тренд, відкат до EMA20
        # і пробій максимуму попередньої свічки
        bullish = (
            current["EMA20"] > current["EMA50"]
            and recent["Low"].min()
            <= recent["EMA20"].max() * 1.002
            and entry > float(df["High"].iloc[-2])
            and entry > current["EMA20"]
            and 50 <= rsi <= 68
        )

        # SELL: низхідний тренд, відкат до EMA20
        # і пробій мінімуму попередньої свічки
        bearish = (
            current["EMA20"] < current["EMA50"]
            and recent["High"].max()
            >= recent["EMA20"].min() * 0.998
            and entry < float(df["Low"].iloc[-2])
            and entry < current["EMA20"]
            and 32 <= rsi <= 50
        )

        if not bullish and not bearish:
            return None

        direction = "BUY" if bullish else "SELL"
        risk = 1.5 * atr

        if bullish:
            sl = entry - risk
            targets = [
                entry + risk,
                entry + 2 * risk,
                entry + 3 * risk,
            ]
        else:
            sl = entry + risk
            targets = [
                entry - risk,
                entry - 2 * risk,
                entry - 3 * risk,
            ]

        digits = 2 if symbol == "XAU/USD" else (
            1 if symbol == "BTC/USD" else 5
        )

        return {
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "targets": targets,
            "rsi": rsi,
            "time": str(df.index[-1]),
            "digits": digits,
        }

    except Exception:
        logging.exception("Помилка аналізу %s", symbol)
        return None


def format_signal(signal):
    d = signal["digits"]
    fmt = lambda price: f"{price:.{d}f}"

    return (
        "📊 DMYTRO GOLD SIGNALS\n\n"
        f"Інструмент: {signal['symbol']}\n"
        f"Напрямок: {signal['direction']}\n"
        f"Таймфрейм: {TIMEFRAME}\n\n"
        f"Вхід: {fmt(signal['entry'])}\n"
        f"Stop Loss: {fmt(signal['sl'])}\n"
        f"TP1: {fmt(signal['targets'][0])}\n"
        f"TP2: {fmt(signal['targets'][1])}\n"
        f"TP3: {fmt(signal['targets'][2])}\n\n"
        f"RSI: {signal['rsi']:.1f}\n"
        f"Час свічки: {signal['time']}\n\n"
        "⚠️ Сигнал автоматичний. Перевір спред "
        "і ризик перед входом."
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribers.add(chat_id)

    await update.effective_message.reply_text(
        "✅ Бот підключений!\n\n"
        "Інструменти: XAU/USD, EUR/USD, GBP/USD, BTC/USD\n"
        "Таймфрейм: 15 хвилин\n"
        "Перевірка ринку: кожні 5 хвилин\n\n"
        "Сигнали надходитимуть за наявності умов.\n\n"
        "/status — стан бота\n"
        "/symbols — інструменти\n"
        "/help — допомога"
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "✅ Бот запущений.\n"
        f"Інструментів: {len(SYMBOLS)}\n"
        f"Підписників: {len(subscribers)}\n"
        f"Таймфрейм: {TIMEFRAME}\n"
        f"Перевірка: кожні {CHECK_MINUTES} хвилин."
    )


async def symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Інструменти:\n"
        "• XAU/USD — золото\n"
        "• EUR/USD\n"
        "• GBP/USD\n"
        "• BTC/USD"
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.effective_message.reply_text(
        "/start — підключити сигнали\n"
        "/status — стан бота\n"
        "/symbols — список інструментів\n"
        "/help — допомога"
    )


async def scan_markets(context: ContextTypes.DEFAULT_TYPE):
    if not subscribers:
        logging.info("Немає підписників. Надішли боту /start.")
        return

    for symbol, ticker in SYMBOLS.items():
        signal = await asyncio.to_thread(
            get_signal, symbol, ticker
        )

        if signal is None:
            logging.info("%s: умов для сигналу немає", symbol)
            continue

        key = (symbol, signal["direction"])
        signature = signal["time"]

        if last_signals.get(key) == signature:
            continue

        message = format_signal(signal)

        for chat_id in list(subscribers):
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                )
            except Exception:
                logging.exception(
                    "Не вдалося надіслати повідомлення"
                )

        last_signals[key] = signature


async def post_init(application: Application):
    application.job_queue.run_repeating(
        scan_markets,
        interval=CHECK_MINUTES * 60,
        first=10,
        name="market_scan",
    )


def main():
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN не налаштований у Railway."
        )

    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("symbols", symbols))
    app.add_handler(CommandHandler("help", help_command))

    app.run_polling()


if __name__ == "__main__":
    main()
