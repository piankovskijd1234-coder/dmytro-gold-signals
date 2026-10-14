

import os
import asyncio
import logging
import yfinance as yf
import pandas as pd

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

logging.basicConfig(level=logging.INFO)
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
            interval=TIMEFRAME
        )

        if df.empty or len(df) < 60:
            return None

        df = df.dropna().copy()

        # Use completed candles only
        df = df.iloc[:-1].copy()

        if len(df) < 30:
            return None

        df["EMA20"] = df["Close"].ewm(
            span=20, adjust=False
        ).mean()

        df["EMA50"] = df["Close"].ewm(
            span=50, adjust=False
        ).mean()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0).ewm(
            alpha=1 / 14, adjust=False
        ).mean()
        loss = (-delta.clip(upper=0)).ewm(
            alpha=1 / 14, adjust=False
        ).mean()

        rs = gain / loss.replace(0, float("nan"))
        df["RSI"] = 100 - (100 / (1 + rs))

        previous_close = df["Close"].shift(1)
        tr = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - previous_close).abs(),
                (df["Low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        df["ATR"] = tr.rolling(14).mean()

        current = df.iloc[-1]
        previous = df.iloc[-11:-1]

        if previous.empty or pd.isna(current["ATR"]):
            return None

        entry = float(current["Close"])
        atr = float(current["ATR"])
        rsi = float(current["RSI"])

        if atr <= 0:
            return None

        resistance = float(previous["High"].max())
        support = float(previous["Low"].min())

        bullish = (
            current["EMA20"] > current["EMA50"]
            and entry > resistance
            and 52 <= rsi <= 68
        )

        bearish = (
            current["EMA20"] < current["EMA50"]
            and entry < support
            and 32 <= rsi <= 48
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

        candle_time = str(df.index[-1])

        return {
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "targets": targets,
            "rsi": rsi,
            "time": candle_time,
        }

    except Exception:
        logging.exception("Data error for %s", symbol)
        return None


def format_price(symbol, price):
    digits = 2 if symbol == "XAU/USD" else (
        1 if symbol == "BTC/USD" else 5
    )
    return f"{price:.{digits}f}"


def format_signal(signal):
    symbol = signal["symbol"]

    targets = "\n".join(
        f"TP{i}: {format_price(symbol, price)}"
        for i, price in enumerate(signal["targets"], 1)
    )

    return (
        f"📊 DMYTRO GOLD SIGNALS\n\n"
        f"Інструмент: {symbol}\n"
        f"Напрямок: {signal['direction']}\n"
        f"Таймфрейм: {TIMEFRAME}\n\n"
        f"Вхід: {format_price(symbol, signal['entry'])}\n"
        f"Stop Loss: {format_price(symbol, signal['sl'])}\n"
        f"{targets}\n\n"
        f"RSI: {signal['rsi']:.1f}\n"
        f"Підтвердження: EMA20/EMA50 + пробій "
        f"+ RSI\n"
        f"Час свічки: {signal['time']}\n\n"
        f"⚠️ Сигнал автоматичний. Перевір котирування "
        f"та ризик перед входом."
    )


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    chat_id = update.effective_chat.id
    subscribers.add(chat_id)

    await update.message.reply_text(
        "Вітаю! Це Dmytro Gold Signals.\n\n"
        "Інструменти: XAU/USD, EUR/USD, "
        "GBP/USD, BTC/USD.\n"
        "Таймфрейм: 15 хвилин.\n"
        "Три тейки: TP1, TP2, TP3.\n\n"
        "Сигнали надсилаються лише після "
        "підтвердження умов.\n\n"
        "/status — стан бота\n"
        "/symbols — список інструментів\n"
        "/help — допомога"
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "/start — підключити отримання сигналів\n"
        "/status — перевірити роботу бота\n"
        "/symbols — список інструментів\n"
        "/help — допомога\n\n"
        "Бот надсилає сигнали, а не відкриває "
        "угоди автоматично."
    )


async def symbols_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "📈 Інструменти:\n"
        "• XAU/USD — золото (GC=F)\n"
        "• EUR/USD\n"
        "• GBP/USD\n"
        "• BTC/USD\n\n"
        "Таймфрейм: 15m\n"
        "Тейки: TP1, TP2, TP3"
    )


async def status(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "Бот запущений.\n"
        f"Інструментів: {len(SYMBOLS)}\n"
        f"Підписників у поточному запуску: "
        f"{len(subscribers)}\n"
        "Дані: Yahoo Finance\n"
        "Сигнали: після підтвердження умов."
    )


async def scan_markets(
    context: ContextTypes.DEFAULT_TYPE
):
    if not subscribers:
        return

    for symbol, ticker in SYMBOLS.items():
        signal = await asyncio.to_thread(
            get_signal, symbol, ticker
        )

        if signal is None:
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
                    text=message
                )
            except Exception:
                logging.exception(
                    "Could not send message to %s",
                    chat_id
                )

        last_signals[key] = signature


async def post_init(application: Application):
    await application.bot.set_my_commands([
        ("start", "Підключити сигнали"),
        ("status", "Стан бота"),
        ("symbols", "Список інструментів"),
        ("help", "Допомога"),
    ])


def main():
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not configured"
        )

    app = (
        Application.builder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("symbols", symbols_command))

    app.job_queue.run_repeating(
        scan_markets,
        interval=CHECK_MINUTES * 60,
        first=10,
        name="market_scan",
    )

    app.run_polling()


if __name__ == "__main__":
    main()
