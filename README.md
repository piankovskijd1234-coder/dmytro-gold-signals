Dmytro Gold Signals — TDA PRO 2.0
Railway Variables:
TELEGRAM_BOT_TOKEN required
FINNHUB_API_KEY optional for /news
Deploy by replacing bot.py, requirements.txt, and Procfile in GitHub, then commit/push. In Railway run exactly one worker/replica for this Telegram token.
In Telegram send /start or /alerts_on to subscribe to automatic pre-alerts. /alerts_off disables them. /scan runs a manual scan; /status shows state.
Pre-alerts: the worker checks once per minute. It estimates approach time from distance to the setup zone and recent 1-minute candle range, and sends a warning when estimated arrival is roughly 1–6 minutes away. This is NOT a guaranteed five-minute prediction. Fast volatility changes, gaps, spread, market closures, and Yahoo Finance delays can make the estimate inaccurate. It does not guarantee a final entry signal. Wait for price to actually reach the zone and confirm on the broker chart.
INTRADAY: W1/D1/H4 bias alignment; search H1 BOS among recent completed candles; wait for retest and M15 confirmation. SCALP: separate pullback-zone logic with M5/M1 candle confirmation. Yahoo Finance is not broker data and does not provide real footprint, bid/ask delta, order book, or broker spread. No trades are executed. Backtest/demo-test first. No performance guarantee.
Automatic subscriptions and alert deduplication are kept in memory and reset when Railway restarts. Only one worker instance should run.
