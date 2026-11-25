# UniversalAI Trading System

An independently installable MT5 signal service and chart EA. The system is divided into a small set of documented Python, HTTP, storage, and MQL5 interfaces. It has no dependency on PHANTOM-TRADERS or any other trading repository.

## Components and boundaries

| Component | Responsibility | Interface |
| --- | --- | --- |
| `brain.py` | MT5 M15/H1 data adapter, indicators, H1 bias, M15 sweep/FVG, Groq Qwen decision gate, account API | Pure functions `quant_metrics`, `market_bias`, `detect_smc`, `validate_decision`; JSON HTTP API |
| `backtest.py` | Historical M15 CSV simulation with the same quant/SMC functions | `run_backtest(DataFrame, ...)` and command line |
| `UniversalAI_EA.mq5` | MT5 polling, account-local drawdown circuit breaker, risk sizing, order execution | GET `/api/ea/poll/{account_id}/{symbol}` |
| `index.html` | Mobile-friendly account configuration and monitoring UI | Dashboard REST endpoints with `X-Dashboard-PIN` |

`brain.py` only publishes signals; it never sends an order. The EA requires a saved and active account, validates the signal and confidence, sizes from account risk and the broker's `OrderCalcProfit`, checks drawdown on every tick, and executes at market. A prop account receives a random 3–15 second delay and a random 1–3 point TP offset. The signal cache keeps a decision stable for the life of one M15 candle and avoids repeatedly asking the model on the EA's 2-second poll.

The API creates `accounts.db` in the project directory by default. Set `UNIVERSALAI_DB` to move it outside the checkout. The database is runtime state and must not be committed.

## Install on Windows

1. Install 64-bit Python 3.11 or 3.12 and MetaTrader 5. Sign in to the intended terminal manually and keep it running. Install MetaEditor with MT5.
2. In PowerShell, from this project directory:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

3. Edit `.env` locally. Set a private dashboard PIN and `GROQ_API_KEY`. Do not commit `.env`. `GROQ_MODEL` defaults to `qwen/qwen3-32b`; use a model available to your Groq account. Configure `MT5_SYMBOLS` using the exact names shown in Market Watch. MT5 package calls use the currently signed-in terminal; the Python account's login must match the account configured in the dashboard.
4. Start the API:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   python brain.py
   ```

   The default bind is `127.0.0.1:8000`, suitable for an EA on the same VM. Open `http://127.0.0.1:8000/`, enter the dashboard PIN, add the MT5 login and exact broker server name, then start the account. For remote phone access, put the app behind a TLS reverse proxy and a network firewall; do not expose the unauthenticated EA poll route directly to the internet.

5. In MT5, open **Tools → Options → Expert Advisors**, enable **Allow WebRequest for listed URL**, and add `http://127.0.0.1:8000`. Open `UniversalAI_EA.mq5` in MetaEditor and compile it, copy the generated `.ex5` into the terminal's `MQL5\Experts` directory if needed, refresh Navigator, and attach it to each symbol/timeframe chart. Confirm the EA's account login/server match the dashboard entry and enable Algo Trading only after reviewing it on a demo account.

`DASHBOARD_PIN` protects dashboard REST operations. The public health, dashboard shell and EA poll routes are intentionally excluded so MT5 does not need the PIN. Keep the server bound to localhost unless a trusted reverse proxy is configured. The page loads Tailwind from its CDN and therefore needs internet access for styling.

## MT5 signal API

`GET /api/ea/poll/{account_id}/{symbol}?balance={balance}&equity={equity}` returns `WAIT` while stopped, when there is no qualified setup, or when a decision fails validation. An accepted signal contains:

```json
{"decision":"BUY","entry":1.08,"sl":1.079,"tp":1.0825,"confidence":0.82,"reason":"...","signal_id":"EURUSD-...","risk_pct":0.5,"execution_delay_seconds":0,"tp_offset_points":0,"max_daily_dd_pct":5,"max_total_dd_pct":10}
```

All account-management routes (`/api/accounts`, `/api/accounts/{id}/status`, `/api/accounts/{id}/toggle`, `/api/logs/{id}`) require the `X-Dashboard-PIN` header. OpenAPI docs are at `/docs` on the local server.

## Quant and signal rules

- The gate requires ATR(14)/ATR(50) > 0.90 and Kaufman ER(20) > 0.25. A 50-close z-score blocks BUY above +2 and SELL below -2. At least 50 valid bars are required.
- H1 bias requires EMA20/EMA50 agreement and two confirmed fractal higher highs/higher lows or lower highs/lower lows. The latest closed M15 setup combines a 20-bar liquidity sweep (close returns inside) with a three-candle FVG. The sweep can occur in the preceding four candles; requiring both on the same candle is incompatible with using that candle's first FVG bar in the 20-bar extreme.
- Forex/Gold use UTC 07:00–10:00 and 12:00–16:00 windows. These fixed UTC approximations do not track London/New York daylight saving changes. Synthetic symbols identified by name skip the session check.
- Groq is called only after both quantitative and SMC checks pass. Its response must match the setup side, have confidence >= 0.75, valid directional stop/target, and reward/risk >= 2.5.

## Backtest

Export an M15 chart from MT5 to CSV with headers `time,open,high,low,close` (volume is optional), convert `time` from broker server time to UTC, then run:

```powershell
python backtest.py .\EURUSD_M15.csv --symbol EURUSD --risk-pct 0.5 --daily-dd 5 --total-dd 10 --output trades.csv
```

The report includes win rate, profit factor, max closed-trade drawdown, total return and monthly trade counts. Optional `--trailing-atr 1.0` enables an ATR trailing stop; default is fixed 1:2.5 reward/risk. For an OHLC candle that touches both stop and target, the simulator assumes the stop fills first. It models one open trade at a time and uses signal-candle wick stop distance; costs, spread, swaps, latency, partial fills, and broker-specific contract values are not modeled. Results are research estimates, not a prediction of live performance.

## Verification and current operational limits

Run automated checks with `python -m pytest`. The test suite uses synthetic data and a temporary SQLite database; it does not claim profitable performance or real-market validation. MT5 connectivity requires a logged-in terminal, exact broker symbols, and a licensed Groq API key. Test on a demo account before considering any live account. The EA's daily equity start resets at broker-server midnight. Its total-drawdown high-water mark begins when the EA is attached; removing/restarting the EA resets that in-memory high-water mark. The server does not authenticate EA poll requests, so keep it on loopback or behind a trusted private network.
