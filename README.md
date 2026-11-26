# UniversalAI Trading System

Independent MT5 signal service, account controls, historical research tools, mobile dashboard, and MQL5 execution adapter. This repository has no dependency on PHANTOM-TRADERS or any other trading project.

## Components

| File | Responsibility | Interface |
| --- | --- | --- |
| `brain.py` | MT5 M15/H1 data, quantitative/SMC gates, Groq Qwen decisions, accounts and license API | Pure engine functions and FastAPI JSON routes |
| `backtest.py` | Historical M15 simulation with rolling indicators, structure, costs, and drawdown | `run_backtest(DataFrame, ...)` or CLI |
| `UniversalAI_EA.mq5` | License-bearing polling, local drawdown breaker, sizing, deduplication, execution | GET `/api/ea/poll/{account_id}/{symbol}` plus `X-License-Key` |
| `index.html` | Mobile PWA activation, account view/settings, license admin panel | License/device headers and admin PIN routes |

The API never submits orders. The EA executes signals and checks its circuit breaker locally. Accounts start stopped unless the owner enables them. `accounts.db`, `.env`, historical CSVs, logs, and compiled `.ex5` files are local runtime artifacts and excluded from Git.

## License and device binding

- Dashboard activation submits a license key, MT5 account login, and browser-generated UUID. First activation atomically binds the account and device. Another account or device receives HTTP 403.
- Dashboard APIs send `X-License-Key`, `X-License-Account-ID`, and `X-Device-ID`. EA polls send `X-License-Key`; the server requires an active, unexpired key bound to that account and previously claimed by a dashboard device. Invalid EA polls return `{"command":"UNAUTHORIZED"}`. Revocation takes effect on the next poll.
- Set a private `ADMIN_PIN` in `.env`. The Admin panel issues keys formatted `UAI-XXXX-XXXX-XXXX`, lists bindings, toggles access, and clears a device binding. The account binding remains when a device is reset.
- Device ID is a persistent browser-generated identifier, not hardware attestation. Protect the app with TLS and network access controls.

## Windows installation

Install 64-bit Python 3.11/3.12, MetaTrader 5, and MetaEditor. Sign in to the intended MT5 terminal manually. In PowerShell from this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `ADMIN_PIN`, optionally `GROQ_API_KEY`, and select a Groq-supported `GROQ_MODEL` (default `qwen/qwen3-32b`). Set `MT5_SYMBOLS` to exact Market Watch names. Do not commit `.env`.

Start the server with `python brain.py`; it binds `127.0.0.1:8000` by default. Open `http://127.0.0.1:8000/`, activate a license with the MT5 login, save account risk settings, then press START. Use the Admin button and `ADMIN_PIN` to issue keys. Without `GROQ_API_KEY`, qualified setups fail closed as WAIT.

In MT5, open **Tools → Options → Expert Advisors**, enable **Allow WebRequest for listed URL**, and add `http://127.0.0.1:8000`. Compile `UniversalAI_EA.mq5` in MetaEditor. Attach the EA to charts, set the license key input, check the account login, and enable Algo Trading only on an account you intend to use. For remote mobile access, keep the service behind TLS and a private firewall/reverse proxy; the default server bind is loopback.

## Azure setup

Run `.\deploy_azure.ps1` from PowerShell in the repository. It creates a virtual environment, installs dependencies, prompts securely for `ADMIN_PIN` and an optional Groq key, writes `.env`, and launches FastAPI hidden with logs under `logs/`. It leaves the host bound to loopback. MT5 terminal installation/sign-in, TLS, firewall, and EA attachment remain explicit deployment steps.

If `.env` does not exist and you need to add a Groq key later:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }; notepad .env
```

Set `GROQ_API_KEY=...` and restart the server. The repository includes `.env.example` with blank secret values.

## API contracts

- `POST /api/activate` body: `license_key`, `account_id`, `device_id`; binds the first account/device pair.
- User account routes require `X-License-Key`, `X-License-Account-ID`, and `X-Device-ID`.
- Admin routes `/api/admin/licenses/create`, `/api/admin/licenses`, `/api/admin/licenses/{key}/toggle`, and `/api/admin/licenses/{key}/reset-device` require `X-Admin-PIN`.
- EA route `GET /api/ea/poll/{account_id}/{symbol}` requires `X-License-Key` and accepts balance/equity in headers to keep them out of access-log URLs; invalid license returns `command=UNAUTHORIZED`.
- `/api/health`, dashboard shell, and activation are public. OpenAPI is at `/docs` on the local server.

## Quant/SMC decisions

- Quantitative gate: ATR14/ATR50 > 0.90, Kaufman ER20 > 0.25, with 50-close z-score blocks BUY above +2 and SELL below -2.
- H1 bias: EMA20/EMA50 aligned with two confirmed fractal HH/HL or LH/LL pivots.
- M15 structure: prior 20-bar liquidity sweep with close back inside, combined with a three-candle FVG. The sweep may occur in the four bars before the completed FVG.
- Forex/Gold use approximate UTC windows 07:00–10:00 and 12:00–16:00; fixed UTC windows do not track London/New York DST. Synthetic symbol names bypass session filters.
- Groq is called only after both gates pass. Output must match the setup side, confidence ≥0.75, directional stop/target, and minimum 2.5 reward/risk.

## 12-month MT5 history and backtests

The exporter requires the connected MT5 session to be a demo account. It discovers every matching visible symbol in Market Watch for Volatility, Boom/Crash, Step, Jump, Range Break, seven major FX pairs, gold/silver, and Nasdaq/GER30/US30 aliases. Broker naming varies; missing expected names are reported and skipped. It writes up to the previous 365 days of completed M15 data, including historical spread points and symbol point size:

```powershell
python tools/export_mt5_history.py
python tools/run_backtests.py
```

The batch summary reports win rate, profit factor, max drawdown, return, and trade count, and writes the complete comparison to `data/historical_12m/backtest_summary.csv`. It uses each CSV's spread/point values, 2 points of adverse slippage, and a 20-point fallback spread when historical spread is zero/missing. Signals are evaluated after the M15 candle closes; OHLC bars touching stop and target together assume stop first. The model uses one trade at a time and does not include swaps, latency, partial fills, or broker contract-value differences. Backtest metrics are historical simulations, not expected or guaranteed results.

Standalone CSV command:

```powershell
python backtest.py .\EURUSD_M15.csv --symbol EURUSD --risk-pct 1 --daily-dd 4 --total-dd 10 --spread-points 20 --slippage-points 2 --output trades.csv
```

CSV timestamps must be UTC. An `MT5` export from broker time must be converted before backtesting.

## 2020–2025 quantitative research lab

`fetch_history.py` connects to the already-running MT5 terminal and refuses to harvest unless the current account is a demo account. It scans visible Market Watch symbols for the configured Forex/gold, index aliases, and Deriv synthetic families, tags each CSV as `forex`, `index`, or `synthetic`, and streams half-year chunks to `data/history_2020_2025/`. The end date is exclusive `2026-01-01`; the terminal only returns bars the broker has made available, so `manifest.json` records the actual first and last bar for each symbol. Increase MT5's **Max bars in chart**, open the desired symbols, and allow the terminal to download history before running it again if the broker returns a shorter window.

```powershell
python fetch_history.py
python research_lab.py
```

The research runner evaluates four independent M15 families: (A) 12/26 EMA cross with RSI, ATR expansion and a +1.2R breakeven stop; (B) FX/index London and New York session Judas/FVG logic and session-free synthetic structural/breakout logic; (C) Bollinger/Keltner squeeze, 50-bar z-score and ER mean reversion; and (D) 20-bar volatility breakouts with an XGBoost classifier trained only on labelled 2020–2023 candidates and applied to 2024–2025 candidates. D's in-sample statistics show the raw breakout training-label pool, not fitted-model training accuracy. London/NY time windows use IANA time zones and handle daylight-saving transitions. Synthetic symbols never enter the session-filter path.

Both in-sample (2020–2023) and out-of-sample (2024–2025) statistics are saved in ignored local files under `data/research_2020_2025/`. Every OOS pair with PF >= 1.25 and at least 80 trades receives a 1,000-order reshuffle Monte Carlo at 0.75% risk, with 5% daily and 10% peak-to-trough total drawdown stops. All such rows go in `monte_carlo_qualified`; the live `approved` list additionally requires ruin probability <= 5% and 95th-percentile drawdown <= 10% (both configurable CLI flags). The dashboard displays only this stricter list and `brain.py` fails closed for symbols/strategies absent from it. Qwen receives the current approved setup and indicator values as a final sanity check. For Strategy D live use, the matching XGBoost model JSON must also be present under `data/research_2020_2025/models/`.

## Research validation audit

See [`docs/research_validation_audit.md`](docs/research_validation_audit.md) for the independent Monte Carlo, backtester, XGBoost, walk-forward, and cost-model audit. The audit's reproduced and corrected outputs live under ignored `data/research_audit_2026_11/`; they do not overwrite the original research reports or approval file. In particular, Monte Carlo “ruin” is the probability of breaching either the 5% daily or 10% total drawdown limit, not a probability of account equity reaching zero. The audit keeps `approved_strategies.json` empty until existing approval requirements pass.

Costs use exported bar spreads, adverse slippage (2 points by default), and a default $7 round-turn commission per standard lot when MT5 tick-value metadata is available; otherwise the configurable commission-point fallback applies. Historical tick values/contracts may differ from historical broker terms, and OHLC bars cannot reproduce queue position, swaps, financing, liquidity gaps, or partial fills. Review the full run metadata and drawdowns before enabling an approved strategy. `xgboost` is listed in `requirements.txt`; if it is unavailable, strategies A–C still run and D is explicitly marked unavailable rather than silently replaced with another classifier.

## Verification

Run `python -m pytest`. Tests use synthetic fixtures and temporary SQLite storage; they do not claim profitability, Groq connectivity, or live order placement. Test EA behavior on a demo terminal before considering any funded account. The server's EA poll route is not protected by an additional network token; license validation and loopback/private network deployment are required.
