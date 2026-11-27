# UniversalAI Trading System

Independent MT5 signal service, account controls, historical research tools, mobile dashboard, and MQL5 execution adapter. This repository has no dependency on PHANTOM-TRADERS or any other trading project.

## Components

| File | Responsibility | Interface |
| --- | --- | --- |
| `brain.py` | MT5 M15/H1 data, quantitative/SMC gates, Groq and Gemini decisions, accounts and license API | Pure engine functions and FastAPI JSON routes |
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

Set `ADMIN_PIN`, optionally `GROQ_API_KEY` and `GEMINI_API_KEY`, and select a Groq-supported `GROQ_MODEL` (default `openai/gpt-oss-20b`). Set `MT5_SYMBOLS` to exact Market Watch names. Do not commit `.env`.

Start the server with `python brain.py`; it binds `127.0.0.1:8000` by default. Open `http://127.0.0.1:8000/`, activate a license with the MT5 login, save account risk settings, then press START. Use the Admin button and `ADMIN_PIN` to issue keys. Without a configured provider, qualified setups fail closed as WAIT.

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

Both in-sample (2020–2023) and out-of-sample (2024–2025) statistics are saved in ignored local files under `data/research_2020_2025/`. Every OOS pair with PF >= 1.25 and at least 80 trades receives a 1,000-order reshuffle Monte Carlo at 0.75% risk, with 5% daily and 10% peak-to-trough total drawdown stops. All such rows go in `monte_carlo_qualified`; the live `approved` list additionally requires ruin probability <= 5% and 95th-percentile drawdown <= 10% (both configurable CLI flags). The dashboard displays only this stricter list and `brain.py` fails closed for symbols/strategies absent from it. The LLM receives the current approved setup and indicator values as a final sanity check. For Strategy D live use, the matching XGBoost model JSON must also be present under `data/research_2020_2025/models/`.

## Research validation audit

See [`docs/research_validation_audit.md`](docs/research_validation_audit.md) for the independent Monte Carlo, backtester, XGBoost, walk-forward, and cost-model audit. The audit's reproduced and corrected outputs live under ignored `data/research_audit_2026_11/`; they do not overwrite the original research reports or approval file. In particular, Monte Carlo “ruin” is the probability of breaching either the 5% daily or 10% total drawdown limit, not a probability of account equity reaching zero. The audit keeps `approved_strategies.json` empty until existing approval requirements pass.

## Deterministic strategy module

`strategies.py` provides `session_orb_retest` and `ema_trend_pullback` for callers that pass completed OHLC bars. Both return `BUY`, `SELL`, or fail-closed `WAIT` dictionaries and do not route trades. `PairPerformanceFilter` is an in-memory research filter. `research_lab.py` exports its results to `performance_filter_registry.json` and `.csv`; this registry is review-only and never changes `approved_strategies.json` or the live `brain.py` strategy allow-list.

Costs use exported bar spreads, adverse slippage (2 points by default), and a default $7 round-turn commission per standard lot when MT5 tick-value metadata is available; otherwise the configurable commission-point fallback applies. Historical tick values/contracts may differ from historical broker terms, and OHLC bars cannot reproduce queue position, swaps, financing, liquidity gaps, or partial fills. Review the full run metadata and drawdowns before enabling an approved strategy. `xgboost` is listed in `requirements.txt`; if it is unavailable, strategies A–C still run and D is explicitly marked unavailable rather than silently replaced with another classifier.

## Verification

Run `python -m pytest`. Tests use synthetic fixtures and temporary SQLite storage; they do not claim profitability, Groq connectivity, or live order placement. Test EA behavior on a demo terminal before considering any funded account. The server's EA poll route is not protected by an additional network token; license validation and loopback/private network deployment are required.


By default, `research_lab.py` writes approval candidates to `<output-dir>/approved_candidates.json`. Use `--approved-path` to explicitly select a different destination.

Run `python evaluate_deterministic.py` to evaluate the rolling ORB and EMA pullback strategies across both historical M15 CSV directories. It prints JSON lines per file, strategy, and cost scenario, including drawdown and trade durations in bars. Use `--pattern` and `--since` to limit the run.

## Mobile paper account control

Run the paper API with `python -m uvicorn paper_control:app --host 127.0.0.1 --port 8010`. Its OpenAPI schema is at `/openapi.json`. Use HTTPS through the deployment reverse proxy for mobile access. The control service uses its own paper database and has no order-sending broker adapter.

Set `PAPER_ACCOUNTS_JSON` in the server environment. Each key is a mobile account identifier. For example:

```json
{"desk1":{"profile":"prop_firm_core","starting_balance":100000,"control_token_env":"DESK1_MOBILE_TOKEN","feed_token_env":"DESK1_FEED_TOKEN"}}
```

Set the two referenced variables to different private bearer tokens. Each account and role requires a unique token. The initial template is `three_setup_portfolio.json`. Future profiles can specify a separate `config` path with a matching profile name and paper mode enabled. State, locks, fills and setting overrides persist by account in SQLite. Set `PAPER_STATE_DB` to override the database path. Later template edits apply to newly provisioned accounts. PATCH updates apply to existing accounts immediately.

Run one market-data process per account/terminal:

```powershell
python paper_mt5_bridge.py --account desk1 --terminal "C:\path\terminal64.exe" --login DEMO_LOGIN --server DEMO_SERVER --feed-token-env DESK1_FEED_TOKEN
```

Use separate terminal instances for different broker logins. The bridge checks the login and server, polls for changed ticks every 250 ms, and periodically sends closed MT5 bars. AI calls run outside the tick-processing and SQLite risk transactions. With no usable LLM response, the bias endpoint still returns deterministic alignment and sets AI commentary to null. Session levels use observed M15 candles during fixed UTC Asia 00�06, London 07�13 and NY 13�17 windows. The terminal supplies the H4 and daily candle boundaries.

| Role | Request | Purpose |
|---|---|---|
| Mobile | GET /accounts/{account}/status | Equity, balance, floating P&L, quote timestamps, positions, locks, configuration and news status |
| Mobile | GET /accounts/{account}/history | Closed paper trades and exit reasons |
| Mobile | GET /market-bias?account={account} | Direction, completed H4/D1 EMA alignment, liquidity levels and optional AI commentary |
| Mobile | PATCH /accounts/{account}/settings | Update risk_pct, daily_loss_cap_pct, max_dd_cap_pct and/or setups (symbol, strategy, enabled) |
| Mobile | POST /accounts/{account}/commands/FLATTEN_ALL_AND_PAUSE | Close paper positions at their latest stored liquidation quotes and persist an emergency lock |
| Feed | POST /accounts/{account}/calendar | Supply covered UTC epoch start/end and events with timestamp, currency and impact |
| Feed | POST /accounts/{account}/paper-intents | Submit an upstream strategy-confirmed id, symbol, strategy, side, stop and target |
| Feed | POST /accounts/{account}/quotes or /accounts/{account}/bars/{symbol} | Publish verified broker quote/specification or closed market history |

Send `Authorization: Bearer <account-role-token>`. Mobile clients can also poll `GET /accounts/{account}/alerts` for accepted signals, setup IDs, entry, stop, first milestone/target and shadow size. Read history for partial exits and status for the current stop.

### Strategy runner

Use the API and bridge commands above. The bridge discovers symbols via authenticated `GET /accounts/{account}/feed-config`. Each M5 boundary sends closed M5/M15/H1/H4/D1 history to the bars endpoint. H1 supplies ATR and EMA context, H4/D1 alignment gates entries. Only the latest signal close within 30 seconds is eligible. Missed entries are skipped on reconnect. Stable setup/bar IDs deduplicate fills.

| Setup | Active rule |
|---|---|
| XAUUSD_ORB | 15-minute range, M5 close confirmation, midpoint stop and 1.5R target. The earlier fold profile `be/30m` uses a 30-minute range and moves to breakeven after a completed M5 reaches 1R. |
| XAUUSD_EMA_VWAP | M15 session VWAP and EMA pullback with H1 EMA-200. Realize half at the first liquidation quote crossing 1R and trail the rest two ATRs from completed M15 closes. |
| GER40_EMA_VWAP | Same pullback detector, fixed 2R target. |

Gold London starts at 07:00 UTC and hands off to NY at 09:30 ET. Gold NY ends at 17:00 UTC. Germany 40 runs 07:00-15:30 UTC. The first available liquidation quote after the session ends closes its paper positions. Paper results use observed quotes, exclude broker fees and can differ from the bar replay.

### Isolated profiles

Set each account record's `profile` and `config` fields in `PAPER_ACCOUNTS_JSON`, using separate control/feed tokens and terminal connections. Start the server from the repository root for relative template paths.

| Profile | Template | Behavior |
|---|---|---|
| prop_firm_core | three_setup_portfolio.json | Automatic three-setup paper book |
| prop_firm_matchtrader | config/prop_firm_matchtrader.json | Signal-only MatchTrader alerts, market bias and shadow equity. Manual broker fills are tracked separately by the trader. |
| deriv_synthetics_mt5 | config/deriv_synthetics_mt5.json | USD terminal, 24/7 eligibility, fixed-dollar risk and daily loss lock, news disabled. The sample external synthetic setup starts disabled. |

For Deriv, enable the desired setup via PATCH settings and submit strategy-confirmed intents via the feed endpoint. `risk_amount_usd` and `daily_loss_cap_usd` are hot-reloadable on dollar-risk profiles. The bridge checks terminal currency. All templates use `paper_mode: true`; the MT5 connector receives market data and the execution ledger simulates fills. No live order switch is exposed.

### Calendar and risk

The bridge calls `POST /accounts/{account}/calendar/refresh` at startup and every 15 minutes. The API fetches Forex Factory weekly JSON, validates its week, and caches events in the account database. High-impact USD/EUR news blocks entries through both 15-minute boundaries. Fetch failure preserves the cache. Set `news.offline_fallback` to `block` (default) or `allow_paper` for offline testing. Known events still block in fallback mode. A feed client can also POST its own covered calendar.

Daily limits use UTC day-opening marked equity. Total drawdown uses peak marked equity. New quotes synchronously update positions and enforce liquidation/locks. Daily locks reset on the next UTC day. Emergency and total-drawdown locks persist.

### Owner licenses, broker enrollment and notifications

Run the mobile API with `python -m uvicorn paper_control:app --host 127.0.0.1 --port 8010`. Use HTTPS for remote clients. This mobile license store shares the paper-account SQLite database and transaction boundary. Legacy brain/EA licenses belong to the separate brain service.

Set a private `OWNER_MASTER_KEY` in the API environment. Send it as an Authorization bearer token only to the owner endpoints. In `PAPER_ACCOUNTS_JSON`, provision the engine account/profile and bridge feed-token environment variable first. A license is issued for that specific engine account. Keep owner, bridge and any legacy control tokens off licensed client devices.

| Owner request | Purpose |
|---|---|
| POST /admin/licenses | JSON fields `account`, `scope`, optional timezone-aware ISO `expires_at`. Returns the key once. SQLite stores its SHA-256 digest. |
| GET /admin/licenses or GET /admin/licenses/{id} | Inspect scope, expiration, revocation and bindings without exposing the bearer key. |
| POST /admin/licenses/{id}/reset | Release device and broker binding and invalidate prior enrollment and push registration. |
| POST /admin/licenses/{id}/revoke | Revoke access and disable enrollment. |

Activate with `POST /licenses/activate`, Authorization bearer license key, and JSON `device_id`. EA_ONLY and FULL_ACCESS additionally require string `mt5_account_id` and exact `mt5_server`. Supply `X-Device-ID` alongside the bearer key on subsequent licensed requests. Device identity is client-asserted, so protect the key using platform secure storage. SIGNALS_ONLY grants bias, alerts and notification access. EA_ONLY grants account controls, status, history and registration. FULL_ACCESS grants both. Existing operator control/feed credentials retain their respective roles.

For an EA-capable license, POST the following to `/ea/register-and-start`:

```json
{"mt5_account_id":"123456","mt5_server":"Deriv-Demo","risk_mode":"percent","risk_value":0.25,"daily_loss_cap":2.0,"max_dd_cap":5.0,"demo_live_execution":false}
```

`percent` uses percentages; `fixed_usd` uses USD trade risk and daily cap on a USD profile. Maximum drawdown is always a percentage. Enrollment configures the account; it does not log into MT5 or launch a terminal. Run exactly one bridge/terminal per broker account using the bridge command above. Leave `demo_live_execution` false for paper-only use. Signal-only profiles reject broker dispatch. Explicit true opts a connected matching bridge into broker orders. Both MT5 demo trade mode and a server name containing Demo are checked. The operator-only `ALLOW_REAL_ACCOUNTS=true` override must be present on the bridge for other accounts and on the API for non-Demo server names. Default is absent/false.

Broker mode uses an account-specific magic number, symbol alias/suffix resolution, current quotes, stop-based broker profit and volume calculations, margin/order precheck and a durable send-attempt journal (`MT5_EXECUTION_DB`). Ambiguous symbols, minimum-lot violations and existing symbol positions block entry. Keep the journal across restarts. Market orders attach a protective SL; fixed-target profiles attach TP. Managed exits use the bridge for half-close at 1R, breakeven and bar-derived ATR stop updates. Keep the bridge, terminal and API online. A lost broker acknowledgement consumes that entry attempt to avoid duplicate orders. Broker equity limits include floating losses. Liquidation targets positions tagged by this adapter. API account metrics and history remain the shadow ledger; confirm broker fills/P&L in MT5. Revocation, reset, expiry or emergency pause removes desired broker exposure on the next successful synchronization. Previously documented paper-only behavior is the default prior to explicit enrollment.

Register a signal-capable activated device via `POST /notifications/register-device` with JSON `{"token":"<FCM registration token>"}`. Read durable per-account alerts with `GET /notifications/events?after=0`; persist the highest returned sequence for the next poll. Local polling works without Firebase. For FCM HTTP v1 delivery, install requirements, set `FCM_PROJECT_ID` and provision Google Application Default Credentials with messaging permission (for example via `GOOGLE_APPLICATION_CREDENTIALS`). New alerts and their delivery jobs commit with the account update. Failed sends retry with backoff; clients should deduplicate by event_id.
