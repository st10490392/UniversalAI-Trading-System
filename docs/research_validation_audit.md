# Quantitative research validation audit

## Scope and preserved evidence

The audit ran against the 63 local MT5 demo-symbol CSVs (3,313,301 M15 bars) and preserved the prior `data/research_2020_2025/` outputs, root approval file, and historical data. New reproducible output is isolated under ignored `data/research_audit_2026_11/`. The earliest available bars are in 2022; most synthetic histories begin in 2024. No 2020–2021 or missing 2022–2023 bars have been manufactured.

## Monte Carlo methodology

In `df59bb5`, each of 1,000 iterations permuted the candidate's net trade R outcomes without replacement using NumPy's seeded generator. It retained the original daily trade counts, applied `equity *= max(0, equity * (1 + R * .75 / 100))` in effect (0.75% of current equity per R), tracked peak-to-trough drawdown, skipped further trades for a day after a 5% daily breach, and stopped the path after a 10% peak-to-trough breach. A path was counted as “ruined” when **either** event occurred. It did not mean equity reached zero. Equity-zero probability is now reported separately.

The audit dynamically loaded the exact committed implementation, recreated the three signal streams, and reproduced the persisted results with the same seeds:

| Pair | Seed | Raw sequence trades reshuffled | Reported OOS executed trades | Daily only | Total only | Both | Either-limit (“ruin”) | Equity zero | p95 max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Range Break 200 / Trend Momentum | 2807 | 956 | 956 | 0.1% | 68.2% | 0.0% | 68.3% | 0.0% | 10.6758% |
| Range Break 200 / ICT / SMC | 2808 | 1,078 | 279 | 0.0% | 99.5% | 0.0% | 99.5% | 0.0% | 10.6919% |
| Volatility 250 (1s) / Trend Momentum | 4607 | 913 | 150 | 0.0% | 0.2% | 99.8% | 100.0% | 0.0% | 37.1213% |

The OOS trade count and PF are calculated from the original chronological run after risk stops; Monte Carlo reshuffles the unfiltered pre-stop outcomes and reapplies stops under each permutation. Thus the candidates' qualifying trade count/PF and Monte Carlo sample do not describe the same executed sequence. The 68.3%, 99.5%, and 100% values are path frequencies, rounded by increments of 0.1 percentage point, not account-zero probabilities.

The old p95 figures are max drawdown percentiles over those same paths. A path may breach a 5% daily limit while total peak-to-trough drawdown remains below 10%; a total-limit breach is at least 10% by definition. Consequently p95 drawdown near 10% does not determine the union probability: it is a quantile, not a breach frequency, and it omits daily-only breaches as a separate cause. A breaching trade is applied before stop handling, so realized p95 can overshoot the limit. The new results include daily-only, total-only, either, both, and equity-zero probabilities separately. Historical day grouping uses UTC timestamps; without the broker's historical server-time offset, the exact prop-firm daily reset boundary cannot be verified.

Trade outcomes are R-multiples, so a large adverse R can mathematically take equity to zero; all three reproduced candidate samples have no individual R outcome sufficient to do that at 0.75% risk. In the new simulation, zero-equity is explicitly counted after compounding. The formula is multiplicative on current equity, not a fixed initial-balance risk amount.

## Backtest integrity findings

- CSV timestamps are parsed as UTC bar-open timestamps, sorted, and deduplicated. Signals use the completed bar and enter at the next bar open. Current MT5 polling uses completed candles. Missing intervals are not forward-filled; the audit reports gaps and does not claim continuous coverage.
- Indicator windows are causal. Strategy B's Asian range used a full-session daily transform. The Judas rule only admits New York 07:00–10:00 signals, after the defined Asian range (before 07:00 NY) has completed, so the audit found no reachable signal-time leak from this transform. It has nevertheless been replaced with prior-bar cumulative Asian highs/lows per New York day, ensuring the intermediate series itself is causal.
- Stop and target checks use intrabar high/low; if both could be touched in one OHLC bar, stop wins. Gap-through fills use the worse opening price. Break-even is applied only after the bar's favorable extreme and is not assumed to have triggered inside that same bar because OHLC ordering is unknown.
- Signal at bar `i` fills no earlier than bar `i+1`. One position is simulated at a time; signals are skipped until the previous simulated position exits. Delayed-entry stress scenarios likewise avoid overlap.
- Long entry buys ask and exits bid; short entry sells bid and covers ask. A confirmed short timeout bug (covering at `close - spread`) was corrected to `close + spread`. Nonzero reported bar spread is used; zero/missing values now use a past-only expanding median, preceded by an explicit 20-point fallback. No future spread is used to impute an earlier bar.
- Fallback commission is documented as total round-turn points and is charged once; a confirmed erroneous double charge was removed. Default is $7 per standard lot round turn if current tick-size/value snapshots are positive.
- Costs remain approximate: 18.90% of the harvested spread observations are zero/missing; the current fallback is not a recovered historical quote. Commission history is absent. Tick/point size/value in each historical row are current terminal contract snapshots repeated across the series, not historical specifications. Currency conversion, contract changes, and exact broker lot sizing cannot be validated from these files. The simulator operates in R-multiples and does not claim exact account-currency or lot PnL. Instruments with different contract currency or CFD formulas need historical specifications before results are treated as execution-grade.

## XGBoost leakage audit

Feature rows are taken at signal bar `i`, with rolling features using current/past bars only. Training candidate signals are restricted to timestamps before 2024-01-01; labels enter next bar and are truncated at that cutoff, preventing training trades from consuming 2024 bars. XGBoost is fit once using only that training matrix; it has no preprocessing fit on test observations. Model threshold is fixed at 0.55. 2024 is reported as validation-only and 2025 as final test; no threshold tuning is done in this run. This is a chronological holdout but not nested walk-forward retraining.

Only instruments for which the existing research JSON says `trained on 2020-2023 only`, with at least 40 labelled trades and both label classes, have usable models. See `data/research_audit_2026_11/model_inventory.csv` for exact symbols and train-row counts. Limited pre-2024 coverage makes this inventory incomplete by construction. The D strategy's in-sample performance row is a raw breakout baseline, not fitted-model performance.

## Walk-forward and stress methodology

All strategies are evaluated on fixed chronological half-year windows: 2024-H1, 2024-H2, 2025-H1, and 2025-H2. Rules A-C are frozen rules, with no fitting. For D, 2024 windows are validation reporting and 2025 windows are final-test reporting from the model trained through 2023. Each window reports actual bar count, observed calendar months, trade count, trades/month, win rate, PF, max drawdown, return, monthly return, and Sharpe. Empty periods are labelled `no_history`; unavailable D models are labelled `model_unavailable`.

Stress cases rerun the 2024–2025 baseline with 1.5x and 2x spread, 2x slippage, 1.5x commission, all costs doubled, fixed 15/30/60 minute entry delays, reproducibly randomized 0–60 minute delay, and .5/.75/1/1.5% equity risk. The separate loss-streak grid computes compound drawdown from 3/5/7/10/15 consecutive full-risk losses, with same-day daily-limit and one-per-day total-limit views. These are deterministic stress scenarios, not forecasts. Random delays model execution latency only; they are not intended to disguise correlated execution.

## Approval

`approved_strategies.json` remains empty. The original candidates all fail the fixed 5% daily / 10% total risk gate, and the new evaluation does not lower those limits. No strategy was activated. Generated audit reports are ignored local artifacts and do not replace prior outputs.

## Limitations

Available market data is a broker-specific demo export, not a complete 2020–2025 panel. The current audit cannot establish historical instrument contract terms, historical commissions, tick-level intrabar ordering, market impact, queueing, or live MT5 execution quality. In particular, a zero-spread observation is not proof of zero executable spread. Walk-forward summaries on the supplied partial history are descriptive and do not establish robust forward profitability.
