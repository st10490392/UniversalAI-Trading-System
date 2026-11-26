"""Offline, chronologically nested walk-forward tests for deterministic strategies."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from research_lab import ROOT, _causal_spreads, _trade_one, load_history, standard_qualification, standard_trade_stats
from strategies import _atr

REPORT_ROOT = ROOT / "data/research_2020_2025"
OUT = REPORT_ROOT / "strategy_backtest_2026"


ANCHOR_HOURS = {"EURUSD": 8, "GBPUSD": 8, "USDJPY": 0, "XAUUSD": 13}
WINDOW = 80


def signal_components(d, mode, spreads):
    atr = _atr(d, 14)
    spread = spreads * d.point_size
    # An 80-bar completed window must pass the same price and timestamp checks.
    o, h, l, c = (d[k] for k in ("open", "high", "low", "close"))
    invalid = (~np.isfinite(d[["open", "high", "low", "close"]])).any(axis=1)
    invalid |= (d[["open", "high", "low", "close"]] <= 0).any(axis=1)
    invalid |= (h < np.maximum(o, c)) | (l > np.minimum(o, c))
    steps = d.time.diff().dt.total_seconds()
    gap_window = steps.rolling(WINDOW - 1)
    eligible = invalid.rolling(WINDOW).sum().eq(0)
    eligible &= gap_window.max().le(np.maximum(gap_window.median() * 6, 345600))
    eligible &= (d.point_size > 0) & np.isfinite(spread) & (atr > 0)
    volatility_ok = pd.Series(True, index=d.index)
    if mode == "ema_trend_pullback":
        def ema(period, offset):
            decay = 1 - 2 / (period + 1)
            count = WINDOW - offset
            weights = (1 - decay) * decay ** np.arange(count)
            weights[-1] = decay ** (count - 1)
            return pd.Series(np.r_[np.full(WINDOW - 1, np.nan), np.convolve(c.to_numpy()[:len(c)-offset], weights, mode="valid")])
        fast, slow = ema(20, 0), ema(50, 0)
        previous_fast = ema(20, 1)
        available = (fast - slow).abs() >= .2 * atr
        buy = available & (fast > slow) & (fast > ema(20, 5)) & (slow > ema(50, 5))
        buy &= (l.shift(1) <= previous_fast) & (c > o) & (c > fast)
        sell = available & (fast < slow) & (fast < ema(20, 5)) & (slow < ema(50, 5))
        sell &= (h.shift(1) >= previous_fast) & (c < o) & (c < fast)
    else:
        if mode == "session_orb_retest_anchored":
            hour = ANCHOR_HOURS[d.attrs["symbol"]]
            anchor = d.time.dt.normalize() + pd.Timedelta(hours=hour)
            positions = pd.Series(np.arange(len(d)), index=d.time)
            anchor_position = anchor.map(positions)
            eligible &= (np.arange(len(d)) - anchor_position).between(8, WINDOW - 1)
            upper = (anchor + pd.Timedelta(minutes=75)).map(pd.Series(h.rolling(6).max().array, index=d.time))
            lower = (anchor + pd.Timedelta(minutes=75)).map(pd.Series(l.rolling(6).min().array, index=d.time))
            eligible &= anchor_position.eq((anchor + pd.Timedelta(minutes=75)).map(positions) - 5)
        else:
            upper, lower = h.shift(3).rolling(6).max(), l.shift(3).rolling(6).min()
        available = upper - lower >= .5 * atr
        volatility_ok = upper - lower >= atr
        buy = available & (c.shift(2) > upper) & l.shift(1).between(lower, upper) & (c > o) & (c > upper)
        sell = available & (c.shift(2) < lower) & h.shift(1).between(lower, upper) & (c < o) & (c < lower)
    return buy & eligible, sell & eligible, atr, spread, volatility_ok


@dataclass(frozen=True)
class Variant:
    rr: float
    atr_filter: bool
    ema200: bool


# Compare RR with and without both refinements. Isolate each refinement at 1.5R.
VARIANTS = tuple(Variant(rr, active, active) for active in (False, True) for rr in (1.5, 2., 3.)) + (
    Variant(1.5, True, False), Variant(1.5, False, True)
)


def stop_levels(data, components, variant):
    buy, sell, atr, spread, volatility_ok = components
    bid = data.close
    ask = bid + spread
    long_stop = data.low.shift(1) - spread
    short_stop = data.high.shift(1) + spread
    if variant.atr_filter:
        # ORB range >= 1 ATR. Both setups buffer the structural stop by
        # .1 ATR and require at least .5 ATR or 3 spreads of initial risk.
        minimum_risk = np.maximum(.5 * atr, 3 * spread)
        long_stop = np.minimum(long_stop - .1 * atr, ask - minimum_risk)
        short_stop = np.maximum(short_stop + .1 * atr, bid + minimum_risk)
        buy = buy & volatility_ok
        sell = sell & volatility_ok
    if variant.ema200:
        regime = bid.ewm(span=200, adjust=False, min_periods=200).mean()
        buy = buy & (bid > regime) & (regime > regime.shift(5))
        sell = sell & (bid < regime) & (regime < regime.shift(5))
    buy = buy & (long_stop > 0) & (ask > long_stop)
    sell = sell & (short_stop > bid) & np.isfinite(short_stop)
    return np.select([buy, sell], [1, -1], default=0), np.select([buy, sell], [long_stop, short_stop], default=np.nan)


def execution_columns(data, spreads):
    names = ("time", "open", "high", "low", "close", "point_size", "tick_size", "tick_value")
    arrays = {name: data[name].array if name == "time" else data[name].to_numpy() for name in names}
    arrays["spread_points"] = spreads.to_numpy()
    return SimpleNamespace(**{name: SimpleNamespace(iloc=values) for name, values in arrays.items()})


def walk_segment(columns, sides, stops, *, lo, hi, rr, slippage, commission):
    trades = []
    next_free = lo
    for i in np.flatnonzero(sides[lo:hi-1]) + lo:
        if i < next_free:
            continue
        trade = _trade_one(columns, int(i), int(sides[i]), float(stops[i]), np.nan, end_idx=hi,
                           breakeven=False, slippage_points=slippage, commission_per_lot=7 * commission,
                           commission_points=commission, absolute_stop=True, reward_risk=rr)
        if trade is not None:
            trades.append(trade)
            next_free = i + trade["bars_held"] + 1
    return trades


def paired_stats(scenarios, groups):
    return tuple(standard_trade_stats([trade for j in groups for trade in scenario[j]]) for scenario in scenarios)


def selection_score(train, validation):
    """Choose on earlier data only, prioritizing stable positive stressed expectancy."""
    bt, st = train
    bv, sv = validation
    eligible = (min(bt["trades"], st["trades"]) >= 50 and min(bv["trades"], sv["trades"]) >= 20
                and min(bt["average_r"], st["average_r"], bv["average_r"], sv["average_r"]) > 0
                and max(bt["max_drawdown_pct"], st["max_drawdown_pct"], bv["max_drawdown_pct"], sv["max_drawdown_pct"]) <= 20)
    return eligible, min(st["average_r"], sv["average_r"]), sv["average_r"]


def evaluate_mode(data, spreads, execution, bounds, mode):
    components = signal_components(data, mode, spreads)
    trials = []
    for variant in VARIANTS:
        sides, stops = stop_levels(data, components, variant)
        trades = [[walk_segment(columns, sides, stops, lo=lo, hi=hi, rr=variant.rr,
                                slippage=slippage, commission=commission)
                   for lo, hi in zip(bounds, bounds[1:])]
                  for columns, slippage, commission in execution]
        trials.append(trades)
    folds = []
    selected = [[], []]
    for test in (2, 3):
        development = [(paired_stats(trades, range(test-1)), paired_stats(trades, [test-1])) for trades in trials]
        best = max(range(len(trials)), key=lambda i: selection_score(*development[i]))
        train, validation = development[best]
        eligible = selection_score(train, validation)[0]
        for scenario, trades in enumerate(trials[best]):
            selected[scenario].extend(trades[test])
        folds.append({"parameter_set": asdict(VARIANTS[best]), "development_pass": eligible,
                      "validation_from": data.time.iloc[bounds[test-1]].isoformat(),
                      "test_from": data.time.iloc[bounds[test]].isoformat(),
                      "train": train, "validation": validation, "test": paired_stats(trials[best], [test])})
    baseline, stress = map(standard_trade_stats, selected)
    qualification = standard_qualification(baseline, stress)
    stable = [asdict(variant) for variant, trades in zip(VARIANTS, trials)
              if all(standard_qualification(*paired_stats(trades, groups))["standard_pass"]
                     for groups in ([0, 1], [2, 3]))]
    comparisons = [{"parameter_set": asdict(variant), "in_sample": paired_stats(trades, [0, 1]),
                    "out_of_sample": paired_stats(trades, [2, 3])}
                   for variant, trades in zip(VARIANTS, trials)]
    return baseline, stress, qualification, folds, stable, comparisons


def evaluate_source(path):
    data = load_history(path)
    symbol = data.attrs["symbol"]
    spreads = _causal_spreads(data)
    execution = [(execution_columns(data, spreads * multiplier), slip, cost)
                 for multiplier, slip, cost in ((1., 2., 1.), (2., 5., 1.5))]
    # Holdout blocks contain approximately 20% of completed bars each.
    bounds = [0, len(data)*2//5, len(data)*3//5, len(data)*4//5, len(data)]
    modes = ["session_orb_retest_rolling", "ema_trend_pullback"]
    if symbol in ANCHOR_HOURS:
        modes.append("session_orb_retest_anchored")
    rows = []
    for mode in modes:
        baseline, stress, qualification, folds, stable, comparisons = evaluate_mode(data, spreads, execution, bounds, mode)
        development_pass = all(f["development_pass"] for f in folds)
        rows.append({
            "symbol": symbol, "strategy": mode, "dataset": path.parent.name, "source": path.name,
            "oos_start_utc": data.time.iloc[bounds[2]].isoformat(), "last_bar_utc": data.time.iloc[-1].isoformat(),
            "oos_trades": baseline["trades"], "stress_trades": stress["trades"],
            "win_rate_pct": baseline["win_rate_pct"], "baseline_pf": baseline["profit_factor"],
            "stressed_pf": stress["profit_factor"], "average_r": baseline["average_r"], "stressed_average_r": stress["average_r"],
            "max_drawdown_pct": max(baseline["max_drawdown_pct"], stress["max_drawdown_pct"]),
            "development_pass": development_pass, "oos_pass": qualification["standard_pass"],
            "research_pass": development_pass and qualification["standard_pass"], "live_approval": False,
            "oos_failures": qualification["failures"], "selected_folds": folds,
            "stable_fixed_variants": stable, "fixed_variant_comparisons": comparisons,
        })
    return rows


def write_reports(rows, output):
    groups = defaultdict(list)
    for row in rows:
        groups[row["symbol"], row["strategy"]].append(row)
    for observations in groups.values():
        passes = all(row["research_pass"] for row in observations)
        worst_pf = min(row["stressed_pf"] if row["stressed_pf"] is not None else -1. for row in observations)
        for row in observations:
            row.update(all_sources_pass=passes, worst_source_stressed_pf=worst_pf)
    rows.sort(key=lambda r: (not r["all_sources_pass"], -r["worst_source_stressed_pf"],
                             r["symbol"], r["strategy"], r["dataset"]))
    with (output / "walkforward_standard.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, list) else v for k, v in row.items()})
    candidates = [
        {"symbol": symbol, "strategy": strategy, "live_approval": False, "sources": [
            {k: v for k, v in row.items() if k != "fixed_variant_comparisons"} for row in observations
        ]}
        for (symbol, strategy), observations in groups.items() if all(row["research_pass"] for row in observations)
    ]
    (output / "approved_candidates.json").write_text(json.dumps({
        "research_only": True, "live_approval": False, "approved": [], "candidates": candidates,
        "dataset_evaluations": len(rows), "pair_count": len(groups),
    }, allow_nan=False, indent=2), encoding="utf-8")
    return len(groups), len(candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pattern", default="*_M15*.csv")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(REPORT_ROOT.resolve()):
        parser.error("--output-dir must be inside the ignored research directory")
    paths = sorted(p for directory in ("history_2020_2025", "historical_12m")
                   for p in (ROOT / "data" / directory).glob(args.pattern))
    if not paths or args.workers < 1:
        parser.error("matching inputs and at least one worker are required")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(evaluate_source, paths), 1):
            rows.extend(result)
            if i % 15 == 0 or i == len(paths):
                print(f"Evaluated {i}/{len(paths)} files", flush=True)
    pairs, candidates = write_reports(rows, output)
    print(f"{len(rows)} dataset evaluations, {pairs} pairs, {candidates} research candidates")


if __name__ == "__main__":
    main()
