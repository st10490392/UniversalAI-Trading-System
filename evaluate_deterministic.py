"""Evaluate completed-bar strategies on local M15 CSVs without live routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_lab import ROOT, _causal_spreads, load_history, simulate, standard_qualification, standard_trade_stats
from strategies import ema_trend_pullback, session_orb_retest, validate_signal


def build_research_signals(data: pd.DataFrame, strategy, lookback: int):
    side = np.zeros(len(data), dtype=np.int8)
    stops = np.full(len(data), np.nan)
    targets = np.full(len(data), np.nan)
    spreads = _causal_spreads(data).to_numpy()
    points = data.point_size.to_numpy()
    candidates = data.close.ne(data.open)
    if strategy is session_orb_retest:
        # Both rolling ORB branches require two closes beyond the same range.
        upper = data.high.shift(3).rolling(6).max()
        lower = data.low.shift(3).rolling(6).min()
        breakout = data.close.shift(2)
        candidates &= ((data.close > upper) & (breakout > upper)) | ((data.close < lower) & (breakout < lower))
    bars = data[["time", "open", "high", "low", "close"]]
    for i in np.flatnonzero(candidates.to_numpy()):
        if i + 1 < lookback or not np.isfinite(points[i]) or points[i] <= 0:
            continue
        window = bars.iloc[i + 1 - lookback:i + 1]
        signal = validate_signal(strategy(window, point_value=points[i], spread_points=spreads[i]))
        if signal["command"] in ("BUY", "SELL"):
            side[i] = 1 if signal["command"] == "BUY" else -1
            stops[i], targets[i] = signal["sl"], signal["tp"]
    return side, stops, targets, side


def evaluate_file(path: Path, lookback: int = 80, since: str | None = None):
    data = load_history(path)
    if since:
        data = data.loc[data.time >= pd.to_datetime(since, utc=True)].reset_index(drop=True)
    if len(data) < lookback + 1:
        raise ValueError(f"{path.name} has too few bars for the evaluation window")
    start = data.time.iloc[0].date().isoformat()
    end = (data.time.iloc[-1].normalize() + pd.Timedelta(days=1)).date().isoformat()
    for strategy in (session_orb_retest, ema_trend_pullback):
        signals = build_research_signals(data, strategy, lookback)
        results = []
        for scenario, spread, slip, commission_scale in (
            ("baseline", 1.0, 2.0, 1.0),
            ("combined_adverse_costs", 2.0, 5.0, 1.5),
        ):
            trades = simulate(data, signals, start=start, end=end, absolute_stops=True,
                              spread_multiplier=spread, slippage_points=slip,
                              commission_per_lot=7.0 * commission_scale,
                              commission_points=commission_scale)
            durations = [trade["bars_held"] for trade in trades]
            stats = standard_trade_stats(trades)
            results.append({
                "source": path.name, "symbol": data.attrs["symbol"], "strategy": strategy.__name__,
                "scenario": scenario, "start_utc": start, "end_exclusive_utc": end,
                "lookback_bars": lookback, "live_approval": False, **stats,
                "median_bars_held": float(np.median(durations)) if durations else None,
                "p95_bars_held": float(np.percentile(durations, 95)) if durations else None,
                "max_bars_held": max(durations, default=None),
            })
        qualification = standard_qualification(*results)
        for row in results:
            yield {**row, **qualification}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, action="append", help="CSV directory; repeat to add another")
    parser.add_argument("--pattern", default="*_M15*.csv", help="glob for selecting input files")
    parser.add_argument("--since", help="optional first UTC date to read into the evaluation window")
    parser.add_argument("--lookback", type=int, default=80)
    args = parser.parse_args()
    if args.lookback < 60:
        parser.error("--lookback must be at least 60")
    directories = args.data_dir or [ROOT / "data/history_2020_2025", ROOT / "data/historical_12m"]
    paths = sorted({path.resolve() for directory in directories for path in directory.glob(args.pattern)})
    if not paths:
        parser.error("no matching historical CSVs")
    for path in paths:
        for result in evaluate_file(path, args.lookback, args.since):
            print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
