"""Research-only walk-forward evaluation of session ORB on cash-market underlyings."""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research_lab import ROOT, _causal_spreads, _trade_one, load_history, standard_trade_stats
from strategies import _atr
from walkforward_standard import execution_columns

DATA = ROOT / "data"
OUTPUT = DATA / "research_2020_2025/fx_stocks_session_orb/walkforward_fx_stocks.csv"
FX = {"AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDJPY"}
METALS = {"XAUUSD", "XAGUSD"}
EQUITIES = {"Wall Street 30", "US Tech 100", "Germany 40"}


@dataclass(frozen=True)
class Session:
    name: str
    timezone: str
    start: int
    cutoff: int
    retest: bool


FX_SESSIONS = (Session("London 07:00 UTC", "UTC", 420, 1260, True),
               Session("London 08:00 UTC", "UTC", 480, 1260, True),
               Session("New York 13:30 UTC", "UTC", 810, 1260, True))
US_SESSION = Session("US cash 09:30 ET", "America/New_York", 570, 960, False)
GERMAN_SESSION = Session("Xetra 09:00 CET/CEST", "Europe/Berlin", 540, 1050, False)
TARGETS = (("1.5R", 1.5), ("2R", 2.), ("session_close", None))


def inventory():
    """Exact allow-list so vendor metadata can never admit a derived symbol."""
    for directory in ("history_2020_2025", "historical_12m"):
        for path in sorted((DATA / directory).glob("*_M15*.csv")):
            symbol = path.name.split("_M15", 1)[0].replace("_", " ")
            if symbol in FX | METALS | EQUITIES:
                yield symbol, path


def session_days(data, session):
    local = data.time.dt.tz_convert(session.timezone)
    minutes = (local.dt.hour * 60 + local.dt.minute).to_numpy()
    dates = local.dt.date.to_numpy()
    for start in np.flatnonzero((minutes == session.start) & (local.dt.weekday.to_numpy() < 5)):
        # Include only candles ending at or before the session cutoff.
        stop = start + 1
        while stop < len(data) and dates[stop] == dates[start] and minutes[stop] + 15 <= session.cutoff:
            stop += 1
        if stop - start >= 4:
            yield start, stop


def session_setup(bounds, opening_bars, retest, indicators):
    start, stop = bounds
    end_range = start + opening_bars
    op, hi, lo, close, atr, ema, ema_slope, volume_ok, spread, eligible, seconds = indicators
    if end_range >= stop - 1 or not eligible[start:end_range].all():
        return None
    if seconds[end_range-1] - seconds[start] != 900 * (opening_bars-1):
        return None
    upper, lower = max(hi[start:end_range]), min(lo[start:end_range])
    bias = np.sign(close[end_range-1] - op[start])
    breakout = 0
    for i in range(end_range, stop-1):
        # Require consecutive candles through confirmation and the next entry.
        if not eligible[i] or not eligible[i+1] or seconds[i] - seconds[i-1] != 900 or seconds[i+1] - seconds[i] != 900:
            return None
        if not retest:
            side = int(close[i] > upper) - int(close[i] < lower)
            if not side or not volume_ok[i] or upper - lower < .5 * atr[i]:
                continue
            level = lower - spread[i] if side > 0 else upper + spread[i]
            return i, side, level
        if not breakout:
            direction = int(close[i] > upper) - int(close[i] < lower)
            if direction and direction == bias:
                breakout = direction
            continue
        if (breakout > 0 and close[i] < lower) or (breakout < 0 and close[i] > upper):
            return None
        if breakout > 0 and lo[i] <= upper < close[i] and close[i] > op[i] and close[i] > ema[i] and ema_slope[i] > 0:
            return i, 1, min(lo[i], upper) - spread[i]
        if breakout < 0 and hi[i] >= lower > close[i] and close[i] < op[i] and close[i] < ema[i] and ema_slope[i] < 0:
            return i, -1, max(hi[i], lower) + spread[i]
    return None


def stats_pair(scenarios, groups):
    return tuple(standard_trade_stats([trade for group in groups for trade in scenario[group]]) for scenario in scenarios)


def metrics(row, baseline, stress):
    return {**row, "oos_trades": baseline["trades"], "stress_trades": stress["trades"],
            "win_rate_pct": baseline["win_rate_pct"], "stress_win_rate_pct": stress["win_rate_pct"],
            "baseline_pf": baseline["profit_factor"], "stressed_pf": stress["profit_factor"],
            "baseline_drawdown_pct": baseline["max_drawdown_pct"], "stress_drawdown_pct": stress["max_drawdown_pct"],
            "max_drawdown_pct": max(baseline["max_drawdown_pct"], stress["max_drawdown_pct"]),
            "average_net_r": baseline["average_r"], "stressed_average_net_r": stress["average_r"],
            "live_approval": False}


def evaluate_session(data, session, indicators, executions, common):
    days = list(session_days(data, session))
    n = len(days)
    split = (0, 2*n//5, 3*n//5, 4*n//5, n)
    if min(np.diff(split)) < 1:
        raise ValueError(f"Too few anchored sessions in {common['source']} for {session.name}")
    common = {**common, "session": session.name, "timezone": session.timezone,
              "session_count": n, "oos_from_utc": data.time.iloc[days[split[2]][0]].isoformat(),
              "oos_last_session_utc": data.time.iloc[days[-1][0]].isoformat()}
    trials = {}
    for opening_bars in (1, 2):
        setups = [session_setup(day, opening_bars, session.retest, indicators) for day in days]
        for label, rr in TARGETS:
            scenarios = []
            for columns, slippage, commission in executions:
                trades = [[] for _ in range(4)]
                for block, (lo, hi) in enumerate(zip(split, split[1:])):
                    for idx in range(lo, hi):
                        if setups[idx] is None:
                            continue
                        i, side, stop = setups[idx]
                        trade = _trade_one(columns, i, side, stop, np.nan, breakeven=False,
                                           commission_per_lot=7*commission, slippage_points=slippage,
                                           commission_points=commission, end_idx=days[idx][1],
                                           absolute_stop=True, reward_risk=rr)
                        if trade:
                            trades[block].append(trade)
                scenarios.append(trades)
            trials[(opening_bars*15, label)] = scenarios
            yield metrics({**common, "evaluation": "fixed_variant", "opening_minutes": opening_bars*15,
                           "exit": label, "fold1_selection": "", "fold2_selection": "",
                           "development_pass": ""}, *stats_pair(scenarios, [2, 3]))

    selections, gates = [], []
    oos = [[], []]
    for test in (2, 3):
        def score(key):
            train, stress_train = stats_pair(trials[key], list(range(test-1)))
            valid, stress_valid = stats_pair(trials[key], [test-1])
            eligible = (min(train["trades"], stress_train["trades"]) >= 20
                        and min(valid["trades"], stress_valid["trades"]) >= 10
                        and min(train["average_r"], stress_train["average_r"], valid["average_r"], stress_valid["average_r"]) > 0)
            return eligible, min(stress_train["average_r"], stress_valid["average_r"]), stress_valid["average_r"]

        chosen = max(trials, key=score)
        selections.append(f"{chosen[0]}m / {chosen[1]}")
        gates.append(score(chosen)[0])
        for scenario, buckets in enumerate(trials[chosen]):
            oos[scenario].extend(buckets[test])
    yield metrics({**common, "evaluation": "walk_forward_selected", "opening_minutes": "", "exit": "",
                   "fold1_selection": selections[0], "fold2_selection": selections[1],
                   "development_pass": all(gates)}, *map(standard_trade_stats, oos))


def evaluate_file(symbol, path):
    data = load_history(path)
    # The recent exports use point/volume. Preserve the broker point rather
    # than the generic defaults applied by the shared legacy loader.
    if "point" in data:
        data["point_size"] = pd.to_numeric(data["point"], errors="coerce")
    source_volume = pd.to_numeric(data.get("volume", data.tick_volume), errors="coerce")
    spread = _causal_spreads(data)
    atr = _atr(data, 14)
    ema = data.close.ewm(span=200, adjust=False, min_periods=200).mean()
    volume = source_volume > 1.2 * source_volume.shift(1).rolling(20).median()
    prices = data[["open", "high", "low", "close"]]
    eligible = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1) & (data.point_size > 0) & np.isfinite(data.point_size) & (atr > 0)
    eligible &= (data.high >= prices[["open", "close"]].max(axis=1)) & (data.low <= prices[["open", "close"]].min(axis=1))
    indicators = tuple(series.to_numpy() for series in (data.open, data.high, data.low, data.close, atr,
                       ema, ema.diff(5), volume, spread*data.point_size, eligible,
                       (data.time - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()))
    executions = [(execution_columns(data, spread*multiplier), slippage, commission)
                  for multiplier, slippage, commission in ((1., 2., 1.), (2., 5., 1.5))]
    sessions = FX_SESSIONS if symbol in FX | METALS else (GERMAN_SESSION if symbol == "Germany 40" else US_SESSION,)
    category = "forex" if symbol in FX else "metal" if symbol in METALS else "equity_index"
    for session in sessions:
        yield from evaluate_session(data, session, indicators, executions,
                                    {"symbol": symbol, "asset_class": category, "dataset": path.parent.name,
                                     "source": path.name, "risk_per_trade_pct": 1.0})


def main():
    rows = [row for symbol, path in inventory() for row in evaluate_file(symbol, path)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} evaluations from {len(list(inventory()))} source CSVs")


if __name__ == "__main__":
    main()
