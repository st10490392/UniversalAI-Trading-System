"""Walk-forward session research on FX, spot metals and equity-index CFDs."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from backtest import _resample_h1, _smc_side_at
from brain import market_bias_series, quant_metrics_series
from fx_stocks_session_orb import EQUITIES, FX_SESSIONS, GERMAN_SESSION, US_SESSION, inventory, metrics, session_setup
from research_lab import (FEATURES, ROOT, _causal_spreads, _trade_one, add_indicators, build_signals,
                          load_history, standard_qualification, standard_trade_stats)
from walkforward_standard import Variant, execution_columns, signal_components, stop_levels

OUT = ROOT / "data/research_2020_2025/fx_stocks_all_strategies"
EXITS = {"1.5R": 1.5, "2R": 2., "3R": 3., "session_close": None}
RULES = {
    "trend_momentum": "EMA12/26 crossover, RSI >=55 long or <=45 short; H1 EMA20/50 trend and ATR14/50 >1; 1.5 ATR stop",
    "judas_sweep_mss": "Complete 00:00-06:00 UTC Asia range; sweep and reclaim with prior 3-bar structure break; 1.5 ATR stop",
    "silver_bullet_fvg": "Three-bar imbalance forms and retests within 10:00-11:00 New York, intersect active session; EMA20/50 bias; 1.5 ATR stop",
    "legacy_sweep_fvg": "Prior 20-bar liquidity sweep, three-bar FVG, confirmed H1 EMA/fractal bias and ATR/efficiency/z-score gates; wick stop capped at 3 ATR",
    "vwap_mean_reversion": "Prior BB/Keltner squeeze, |z| >2, ER20 <0.25, ADX14 <=20, ATR ratio <=1.2; session VWAP or generic exit; 1.5 ATR stop",
    "volatility_breakout_xgb": "20-bar range break, ATR14/50 >1.1, H1 EMA20/50 alignment; train-only XGBoost net-win probability >=0.55; 1.25 ATR stop",
    "ema_trend_pullback": "M15 EMA20/50 pullback and resumption; H1 EMA20/50 alignment and ATR14/50 >1; spread-buffered wick stop",
    "session_orb_15m": "Exact opening anchor; 15m opening range; FX/metals breakout, retest and EMA200/session bias; equity breakout, 1.2x volume and 0.5 ATR range",
    "session_orb_30m": "Exact opening anchor; 30m opening range; FX/metals breakout, retest and EMA200/session bias; equity breakout, 1.2x volume and 0.5 ATR range",
}
SWEEPS = {"judas_sweep_mss", "silver_bullet_fvg", "legacy_sweep_fvg"}
ORB = {"session_orb_15m", "session_orb_30m"}


def signed(up, down):
    return np.select([up, down], [1, -1], default=0)


def adx14(data):
    up, down = data.high.diff(), -data.low.diff()
    plus = up.where((up > down) & (up > 0), 0.)
    minus = down.where((down > up) & (down > 0), 0.)
    a, b = (v.ewm(alpha=1/14, adjust=False, min_periods=14).mean() for v in (plus, minus))
    return (100*(a-b).abs()/(a+b).replace(0, np.nan)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()


def prepare(path):
    data = load_history(path)
    if "point" in data:
        data["point_size"] = pd.to_numeric(data["point"], errors="coerce")
    if "volume" in data:
        data["tick_volume"] = pd.to_numeric(data["volume"], errors="coerce")
    data = add_indicators(data)
    prices = data[["open", "high", "low", "close"]]
    valid = (np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
             & (data.high >= prices[["open", "close"]].max(axis=1))
             & (data.low <= prices[["open", "close"]].min(axis=1))
             & (data.point_size > 0) & np.isfinite(data.point_size) & (data.atr14 > 0))
    return data, _causal_spreads(data), valid


def regular_signals(data, spreads, valid):
    h, l, c = data.high, data.low, data.close
    spread = spreads * data.point_size
    h1 = _resample_h1(data)
    # Only complete H1 candles, indexed by timestamp when their close is known.
    counts = data.set_index("time").close.resample("1h", label="right", closed="left").count()
    h1 = h1.loc[h1.time.map(counts).eq(4)].reset_index(drop=True)
    clock = data.time + pd.Timedelta(minutes=15)
    trend = np.sign(h1.close.ewm(span=20, adjust=False, min_periods=20).mean()
                    - h1.close.ewm(span=50, adjust=False, min_periods=50).mean()).fillna(0)
    known = pd.Series(trend.to_numpy(), index=h1.time).reindex(clock, method="ffill").fillna(0).to_numpy()
    structure = pd.Series(market_bias_series(h1).to_numpy(), index=h1.time).reindex(clock, method="ffill").fillna("NEUTRAL").to_numpy()

    def structural(sides, distance):
        return sides, c.to_numpy() - sides*distance

    result = {}
    for name, family in (("trend_momentum", "A"), ("vwap_mean_reversion", "C"), ("volatility_breakout_xgb", "D")):
        side, distance, _, _ = build_signals(data, data.attrs["symbol"], family)
        gate = (adx14(data) <= 20) & (data.atr_ratio <= 1.2) if family == "C" else (side == known) & (data.atr_ratio > 1)
        result[name] = structural(np.where(gate, side, 0), distance)
    side, level = stop_levels(data, signal_components(data, "ema_trend_pullback", spreads), Variant(1.5, False, False))
    result["ema_trend_pullback"] = np.where((side == known) & (data.atr_ratio > 1), side, 0), level

    utc_minute = data.time.dt.hour*60 + data.time.dt.minute
    day = data.time.dt.normalize()
    asia = (utc_minute < 360) & valid
    asia_hi = h.where(asia).groupby(day).transform("max")
    asia_lo = l.where(asia).groupby(day).transform("min")
    asia_ready = asia.groupby(day).transform("sum").eq(24) & (utc_minute >= 360)
    side = signed(asia_ready & (l < asia_lo) & (c > asia_lo) & (c > h.shift(1).rolling(3).max()),
                  asia_ready & (h > asia_hi) & (c < asia_hi) & (c < l.shift(1).rolling(3).min()))
    result["judas_sweep_mss"] = structural(side, data.atr14*1.5)
    ny = data.time.dt.tz_convert("America/New_York")
    silver = (ny.dt.hour == 10)
    # Setup and retest both belong to the permitted window. Four bars must be contiguous.
    consecutive = (data.time-data.time.shift(3)).eq(pd.Timedelta(minutes=45))
    side = signed(silver & silver.shift(1, fill_value=False) & consecutive & (l.shift(1) > h.shift(3))
                  & (l <= l.shift(1)) & (h >= h.shift(3)) & (data.ema20 > data.ema50),
                  silver & silver.shift(1, fill_value=False) & consecutive & (h.shift(1) < l.shift(3))
                  & (h >= h.shift(1)) & (l <= l.shift(3)) & (data.ema20 < data.ema50))
    result["silver_bullet_fvg"] = structural(side, data.atr14*1.5)

    highs, lows, closes = (series.to_numpy(float) for series in (h, l, c))
    quant = quant_metrics_series(data)
    possible = valid & quant.gate & (((structure == "BULLISH") & (quant.zscore <= 2) & (l > h.shift(2)))
                                     | ((structure == "BEARISH") & (quant.zscore >= -2) & (h < l.shift(2))))
    sides = np.zeros(len(data), dtype=np.int8)
    for i in np.flatnonzero(possible):
        side = _smc_side_at(highs, lows, closes, i, structure[i])
        sides[i] = 1 if side == "BUY" else -1 if side == "SELL" else 0
    stop = np.where(sides > 0, lows, highs+spread)
    sides = np.where(abs(closes-stop) <= data.atr14.to_numpy()*3, sides, 0)
    result["legacy_sweep_fvg"] = sides, stop
    return result


def trading_days(data, timezone, start, entry_end, cutoff, *, exact_open=False, fixed_us_orb=False):
    local = data.time.dt.tz_convert(timezone)
    minutes = (local.dt.hour*60 + local.dt.minute).to_numpy()
    result = []
    for _, offsets in data.groupby(local.dt.date, sort=False).indices.items():
        day_minutes = minutes[offsets]
        if local.iloc[offsets[0]].weekday() >= 5:
            continue
        active = offsets[(day_minutes >= start) & (day_minutes < entry_end)]
        closing = offsets[(day_minutes >= start) & (day_minutes + 15 <= cutoff)]
        if len(active) and len(closing):
            lo, hi, end = int(active[0]), int(active[-1])+1, int(closing[-1])+1
            if exact_open and minutes[lo] != start:
                continue
            # Both the explicit 14:30 UTC anchor and the true 09:30 ET
            # cash open must hold. Skip daylight-time dates before splitting.
            if fixed_us_orb and (data.time.iloc[lo].hour, data.time.iloc[lo].minute) != (14, 30):
                continue
            result.append((lo, hi, end))
    return result


def order_candidates(data, days, name, signals, indicators, valid, *, retest):
    candidates = []
    seconds = indicators[-1]
    for lo, hi, end in days:
        if name in ORB:
            setup = session_setup((lo, hi), int(name[-3:-1])//15, retest, indicators)
            candidates.append([] if setup is None else [setup])
            continue
        sides, stops = signals[name]
        raw = np.flatnonzero(sides[lo:hi-1]) + lo
        candidates.append([(int(i), int(sides[i]), float(stops[i])) for i in raw
                           if valid[i] and valid[i+1] and seconds[i+1]-seconds[i] == 900
                           and (name != "silver_bullet_fvg" or
                                data.time.iloc[i+1].tz_convert("America/New_York").hour == 10)])
    return candidates


def play(columns, choices, days, lo, hi, rr, slip, fee, vwap, allowed=None, *, labels=False):
    trades = []
    for day in range(lo, hi):
        _, _, end = days[day]
        for i, side, initial_stop in choices[day]:
            if allowed is not None and i not in allowed:
                continue
            target = vwap[i] if vwap is not None else np.nan
            if vwap is not None and not np.isfinite(target):
                continue
            trade = _trade_one(columns, i, side, initial_stop, target, breakeven=False,
                               commission_per_lot=7*fee, slippage_points=slip, commission_points=fee,
                               end_idx=end, absolute_stop=True, reward_risk=rr)
            if trade is not None:
                trades.append((i, trade) if labels else trade)
                if not labels:
                    break  # one completed-bar opportunity per session-day
    return trades


def train_model(data, columns, choices, days, train_end):
    samples = play(columns, choices, days, 0, train_end, 2., 2., 1., None, labels=True)
    labels = [int(trade["r"] > 0) for _, trade in samples]
    if len(labels) < 40 or len(set(labels)) < 2:
        return None, len(labels)
    features = data.loc[:, list(FEATURES)].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    train = xgb.DMatrix(features[[i for i, _ in samples]], label=labels)
    model = xgb.train({"max_depth": 3, "eta": .04, "subsample": .8, "colsample_bytree": .85,
                       "min_child_weight": 5, "lambda": 2, "objective": "binary:logistic",
                       "eval_metric": "logloss", "nthread": 1, "seed": 42}, train, num_boost_round=160)
    indices = [i for choices_on_day in choices[train_end:] for i, _, _ in choices_on_day]
    prediction = model.predict(xgb.DMatrix(features[indices])) if indices else []
    return {i for i, probability in zip(indices, prediction) if probability >= .55}, len(labels)


def evaluate_strategy(data, days, choices, executions, name, row):
    n = len(days)
    split = (0, 2*n//5, 3*n//5, 4*n//5, n)
    output = {profile: [[], []] for profile in EXITS}
    exits = dict(EXITS)
    if name == "vwap_mean_reversion":
        exits["session_vwap"] = None
        output["session_vwap"] = [[], []]
    selected = [[], []]
    picks, gates, model_labels = [], [], []
    model_available = True
    for test in (2, 3):
        train_end, valid_end, test_end = split[test-1:test+2]
        allowed = None
        if name == "volatility_breakout_xgb":
            allowed, count = train_model(data, executions[0][0], choices, days, train_end)
            model_labels.append(count)
            if allowed is None:
                model_available = False
                picks.append("model_unavailable")
                gates.append(False)
                continue
        trials = {}
        for profile, rr in exits.items():
            vwap = row["_session_vwap"] if profile == "session_vwap" else None
            scenes = []
            for scenario, (columns, slip, fee) in enumerate(executions):
                validation = play(columns, choices, days, train_end, valid_end, rr, slip, fee, vwap, allowed)
                test_trades = play(columns, choices, days, valid_end, test_end, rr, slip, fee, vwap, allowed)
                output[profile][scenario].extend(test_trades)
                scenes.append((standard_trade_stats(validation), test_trades))
            trials[profile] = scenes
        def score(profile):
            base, stress = (scene[0] for scene in trials[profile])
            eligible = min(base["trades"], stress["trades"]) >= 10 and min(base["average_r"], stress["average_r"]) > 0
            return eligible, stress["average_r"], -(stress["max_drawdown_pct"])
        profile = max(trials, key=score)
        picks.append(profile)
        gates.append(score(profile)[0])
        for scenario, (_, trades) in enumerate(trials[profile]):
            selected[scenario].extend(trades)

    public = {key: value for key, value in row.items() if key != "_session_vwap"}
    common = {**public, "strategy": name, "rules": RULES[name], "candidate_sessions": n,
              "oos_start_utc": data.time.iloc[days[split[2]][0]].isoformat(),
              "oos_end_utc": data.time.iloc[days[-1][2]-1].isoformat(),
              "model_labels": ";".join(map(str, model_labels)), "model_available_both_folds": model_available}
    for label, trades in [*output.items(), ("walk_forward_selected", selected)]:
        baseline, stress = map(standard_trade_stats, trades)
        decision = standard_qualification(baseline, stress)
        yield metrics({**common, "exit_profile": label, "fold1_exit": picks[0], "fold2_exit": picks[1],
                       "development_pass": all(gates), "oos_pass": decision["standard_pass"],
                       "research_pass": label == "walk_forward_selected" and all(gates) and decision["standard_pass"],
                       "oos_failures": "; ".join(decision["failures"])}, baseline, stress)


def evaluate_source(symbol, path):
    data, spreads, valid = prepare(path)
    signals = regular_signals(data, spreads, valid)
    ema = data.close.ewm(span=200, adjust=False, min_periods=200).mean()
    vol_ok = data.tick_volume > 1.2*data.tick_volume.shift(1).rolling(20).median()
    indicators = tuple(series.to_numpy() for series in (data.open, data.high, data.low, data.close, data.atr14,
                       ema, ema.diff(5), vol_ok, spreads*data.point_size, valid,
                       (data.time-pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()))
    executions = [(execution_columns(data, spreads*multiple), slip, fee)
                  for multiple, slip, fee in ((1., 2., 1.), (2., 5., 1.5))]
    if symbol in EQUITIES:
        germany = symbol == "Germany 40"
        active = [("Xetra cash morning", "Europe/Berlin", 540, 720, 1050)] if germany else [
            ("US cash morning", "America/New_York", 570, 720, 960)]
        orb = [(GERMAN_SESSION, 720)] if germany else [(US_SESSION, 720)]
    else:
        active = [("London killzone UTC", "UTC", 420, 600, 1260), ("New York killzone UTC", "UTC", 780, 960, 1260)]
        orb = [(s, 600 if s.start < 600 else 960) for s in FX_SESSIONS]
    for name in RULES:
        specs = [(s.name, s.timezone, s.start, entry_end, s.cutoff, s.retest) for s, entry_end in orb] if name in ORB else [
            (label, tz, start, end, end if name in SWEEPS else cutoff, False) for label, tz, start, end, cutoff in active]
        for label, tz, start, entry_end, cutoff, retest in specs:
            fixed_us_orb = name in ORB and symbol in {"Wall Street 30", "US Tech 100"}
            days = trading_days(data, tz, start, entry_end, cutoff, exact_open=name in ORB, fixed_us_orb=fixed_us_orb)
            if len(days) < 5:
                raise ValueError(f"Too few sessions: {path.name}, {label}")
            choices = order_candidates(data, days, name, signals, indicators, valid.to_numpy(), retest=retest)
            # Reset VWAP at the active session's first bar. It is frozen at
            # signal time, then checked against the actual next-open fill.
            vwap = np.full(len(data), np.nan)
            if name == "vwap_mean_reversion":
                typical = (data.high+data.low+data.close).to_numpy()/3
                volume = data.tick_volume.to_numpy()
                for lo, hi, _ in days:
                    vwap[lo:hi] = np.cumsum(typical[lo:hi]*volume[lo:hi])/np.cumsum(volume[lo:hi])
            yield from evaluate_strategy(data, days, choices, executions, name, {
                "symbol": symbol, "source": path.name, "dataset": path.parent.name, "session": label,
                "session_timezone": tz, "required_utc_open": "14:30" if fixed_us_orb else "",
                "entry_start_min": start, "entry_end_min_exclusive": entry_end,
                "flat_by_min": cutoff, "risk_per_trade_pct": 1.0, "_session_vwap": vwap,
            })


def main():
    sources = list(inventory())
    rows = []
    for symbol, path in sources:
        rows.extend(evaluate_source(symbol, path))
        print("Evaluated", path.name, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/"walkforward_all_strategies.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pairs = defaultdict(list)
    for row in rows:
        if row["exit_profile"] == "walk_forward_selected":
            pairs[row["symbol"], row["strategy"], row["session"]].append(row)
    candidates = [{"symbol": key[0], "strategy": key[1], "session": key[2], "live_approval": False, "sources": observations}
                  for key, observations in pairs.items() if all(r["research_pass"] for r in observations)]
    (OUT/"approved_candidates.json").write_text(json.dumps({
        "research_only": True, "live_approval": False, "approved": [], "candidates": candidates,
        "input_csvs": [str(path.relative_to(ROOT)).replace("\\", "/") for _, path in sources],
        "strategies": RULES, "dataset_evaluations": len(rows),
    }, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{len(rows)} evaluations, {len(pairs)} symbol/strategy/session groups, {len(candidates)} research candidates")


if __name__ == "__main__":
    main()
