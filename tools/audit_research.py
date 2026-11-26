"""Reproduce legacy candidate Monte Carlo and summarize historical-data quality."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "history_2020_2025"
LEGACY_COMMIT = "df59bb5"
TARGETS = (
    ("Range Break 200 Index", "A"),
    ("Range Break 200 Index", "B"),
    ("Volatility 250 (1s) Index", "A"),
)


def _legacy_module() -> dict[str, Any]:
    source = subprocess.check_output(
        ["git", "show", f"{LEGACY_COMMIT}:research_lab.py"], cwd=ROOT, text=True
    )
    namespace: dict[str, Any] = {"__name__": "legacy_research_audit", "__file__": str(ROOT / "research_lab.py")}
    exec(compile(source, f"{LEGACY_COMMIT}:research_lab.py", "exec"), namespace)
    return namespace


def _legacy_event_breakdown(trades: list[dict[str, Any]], *, iterations: int, risk_pct: float, seed: int) -> dict[str, float]:
    """Instrument the committed path logic without changing its event semantics."""
    from collections import Counter

    ordered = sorted(trades, key=lambda t: pd.Timestamp(t["time"]))
    returns = np.asarray([float(t["r"]) for t in ordered], dtype=float)
    dates = [pd.Timestamp(t["time"]) for t in ordered]
    day_counts = list(Counter(d.date() for d in dates).values())
    rng = np.random.default_rng(seed)
    daily_hits = total_hits = both_hits = zero_hits = 0
    for _ in range(iterations):
        shuffled = rng.permutation(returns)
        equity = peak = 1.0
        cursor = 0
        daily_hit = total_hit = equity_zero = False
        for day_count in day_counts:
            day_start = equity
            day_breached = False
            for outcome in shuffled[cursor:cursor + day_count]:
                cursor += 1
                if day_breached or total_hit or equity_zero:
                    continue
                equity *= max(0.0, 1.0 + outcome * risk_pct / 100.0)
                equity_zero = equity <= 0.0
                peak = max(peak, equity)
                drawdown = (peak - equity) / peak * 100.0 if peak else 100.0
                if (day_start - equity) / max(day_start, 1e-12) * 100.0 >= 5.0:
                    day_breached = daily_hit = True
                if drawdown >= 10.0:
                    total_hit = True
        daily_hits += int(daily_hit)
        total_hits += int(total_hit)
        both_hits += int(daily_hit and total_hit)
        zero_hits += int(equity_zero)
    either_hits = daily_hits + total_hits - both_hits
    return {
        "daily_only_pct": (daily_hits - both_hits) / iterations * 100,
        "total_only_pct": (total_hits - both_hits) / iterations * 100,
        "both_limits_pct": both_hits / iterations * 100,
        "either_limit_pct": either_hits / iterations * 100,
        "equity_zero_pct": zero_hits / iterations * 100,
    }


def reproduce_legacy_candidates() -> list[dict[str, Any]]:
    old = _legacy_module()
    files = sorted(DATA.glob("*_M15_2020_2025.csv"))
    snapshots = json.loads((ROOT / "approved_strategies.json").read_text(encoding="utf-8"))
    stored = {(r["symbol"], r["strategy_id"]): r for r in snapshots.get("monte_carlo_qualified", [])}
    output = []
    for symbol, strategy_id in TARGETS:
        found = None
        for file_ix, path in enumerate(files, 1):
            peek = pd.read_csv(path, nrows=1)
            if "symbol" in peek and str(peek.symbol.iloc[0]) == symbol:
                found = (file_ix, path)
                break
        if found is None:
            output.append({"symbol": symbol, "strategy_id": strategy_id, "status": "historical CSV missing"})
            continue
        file_ix, path = found
        d = old["add_indicators"](old["load_history"](path))
        sides, stops, targets, _ = old["build_signals"](d, symbol, strategy_id)
        trades = old["simulate"](
            d, (sides, stops, targets, np.full(len(d), "", dtype=object)),
            start="2024-01-01", end="2026-01-01", commission_per_lot=7.0,
            slippage_points=2.0, commission_points=1.0,
            breakeven=(strategy_id == "A"),
        )
        seed = 42 + file_ix * 100 + ord(strategy_id)
        mc = old["_monte_carlo"](trades, iterations=1000, risk_pct=.75, seed=seed)
        event_breakdown = _legacy_event_breakdown(trades, iterations=1000, risk_pct=.75, seed=seed)
        rvals = np.asarray([float(t["r"]) for t in trades])
        theoretical_zero_count = int(np.sum(1 + rvals * .0075 <= 0))
        snapshot = stored.get((symbol, strategy_id), {})
        output.append({
            "symbol": symbol, "strategy_id": strategy_id, "status": "reproduced",
            "trade_count": len(trades), "seed": seed,
            "reported_legacy_ruin_pct": snapshot.get("monte_carlo", {}).get("probability_of_ruin_pct"),
            "reproduced_legacy_ruin_pct": mc["probability_of_ruin_pct"],
            "legacy_p95_max_drawdown_pct": mc["p95_max_drawdown_pct"],
            "legacy_event_breakdown": event_breakdown,
            "any_single_trade_nonpositive_equity_possible": bool(theoretical_zero_count),
            "legacy_definition": "a simulated path hit daily >=5% OR peak-to-trough total >=10%; not account equity zero",
        })
    return output


def data_quality() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(DATA.glob("*_M15_2020_2025.csv")):
        d = pd.read_csv(path, usecols=lambda c: c in {"symbol", "time", "spread_points", "point_size", "tick_size", "tick_value"})
        times = pd.to_datetime(d.time, utc=True, errors="coerce")
        diffs = times.sort_values().diff().dt.total_seconds() / 60
        symbol = str(d.symbol.iloc[0]) if "symbol" in d and len(d) else path.stem
        spreads = pd.to_numeric(d.get("spread_points", pd.Series(0, index=d.index)), errors="coerce")
        rows.append({
            "symbol": symbol, "bars": len(d),
            "first_bar_utc": str(times.min()), "last_bar_utc": str(times.max()),
            "duplicate_timestamps": int(times.duplicated().sum()),
            "zero_or_missing_spread_pct": float((~np.isfinite(spreads) | (spreads <= 0)).mean() * 100),
            "median_positive_spread_points": float(spreads[spreads > 0].median()) if (spreads > 0).any() else None,
            "nonpositive_point_size_rows": int((pd.to_numeric(d.get("point_size", pd.Series(np.nan, index=d.index)), errors="coerce") <= 0).sum()),
            "nonpositive_tick_size_rows": int((pd.to_numeric(d.get("tick_size", pd.Series(np.nan, index=d.index)), errors="coerce") <= 0).sum()),
            "nonpositive_tick_value_rows": int((pd.to_numeric(d.get("tick_value", pd.Series(np.nan, index=d.index)), errors="coerce") <= 0).sum()),
            "gaps_over_15m": int((diffs > 15.01).sum()),
        })
    return rows


def main() -> int:
    output = {"legacy_commit": LEGACY_COMMIT, "legacy_candidates": reproduce_legacy_candidates(), "data_quality": data_quality(),
              "limitations": ["Historical spreads are bar-reported and zero/missing 18.9% of bars in the harvested set; no historical commissions were captured.",
                              "Point, tick size, and tick value in exported rows are current terminal snapshots repeated over historical bars, not date-specific contract specifications.",
                              "Exact-lot PnL cannot be reconstructed from R outcomes without historical contract and account-currency conversion data."]}
    dest = ROOT / "data" / "research_audit_2026_11"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "independent_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"legacy_candidates": output["legacy_candidates"], "symbols_audited": len(output["data_quality"]), "audit_file": str(dest / "independent_audit.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
