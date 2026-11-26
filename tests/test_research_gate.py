"""Shared sizing/gate engine: OOS stats and Monte Carlo must evaluate the same trade population."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research_lab as rl  # noqa: E402


def _trades(values, *, per_day=1, mae=None):
    base = pd.Timestamp("2025-01-01T00:00:00Z")
    out = []
    for i, r in enumerate(values):
        t = base + pd.Timedelta(days=i // per_day, minutes=15 * (i % per_day))
        out.append({"time": t.isoformat(), "exit_time": (t + pd.Timedelta(minutes=10)).isoformat(), "r": float(r), "mae_r": float(mae[i]) if mae else max(-r, 0.0), "mfe_r": max(r, 0.0)})
    return out


def _edge(n=600, seed=1):
    rng = np.random.default_rng(seed)
    return _trades(np.where(rng.random(n) < .4, 2.4, -1.02), per_day=2)


def test_stats_and_monte_carlo_share_one_population():
    t = _edge()
    s = rl._stats(t); mc = rl._monte_carlo(t, iterations=5, seed=3)
    assert mc["population_trades"] == s["signal_trades"] == len(t)
    assert mc["chrono_total_return_pct"] == pytest.approx(s["total_return_pct"])
    assert mc["chrono_max_drawdown_uncensored_pct"] == pytest.approx(s["ungated_max_drawdown_pct"])
    assert mc["chrono_either_limit_breached"] == (s["daily_limit_breached"] or s["total_limit_breached"])


def test_gate_truncation_is_reported_not_hidden():
    t = _trades([-1.0] * 40 + [2.0] * 40)          # breaches 10% trailing DD on trade 14, then recovers
    s = rl._stats(t)
    assert s["gate_truncated"] and s["trades"] == 14 and s["signal_trades"] == 80
    assert s["first_total_breach_trade"] == 13
    assert s["ungated_total_return_pct"] > s["total_return_pct"]
    assert rl._stats(t, enforce=False)["trades"] == 80


def test_permutation_has_zero_terminal_variance_bootstrap_does_not():
    t = _trades(np.r_[np.full(30, 1.0), np.full(20, -.5)])
    assert rl._monte_carlo(t, iterations=50, seed=1, method="permute")["terminal_return_std_pct"] == pytest.approx(0, abs=1e-9)
    assert rl._monte_carlo(t, iterations=50, seed=1, method="bootstrap")["terminal_return_std_pct"] > 1.0
    assert rl._monte_carlo(t, iterations=50, seed=1, method="block", block_len=5)["terminal_return_std_pct"] > 0.0


def test_sizing_modes():
    t = _trades([1.0] * 200, per_day=1)
    frac = rl._stats(t, mode="fractional")["total_return_pct"]
    fixed = rl._stats(t, mode="fixed")["total_return_pct"]
    cap = rl._stats(t, mode="capped", cap_mult=2.0)["total_return_pct"]
    assert frac == pytest.approx((1.0075 ** 200 - 1) * 100)
    assert fixed == pytest.approx(200 * .75)
    assert fixed < cap < frac
    assert rl._stats(_trades([-1.0] * 12), mode="dd_throttle")["max_drawdown_pct"] < rl._stats(_trades([-1.0] * 12))["max_drawdown_pct"]
    assert rl._stats(_edge(), mode="vol_target")["signal_trades"] == 600


def test_static_vs_trailing_drawdown_basis():
    t = _trades([2.0] * 40 + [-1.0] * 16)          # +81% then an ~11% giveback from peak, never below initial
    assert rl._stats(t, dd_basis="trailing")["total_limit_breached"]
    assert not rl._stats(t, dd_basis="static")["total_limit_breached"]


def test_p95_drawdown_is_censored_at_the_limit_uncensored_is_not():
    t = _trades(np.where(np.random.default_rng(5).random(400) < .3, 2.0, -1.0))
    mc = rl._monte_carlo(t, iterations=200, seed=7)
    assert mc["probability_total_limit_breach_pct"] > 50
    assert 10.0 <= mc["p95_max_drawdown_pct"] < 10.0 + 1.0
    assert mc["p95_max_drawdown_uncensored_pct"] > mc["p95_max_drawdown_pct"] + 2.0


def test_intratrade_mae_triggers_daily_limit_that_closed_pnl_misses():
    assert not rl._stats(_trades([.5], mae=[0.0]), risk_pct=5)["daily_limit_breached"]
    assert rl._stats(_trades([.5], mae=[1.1]), risk_pct=5)["daily_limit_breached"]   # floating -5.5% before closing +2.5%


def test_target_before_breach_probability():
    t = _edge()
    mc = rl._monte_carlo(t, iterations=100, seed=2, dd_basis="static", target_pct=10.0)
    assert 0 <= mc["probability_target_before_breach_pct"] <= 100 and mc["probability_target_before_breach_pct"] > 50


def test_trade_one_records_excursions_and_exit_time():
    d = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC"), "open": [100.] * 4, "high": [100.5, 100.5, 101.5, 100.5],
                      "low": [99.5, 99.5, 99.5, 99.5], "close": [100.] * 4, "point_size": [1.] * 4, "spread_points": [0.] * 4, "tick_size": [0.] * 4, "tick_value": [0.] * 4})
    tr = rl._trade_one(d, 0, 1, 2., np.nan, breakeven=False, commission_per_lot=0., slippage_points=0., commission_points=0., end_idx=4, horizon=3)
    assert tr["mae_r"] == pytest.approx(.25) and tr["mfe_r"] == pytest.approx(.75) and tr["exit_time"].startswith("2025-01-01T00:45")
