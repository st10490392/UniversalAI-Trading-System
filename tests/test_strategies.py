import numpy as np
import pandas as pd
import pytest

from strategies import PairPerformanceFilter, ema_trend_pullback, session_orb_retest
from research_lab import performance_filter_registry


def _bars(prices: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"open": prices, "high": prices + .05, "low": prices - .05, "close": prices})


def _orb_buy_frame() -> pd.DataFrame:
    frame = _bars(np.linspace(99.4, 99.9, 35))
    opening_range = pd.DataFrame({"open": [100.0] * 6, "high": [100.1] * 6, "low": [99.9] * 6, "close": [100.0] * 6})
    pattern = pd.DataFrame({
        "open": [100.05, 100.2, 100.1],
        "high": [100.45, 100.25, 100.45],
        "low": [100.0, 100.0, 100.08],
        "close": [100.4, 100.15, 100.4],
    })
    return pd.concat((frame, opening_range, pattern), ignore_index=True)


def test_orb_buy_signal_is_completed_bar_deterministic():
    frame = _orb_buy_frame()
    signal = session_orb_retest(frame, point_value=.01, spread_points=2)
    assert signal["command"] == "BUY"
    assert signal["setup"] == "session_orb_retest"
    assert signal["sl"] == pytest.approx(99.98)
    assert signal["tp"] > 100.4
    assert .75 <= signal["confidence"] <= 1.0
    assert signal == session_orb_retest(frame, point_value=.01, spread_points=2)


def test_strategies_fail_closed_for_invalid_or_insufficient_ohlc():
    signal = session_orb_retest(pd.DataFrame({"open": [1.0]}))
    assert signal["command"] == "WAIT"
    invalid = _bars(np.linspace(1, 2, 70))
    invalid.loc[3, "high"] = invalid.loc[3, "low"] - 1
    assert ema_trend_pullback(invalid)["command"] == "WAIT"
    assert ema_trend_pullback(_bars(np.linspace(1, 2, 70)), rr_ratio=0)["reason"] == "Invalid strategy parameters"


def test_ema_trend_pullback_returns_buy_after_completed_resumption():
    prices = np.linspace(100, 120, 80)
    frame = _bars(prices)
    # The preceding bar retests the fast EMA and the final completed bar resumes.
    frame.loc[len(frame) - 2, ["open", "high", "low", "close"]] = [119.0, 119.4, 116.0, 119.1]
    frame.loc[len(frame) - 1, ["open", "high", "low", "close"]] = [119.2, 120.8, 119.0, 120.5]
    signal = ema_trend_pullback(frame, min_separation_atr_mult=.01)
    assert signal["command"] == "BUY"
    assert signal["sl"] < 120.5 < signal["tp"]


def test_pair_performance_filter_is_fail_closed_and_does_not_write_files():
    filter_ = PairPerformanceFilter()
    missing = filter_.evaluate_metrics("EURUSD", "ema_trend_pullback", {"trades": 100})
    assert missing["approved"] is False
    assert "Missing or invalid metric: pf" in missing["failures"]
    blocked = filter_.evaluate_metrics("EURUSD", "ema_trend_pullback", {"trades": 79, "pf": 1.3, "max_dd_pct": 9, "cost_stressed_pf": 1.2})
    assert blocked["regime"] == "BLOCKED"
    accepted = filter_.evaluate_metrics("USDJPY", "session_orb_retest", {"trades": 80, "pf": 1.25, "max_dd_pct": 10, "cost_stressed_pf": 1.10})
    assert accepted["approved"] is True
    assert filter_.is_approved("USDJPY", "session_orb_retest") is True
    assert filter_.is_approved("MISSING", "session_orb_retest") is False
    assert filter_.get_registry_export()[-1]["symbol"] == "USDJPY"


def test_research_performance_registry_requires_adverse_cost_stress():
    results = [
        {"symbol": "EURUSD", "strategy": "Trend Momentum", "oos": {"trades": 100, "profit_factor": 1.4, "max_drawdown_pct": 8}, "stress": []},
        {"symbol": "USDJPY", "strategy": "Trend Momentum", "oos": {"trades": 100, "profit_factor": 1.4, "max_drawdown_pct": 8}, "stress": [{"scenario": "combined_adverse_costs", "profit_factor": 1.2}]},
    ]
    registry = performance_filter_registry(results)
    assert registry[0]["approved"] is False
    assert "Missing or invalid metric: cost_stressed_pf" in registry[0]["failures"]
    assert registry[1]["approved"] is True
