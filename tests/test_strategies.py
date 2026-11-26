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

def test_anchored_orb_uses_session_opening_bars_from_datetime_index():
    frame = _orb_buy_frame()
    frame.index = pd.date_range("2026-01-05", periods=len(frame), freq="15min", tz="UTC")
    anchor = frame.index[-9]
    # Put an intervening candle outside the ORB before the breakout triple.
    extra = frame.iloc[[-4]].copy()
    extra.loc[:, "high"] = 102.0
    extended = pd.concat([frame.iloc[:-3], extra, frame.iloc[-3:]], ignore_index=True)
    extended.index = pd.date_range(frame.index[0], periods=len(extended), freq="15min", tz="UTC")
    assert session_orb_retest(extended)["command"] == "WAIT"
    assert session_orb_retest(extended, session_start_utc=anchor)["command"] == "BUY"
    assert session_orb_retest(extended, session_start_utc=anchor + pd.Timedelta(minutes=1))["command"] == "WAIT"


@pytest.mark.parametrize("fault", ["reverse", "duplicate", "nat", "gap", "nonpositive"])
def test_bad_bars_fail_closed(fault):
    frame = _orb_buy_frame()
    frame["time"] = pd.date_range("2026-01-05", periods=len(frame), freq="15min", tz="UTC")
    if fault == "reverse":
        frame = frame.iloc[::-1]
    elif fault == "duplicate":
        frame.loc[10, "time"] = frame.loc[9, "time"]
    elif fault == "nat":
        frame.loc[10, "time"] = pd.NaT
    elif fault == "gap":
        frame.loc[20:, "time"] += pd.Timedelta(days=30)
    else:
        frame.loc[5, "low"] = -1.0
    signal = session_orb_retest(frame)
    assert signal["command"] == "WAIT"
    assert signal["sl"] == signal["tp"] == signal["confidence"] == 0.0


@pytest.mark.parametrize("spread_points", [0, 20, 50])
def test_orb_buy_and_sell_preserve_spread_adjusted_rr(spread_points):
    frame = _orb_buy_frame()
    spread = spread_points * .01
    buy = session_orb_retest(frame, point_value=.01, spread_points=spread_points)
    ask = 100.4 + spread
    assert (buy["tp"] - ask) / (ask - buy["sl"]) == pytest.approx(1.5)
    mirror = pd.DataFrame({"open": 200 - frame.open, "high": 200 - frame.low,
                           "low": 200 - frame.high, "close": 200 - frame.close})
    sell = session_orb_retest(mirror, point_value=.01, spread_points=spread_points)
    bid = 99.6
    assert (bid - (sell["tp"] - spread)) / (sell["sl"] - bid) == pytest.approx(1.5)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1, True, np.bool_(True), 1j])
def test_pair_filter_rejects_invalid_metrics_and_isolates_copies(bad):
    from strategies import validate_signal

    with pytest.raises(ValueError):
        PairPerformanceFilter(min_pf=bad)
    registry = PairPerformanceFilter()
    row = registry.evaluate_metrics("EURUSD", "test", {"trades": 100, "pf": 2, "max_dd_pct": bad, "cost_stressed_pf": 2})
    assert not row["approved"] and not row["live_approval"]
    row["failures"].clear()
    export = registry.get_registry_export()[0]
    assert export["failures"]
    export["approved"] = True
    assert not registry.is_approved("EURUSD", "test")
    assert validate_signal({"command": "BUY"})["command"] == "WAIT"


def test_default_research_output_does_not_touch_live_approval(tmp_path, monkeypatch):
    import json
    import sys
    import research_lab

    live = tmp_path / "approved_strategies.json"
    output = tmp_path / "reports"
    monkeypatch.setattr(research_lab, "APPROVED_PATH", live)
    monkeypatch.setattr(research_lab, "run", lambda args: ([], []))
    monkeypatch.setattr(sys, "argv", ["research_lab.py", "--output-dir", str(output)])
    assert research_lab.main() == 0
    assert not live.exists()
    assert json.loads((output / "approved_candidates.json").read_text())["approved"] == []
    sys.argv.extend(["--approved-path", str(live)])
    assert research_lab.main() == 0 and live.exists()


def test_standard_research_replays_trades_past_legacy_drawdown_stop():
    from research_lab import _stats, standard_trade_stats

    trades = [{"time": "2025-01-01", "r": -1.0}] * 12 + [{"time": "2025-01-02", "r": 1.5}] * 8
    assert _stats(trades)["trades"] < len(trades)
    standard = standard_trade_stats(trades)
    assert standard["trades"] == len(trades)
    assert standard["average_r"] == 0.0
    assert standard["max_drawdown_pct"] == pytest.approx((1 - .99 ** 12) * 100)


@pytest.mark.parametrize(("overrides", "passes"), [
    ({}, True),
    ({"trades": 49}, False),
    ({"profit_factor": 1.1499}, False),
    ({"average_r": 0.0}, False),
    ({"max_drawdown_pct": 20.0001}, False),
    ({"max_drawdown_pct": float("nan")}, False),
])
def test_standard_qualification_boundaries_ignore_hold_time(overrides, passes):
    from research_lab import standard_qualification

    stats = {"trades": 50, "average_r": .1, "profit_factor": 1.15,
             "max_drawdown_pct": 20.0, "holds_not_verified_gt_2": 50}
    result = standard_qualification(stats, {**stats, **overrides})
    assert result["standard_pass"] is passes
    assert result["live_approval"] is False
    assert bool(result["failures"]) is not passes
    assert not standard_qualification({**stats, "average_r": -.1}, stats)["standard_pass"]


@pytest.mark.parametrize(("override", "status", "failure"), [
    ({}, "pass", None),
    ({"max_floating_drawdown_pct": 1.001}, "fail", "floating drawdown"),
    ({"best_day_profit_share_pct": 15.001}, "fail", "best day profit share"),
    ({"minimum_hold_minutes": 2.0}, "fail", "minimum hold"),
    ({"minimum_hold_minutes": None}, "unverified", None),
    ({"max_floating_drawdown_pct": False}, "unverified", None),
])
def test_optional_prop_execution_profile(override, status, failure):
    from research_lab import evaluate_prop_execution_profile

    measured = {"max_floating_drawdown_pct": 1.0, "best_day_profit_share_pct": 15.0, "minimum_hold_minutes": 2.001}
    result = evaluate_prop_execution_profile(**{**measured, **override})
    assert result["status"] == status
    assert result["failures"] == ([failure] if failure else [])
    assert result["live_approval"] is False


def test_walkforward_risk_normalizes_actual_fill_across_price_scales():
    from walkforward_standard import execution_columns, walk_segment
    from research_lab import standard_trade_stats

    for scale in (.01, 1., 100.):
        data = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC"),
            "open": np.array([100, 101, 103, 104]) * scale,
            "high": np.array([101, 102, 105, 105]) * scale,
            "low": np.array([99, 100.5, 102, 103]) * scale,
            "close": np.array([100, 101, 104, 104]) * scale,
            "point_size": .01 * scale, "tick_size": .01 * scale, "tick_value": 0.,
        })
        columns = execution_columns(data, pd.Series([2.] * len(data)))
        stops = np.array([100. * scale, np.nan, np.nan, np.nan])
        trades = walk_segment(columns, np.array([1, 0, 0, 0]), stops,
                              lo=0, hi=3, rr=2., slippage=2., commission=1.)
        assert len(trades) == 1
        entry = 101.04
        assert trades[0]["r"] == pytest.approx(2 - .03 / (entry - 100))
        assert standard_trade_stats(trades)["total_return_pct"] == pytest.approx(trades[0]["r"])


def test_walkforward_selects_parameters_using_earlier_blocks_only(monkeypatch):
    import walkforward_standard as wf

    good, bad = wf.VARIANTS[:2]
    monkeypatch.setattr(wf, "VARIANTS", (good, bad))
    monkeypatch.setattr(wf, "signal_components", lambda *args: None)
    monkeypatch.setattr(wf, "stop_levels", lambda *args: (None, None))
    dates = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")})

    def block(_columns, _sides, _stops, *, lo, hi, rr, slippage, commission):
        r = (.3 if rr == good.rr else -.3) if lo < 2 else (-.3 if rr == good.rr else .3)
        return [{"time": dates.time.iloc[lo].isoformat(), "r": r}] * 60

    monkeypatch.setattr(wf, "walk_segment", block)
    _, stress, _, folds, _, _ = wf.evaluate_mode(dates, None, [(None, 2, 1), (None, 5, 1.5)], list(range(5)), "ema_trend_pullback")
    assert folds[0]["development_pass"]
    assert folds[0]["parameter_set"]["rr"] == good.rr
    assert folds[0]["test"][1]["average_r"] == -.3
    assert stress["trades"] == 120


def test_walkforward_base_orb_matches_completed_bar_strategy():
    from walkforward_standard import VARIANTS, signal_components, stop_levels

    frame = pd.concat([_bars(np.linspace(98, 99.39, 36)), _orb_buy_frame()], ignore_index=True)
    frame["time"] = pd.date_range("2025-01-01", periods=len(frame), freq="15min", tz="UTC")
    frame["point_size"] = .01
    spreads = pd.Series(2., index=frame.index)
    components = signal_components(frame, "session_orb_retest_rolling", spreads)
    sides, stops = stop_levels(frame, components, VARIANTS[0])
    single = session_orb_retest(frame, point_value=.01, spread_points=2.)
    assert single["command"] == "BUY"
    assert sides[-1] == 1 and stops[-1] == pytest.approx(single["sl"])
