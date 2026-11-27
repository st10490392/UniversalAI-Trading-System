from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research_lab import _trade_one, add_indicators, build_signals, load_history, simulate
from strategy_audit_variants import Market, nonoverlap, orb_choices, session_ranges


def prices(size=5):
    return pd.DataFrame({"time": pd.date_range("2025-01-06", periods=size, freq="15min", tz="UTC"),
                         "open": 100., "high": 101., "low": 99., "close": 100.,
                         "tick_size": 0., "tick_value": 0., "point_size": 1., "spread_points": 0.})


def test_load_history_uses_exported_point_and_volume(tmp_path: Path):
    frame = prices(300).drop(columns=["point_size"])
    frame["point"] = .01
    frame["volume"] = 123
    path = tmp_path / "market.csv"
    frame.to_csv(path, index=False)
    data = load_history(path)
    assert data.point_size.eq(.01).all() and data.tick_volume.eq(123).all()


def test_judas_asian_extreme_survives_post_session_nan():
    data = prices(100)
    data.time = pd.date_range("2025-01-06 05:00", periods=len(data), freq="15min", tz="UTC")
    data.loc[30, ["high", "low", "close"]] = [102., 98., 98.5]
    side, _, _, labels = build_signals(add_indicators(data), "EURUSD", "B")
    assert side[30] == -1 and labels[30] == "judas_sweep_mss_short"
    assert not np.any(side[:28])


def test_cutoff_uses_actual_timestamp_units():
    data = prices()
    data.time = data.time.astype("datetime64[us, UTC]")
    data.time += pd.Timedelta(hours=23, minutes=30)
    sides = np.array([0, 1, 0, 0, 0])
    stops = np.full(5, 5.)
    targets = np.full(5, np.nan)
    # The signal at 23:45 would enter at midnight, outside the slice.
    assert simulate(data, (sides, stops, targets, sides), start="2025-01-06", end="2025-01-07") == []


def test_session_close_uses_exit_bar_point_for_short_costs():
    data = prices(3)
    data.loc[2, ["open", "high", "low", "close", "point_size", "spread_points"]] = [96., 97., 94., 95., .5, 2.]
    trade = _trade_one(data, 0, -1, 110., np.nan, absolute_stop=True, reward_risk=None,
                       end_idx=3, breakeven=False, commission_per_lot=0., commission_points=0.,
                       slippage_points=2.)
    assert trade["entry"] == 98 and trade["exit"] == 97
    assert trade["r"] == pytest.approx(1/12)
    assert trade["exit_index"] == 2


def test_orb_reentries_require_fresh_close_crossing():
    close = np.array([100., 102., 103., 100., 102., 102.])
    high, low = close + .25, close - .25
    indicators = (close, high, low, close, close, close, np.full(6, .5), np.ones(6, bool),
                  np.zeros(6), np.ones(6, bool), np.arange(6)*900)
    assert [i for i, _, _ in orb_choices((0, 6, 6), 1, False, True, indicators)] == [1, 4]


def test_local_us_open_shifts_with_dst_and_requires_cutoff_bar():
    market = Market("US cash", "America/New_York", 570, 720, 960)
    winter = pd.date_range("2025-01-06 14:30", periods=26, freq="15min", tz="UTC")
    summer = pd.date_range("2025-07-07 13:30", periods=26, freq="15min", tz="UTC")
    data = pd.DataFrame({"time": winter.append(summer)})
    assert list(session_ranges(data, market).values()) == [(0, 10, 26), (26, 36, 52)]
    assert len(session_ranges(data.iloc[:-1], market)) == 1


def test_combined_execution_does_not_overlap_even_on_exit_bar():
    first = {"entry_index": 2, "exit_index": 6}
    overlaps = {"entry_index": 6, "exit_index": 8}
    distinct = {"entry_index": 7, "exit_index": 9}
    assert list(nonoverlap([first], [overlaps, distinct])) == [first, distinct]


@pytest.mark.parametrize(("stop", "target", "rr", "side"), [
    (105., -1., 2., -1),
    (95., np.nan, 0., 1),
    (200., np.nan, 2., 1),
    (1000., np.nan, 2., -1),
])
def test_trade_rejects_invalid_stop_target_or_reward_risk(stop, target, rr, side):
    assert _trade_one(prices(), 0, side, stop, target, absolute_stop=True, reward_risk=rr,
                      end_idx=4, breakeven=False, commission_per_lot=0., commission_points=0.,
                      slippage_points=0.) is None

def test_risk_gate_requires_parameter_and_source_replication():
    from risk_gate_audit import audit
    common = dict(symbol="XAUUSD", strategy="intraday_range_break", variant="4_bar_close",
                  session="NY killzone UTC", max_trades_per_session=1, folds_available=2,
                  oos_trades=60, stress_trades=60, avg_net_r=.25, stressed_avg_net_r=.18,
                  max_drawdown_pct=12., baseline_pf=1.6, stressed_pf=1.3)
    rows = pd.DataFrame([{**common, "source":source, "exit_profile":target}
                         for source in ("old","recent") for target in ("2R","3R")])
    assert audit(rows).research_pass.all()
    rows.loc[0,"stressed_pf"]=1.0
    failed=audit(rows)
    assert not failed.research_pass.any()
    assert "stressed PF" in failed.failures.iloc[0]


def test_regime_and_rsi_use_only_completed_history():
    from research_lab import add_indicators
    closes=np.arange(1.,81.)
    data=pd.DataFrame({"open":closes,"close":closes,"high":closes+.1,"low":closes-.1,
                       "time":pd.date_range("2025-01-01",periods=80,freq="15min",tz="UTC")})
    short=add_indicators(data.iloc[:65]).iloc[-1]
    full=add_indicators(data).iloc[64]
    assert short.market_regime==full.market_regime=="trend"
    assert short.rsi==full.rsi==100.
