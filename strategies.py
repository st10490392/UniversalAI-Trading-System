"""Deterministic completed-bar strategies and research performance filter.

This module is deliberately independent from live routing.  Callers must supply
only completed OHLC bars; both strategies return a JSON-compatible, fail-closed
signal dictionary and never modify the approval registry on disk.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


MAX_GAP_BARS = 6


SIGNAL_SCHEMA = {
    "command": str,
    "sl": float,
    "tp": float,
    "confidence": float,
    "setup": str,
    "reason": str,
}


def _make_wait(reason: str, setup: str) -> dict[str, Any]:
    return {"command": "WAIT", "sl": 0.0, "tp": 0.0, "confidence": 0.0, "setup": setup, "reason": reason}


def _normalise_ohlc(df: pd.DataFrame | None, min_bars: int, setup: str) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Return a finite lower-case OHLC frame or a deterministic WAIT result."""
    if df is None or len(df) < min_bars:
        return None, _make_wait(f"Insufficient bars: need {min_bars}, got {len(df) if df is not None else 0}", setup)
    renamed = df.rename(columns=lambda column: str(column).lower()).copy()
    required = {"open", "high", "low", "close"}
    if not required.issubset(renamed.columns):
        return None, _make_wait("Missing OHLC columns", setup)
    try:
        for column in required:
            renamed[column] = pd.to_numeric(renamed[column], errors="raise")
    except (TypeError, ValueError):
        return None, _make_wait("Non-numeric OHLC data", setup)
    values = renamed[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any() or (renamed.high < renamed.low).any():
        return None, _make_wait("Invalid price data", setup)
    if (renamed.high < renamed[["open", "close"]].max(axis=1)).any() or (renamed.low > renamed[["open", "close"]].min(axis=1)).any():
        return None, _make_wait("Invalid OHLC candle bounds", setup)
    stamps = renamed.get("time")
    if stamps is None and isinstance(renamed.index, pd.DatetimeIndex):
        stamps = renamed.index.to_series()
    if stamps is not None:
        # MT5 integer timestamps are Unix seconds.
        unit = "s" if pd.api.types.is_numeric_dtype(stamps) else None
        parsed = pd.to_datetime(stamps, errors="coerce", utc=True, unit=unit)
        if parsed.isna().any() or not parsed.is_monotonic_increasing or parsed.duplicated().any():
            return None, _make_wait("Invalid timestamps: NaT, duplicate or non-increasing", setup)
        steps = parsed.diff().dropna()
        if len(steps) and steps.max() > max(steps.median() * MAX_GAP_BARS, pd.Timedelta(days=4)):
            return None, _make_wait("Timestamp gap too large inside the lookback window", setup)
        renamed["time"] = parsed.array
    return renamed, None


def validate_signal(signal: Any) -> dict[str, Any]:
    """Validate the deterministic strategy envelope and stop/target ordering."""
    setup = str(signal.get("setup", "unknown")) if isinstance(signal, Mapping) else "unknown"
    wait = _make_wait("Signal failed schema validation", setup)
    if not isinstance(signal, Mapping) or set(signal) != set(SIGNAL_SCHEMA):
        return wait
    for key, kind in SIGNAL_SCHEMA.items():
        value = signal[key]
        if not isinstance(value, kind) or (kind is float and not np.isfinite(value)):
            return wait
    command, sl, tp, confidence = (signal[k] for k in ("command", "sl", "tp", "confidence"))
    if not signal["reason"].strip() or not signal["setup"].strip():
        return wait
    if command == "WAIT":
        return dict(signal) if (sl, tp, confidence) == (0.0, 0.0, 0.0) else wait
    if not ((command == "BUY" and sl < tp) or (command == "SELL" and tp < sl)):
        return wait
    return dict(signal) if min(sl, tp) > 0 and 0 <= confidence <= 1 else wait


def _atr(frame: pd.DataFrame, period: int) -> pd.Series:
    previous = frame.close.shift(1)
    tr = pd.concat((frame.high - frame.low, (frame.high - previous).abs(), (frame.low - previous).abs()), axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _parameters_valid(*values: float) -> bool:
    return all(np.isfinite(value) and value >= 0 for value in values)


def session_orb_retest(
    df: pd.DataFrame,
    orb_lookback: int = 6,
    atr_period: int = 14,
    min_range_atr_mult: float = 0.5,
    rr_ratio: float = 1.5,
    point_value: float = 1.0,
    spread_points: float = 0.0,
    *,
    session_start_utc: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Opening-range breakout, pullback, and completed-bar resumption signal."""
    setup = "session_orb_retest"
    if orb_lookback < 1 or atr_period < 1 or rr_ratio <= 0 or not _parameters_valid(rr_ratio, min_range_atr_mult, point_value, spread_points):
        return _make_wait("Invalid strategy parameters", setup)
    frame, error = _normalise_ohlc(df, max(orb_lookback + atr_period + 5, 30), setup)
    if error:
        return error
    assert frame is not None
    timestamped_frame = frame
    frame["atr"] = _atr(frame, atr_period)
    frame = frame.dropna(subset=["atr"]).reset_index(drop=True)
    if len(frame) < orb_lookback + 3:
        return _make_wait("Insufficient bars after ATR calculation", setup)

    opening_range = frame.iloc[-(orb_lookback + 3):-3]
    if session_start_utc is not None:
        anchor = pd.to_datetime(session_start_utc, errors="coerce", utc=True)
        if pd.isna(anchor) or "time" not in frame:
            return _make_wait("Session anchor requires valid timestamps", setup)
        session = timestamped_frame.loc[timestamped_frame.time >= anchor]
        if len(session) < orb_lookback + 3 or session.time.iloc[0] != anchor:
            return _make_wait("Session opening range or retest is incomplete", setup)
        opening_range = session.iloc[:orb_lookback]
    orb_high, orb_low = float(opening_range.high.max()), float(opening_range.low.min())
    atr = float(frame.atr.iloc[-1])
    if not np.isfinite(atr) or atr <= 0 or orb_high - orb_low < atr * min_range_atr_mult:
        return _make_wait("ORB range too small or zero ATR (choppy/flat)", setup)

    breakout, pullback, trigger = frame.iloc[-3], frame.iloc[-2], frame.iloc[-1]
    spread = spread_points * point_value
    if breakout.close > orb_high and orb_low <= pullback.low <= orb_high and trigger.close > trigger.open and trigger.close > orb_high:
        sl = float(pullback.low - spread)
        entry = float(trigger.close) + spread
        risk = entry - sl
        if risk <= 0:
            return _make_wait("Invalid SL/Risk geometry for BUY", setup)
        confidence = min(1.0, .75 + (float(trigger.close) - orb_high) / atr * .1)
        return validate_signal({"command": "BUY", "sl": sl, "tp": float(entry + risk * rr_ratio), "confidence": round(float(confidence), 3), "setup": setup, "reason": "OK"})
    if breakout.close < orb_low and orb_low <= pullback.high <= orb_high and trigger.close < trigger.open and trigger.close < orb_low:
        sl = float(pullback.high + spread)
        risk = float(sl - trigger.close)
        if risk <= 0:
            return _make_wait("Invalid SL/Risk geometry for SELL", setup)
        confidence = min(1.0, .75 + (orb_low - float(trigger.close)) / atr * .1)
        return validate_signal({"command": "SELL", "sl": sl, "tp": float(trigger.close - risk * rr_ratio + spread), "confidence": round(float(confidence), 3), "setup": setup, "reason": "OK"})
    return _make_wait("No valid ORB breakout + retest pattern detected", setup)


def ema_trend_pullback(
    df: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    slope_lookback: int = 5,
    min_separation_atr_mult: float = 0.2,
    rr_ratio: float = 1.5,
    point_value: float = 1.0,
    spread_points: float = 0.0,
) -> dict[str, Any]:
    """EMA trend pullback and completed-bar resumption signal."""
    setup = "ema_trend_pullback"
    if fast_period < 1 or slow_period < fast_period or slope_lookback < 1 or rr_ratio <= 0 or not _parameters_valid(rr_ratio, min_separation_atr_mult, point_value, spread_points):
        return _make_wait("Invalid strategy parameters", setup)
    frame, error = _normalise_ohlc(df, slow_period + slope_lookback + 5, setup)
    if error:
        return error
    assert frame is not None
    frame["ema_fast"] = frame.close.ewm(span=fast_period, adjust=False).mean()
    frame["ema_slow"] = frame.close.ewm(span=slow_period, adjust=False).mean()
    frame["atr"] = _atr(frame, 14)
    frame = frame.dropna(subset=["atr"]).reset_index(drop=True)
    if len(frame) < slope_lookback + 2:
        return _make_wait("Insufficient data after indicator calculation", setup)

    current, previous = frame.iloc[-1], frame.iloc[-2]
    atr = float(current.atr)
    if not np.isfinite(atr) or atr <= 0:
        return _make_wait("Zero ATR (flat market)", setup)
    separation = abs(float(current.ema_fast - current.ema_slow))
    if separation < atr * min_separation_atr_mult:
        return _make_wait("EMAs tangled; no clear trend separation", setup)
    fast_slope = float(current.ema_fast - frame.iloc[-1 - slope_lookback].ema_fast) / slope_lookback
    slow_slope = float(current.ema_slow - frame.iloc[-1 - slope_lookback].ema_slow) / slope_lookback
    spread = spread_points * point_value
    if current.ema_fast > current.ema_slow and fast_slope > 0 and slow_slope > 0 and previous.low <= previous.ema_fast and current.close > current.open and current.close > current.ema_fast:
        sl = float(previous.low - spread)
        entry = float(current.close) + spread
        risk = entry - sl
        if risk <= 0:
            return _make_wait("Invalid BUY risk geometry", setup)
        return validate_signal({"command": "BUY", "sl": sl, "tp": float(entry + risk * rr_ratio), "confidence": round(float(min(1.0, .75 + separation / atr * .05)), 3), "setup": setup, "reason": "OK"})
    if current.ema_fast < current.ema_slow and fast_slope < 0 and slow_slope < 0 and previous.high >= previous.ema_fast and current.close < current.open and current.close < current.ema_fast:
        sl = float(previous.high + spread)
        risk = float(sl - current.close)
        if risk <= 0:
            return _make_wait("Invalid SELL risk geometry", setup)
        return validate_signal({"command": "SELL", "sl": sl, "tp": float(current.close - risk * rr_ratio + spread), "confidence": round(float(min(1.0, .75 + separation / atr * .05)), 3), "setup": setup, "reason": "OK"})
    return _make_wait("No trend pullback + resumption pattern", setup)


def _nonnegative_number(value: Any) -> bool:
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value) and value >= 0)


class PairPerformanceFilter:
    """In-memory, deterministic pair filter for research/manual review only."""

    def __init__(self, min_trades: int = 80, min_pf: float = 1.25, max_dd_pct: float = 10.0, min_cost_stressed_pf: float = 1.10):
        limits = (min_trades, min_pf, max_dd_pct, min_cost_stressed_pf)
        if not all(_nonnegative_number(value) and value > 0 for value in limits) or min_trades < 1:
            raise ValueError("Performance-filter thresholds must be positive")
        self.min_trades, self.min_pf = min_trades, min_pf
        self.max_dd_pct, self.min_cost_stressed_pf = max_dd_pct, min_cost_stressed_pf
        self.registry: dict[tuple[str, str], dict[str, Any]] = {}

    def evaluate_metrics(self, symbol: str, strategy: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {"symbol": symbol, "strategy": strategy, "approved": False, "live_approval": False, "regime": "BLOCKED", "failures": []}
        required = ("trades", "pf", "max_dd_pct", "cost_stressed_pf")
        for key in required:
            value = metrics.get(key) if isinstance(metrics, Mapping) else None
            if not _nonnegative_number(value):
                result["failures"].append(f"Missing or invalid metric: {key}")
        if result["failures"]:
            self.registry[(symbol, strategy)] = {**result, "failures": result["failures"].copy()}
            return result
        if metrics["trades"] < self.min_trades:
            result["failures"].append(f"Trades {metrics['trades']} < {self.min_trades}")
        if metrics["pf"] < self.min_pf:
            result["failures"].append(f"PF {metrics['pf']:.2f} < {self.min_pf}")
        if metrics["max_dd_pct"] > self.max_dd_pct:
            result["failures"].append(f"MaxDD {metrics['max_dd_pct']:.2f}% > {self.max_dd_pct}%")
        if metrics["cost_stressed_pf"] < self.min_cost_stressed_pf:
            result["failures"].append(f"Cost PF {metrics['cost_stressed_pf']:.2f} < {self.min_cost_stressed_pf}")
        if not result["failures"]:
            result.update(approved=True, regime="ACTIVE")
        self.registry[(symbol, strategy)] = {**result, "failures": result["failures"].copy()}
        return result

    def is_approved(self, symbol: str, strategy: str) -> bool:
        return bool(self.registry.get((symbol, strategy), {}).get("approved", False))

    def get_registry_export(self) -> list[dict[str, Any]]:
        return [{**row, "failures": row["failures"].copy()} for row in self.registry.values()]
