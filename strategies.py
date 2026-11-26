"""Deterministic completed-bar strategies and research performance filter.

This module is deliberately independent from live routing.  Callers must supply
only completed OHLC bars; both strategies return a JSON-compatible, fail-closed
signal dictionary and never modify the approval registry on disk.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


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
    if not np.isfinite(values).all() or (renamed.close <= 0).any() or (renamed.high < renamed.low).any():
        return None, _make_wait("Invalid price data", setup)
    if (renamed.high < renamed[["open", "close"]].max(axis=1)).any() or (renamed.low > renamed[["open", "close"]].min(axis=1)).any():
        return None, _make_wait("Invalid OHLC candle bounds", setup)
    return renamed, None


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
) -> dict[str, Any]:
    """Opening-range breakout, pullback, and completed-bar resumption signal."""
    setup = "session_orb_retest"
    if orb_lookback < 1 or atr_period < 1 or rr_ratio <= 0 or not _parameters_valid(min_range_atr_mult, point_value, spread_points):
        return _make_wait("Invalid strategy parameters", setup)
    frame, error = _normalise_ohlc(df, max(orb_lookback + atr_period + 5, 30), setup)
    if error:
        return error
    assert frame is not None
    frame["atr"] = _atr(frame, atr_period)
    frame = frame.dropna(subset=["atr"]).reset_index(drop=True)
    if len(frame) < orb_lookback + 3:
        return _make_wait("Insufficient bars after ATR calculation", setup)

    opening_range = frame.iloc[-(orb_lookback + 3):-3]
    orb_high, orb_low = float(opening_range.high.max()), float(opening_range.low.min())
    atr = float(frame.atr.iloc[-1])
    if not np.isfinite(atr) or atr <= 0 or orb_high - orb_low < atr * min_range_atr_mult:
        return _make_wait("ORB range too small or zero ATR (choppy/flat)", setup)

    breakout, pullback, trigger = frame.iloc[-3], frame.iloc[-2], frame.iloc[-1]
    spread = spread_points * point_value
    if breakout.close > orb_high and orb_low <= pullback.low <= orb_high and trigger.close > trigger.open and trigger.close > orb_high:
        sl = float(pullback.low - spread)
        risk = float(trigger.close - sl)
        if risk <= 0:
            return _make_wait("Invalid SL/Risk geometry for BUY", setup)
        confidence = min(1.0, .75 + (float(trigger.close) - orb_high) / atr * .1)
        return {"command": "BUY", "sl": sl, "tp": float(trigger.close + risk * rr_ratio), "confidence": round(float(confidence), 3), "setup": setup, "reason": "OK"}
    if breakout.close < orb_low and orb_low <= pullback.high <= orb_high and trigger.close < trigger.open and trigger.close < orb_low:
        sl = float(pullback.high + spread)
        risk = float(sl - trigger.close)
        if risk <= 0:
            return _make_wait("Invalid SL/Risk geometry for SELL", setup)
        confidence = min(1.0, .75 + (orb_low - float(trigger.close)) / atr * .1)
        return {"command": "SELL", "sl": sl, "tp": float(trigger.close - risk * rr_ratio), "confidence": round(float(confidence), 3), "setup": setup, "reason": "OK"}
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
    if fast_period < 1 or slow_period < fast_period or slope_lookback < 1 or rr_ratio <= 0 or not _parameters_valid(min_separation_atr_mult, point_value, spread_points):
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
        risk = float(current.close - sl)
        if risk <= 0:
            return _make_wait("Invalid BUY risk geometry", setup)
        return {"command": "BUY", "sl": sl, "tp": float(current.close + risk * rr_ratio), "confidence": round(float(min(1.0, .75 + separation / atr * .05)), 3), "setup": setup, "reason": "OK"}
    if current.ema_fast < current.ema_slow and fast_slope < 0 and slow_slope < 0 and previous.high >= previous.ema_fast and current.close < current.open and current.close < current.ema_fast:
        sl = float(previous.high + spread)
        risk = float(sl - current.close)
        if risk <= 0:
            return _make_wait("Invalid SELL risk geometry", setup)
        return {"command": "SELL", "sl": sl, "tp": float(current.close - risk * rr_ratio), "confidence": round(float(min(1.0, .75 + separation / atr * .05)), 3), "setup": setup, "reason": "OK"}
    return _make_wait("No trend pullback + resumption pattern", setup)


class PairPerformanceFilter:
    """In-memory, deterministic pair filter for research/manual review only."""

    def __init__(self, min_trades: int = 80, min_pf: float = 1.25, max_dd_pct: float = 10.0, min_cost_stressed_pf: float = 1.10):
        if min_trades < 1 or min_pf <= 0 or max_dd_pct <= 0 or min_cost_stressed_pf <= 0:
            raise ValueError("Performance-filter thresholds must be positive")
        self.min_trades, self.min_pf = min_trades, min_pf
        self.max_dd_pct, self.min_cost_stressed_pf = max_dd_pct, min_cost_stressed_pf
        self.registry: dict[tuple[str, str], dict[str, Any]] = {}

    def evaluate_metrics(self, symbol: str, strategy: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {"symbol": symbol, "strategy": strategy, "approved": False, "regime": "BLOCKED", "failures": []}
        required = ("trades", "pf", "max_dd_pct", "cost_stressed_pf")
        for key in required:
            value = metrics.get(key) if isinstance(metrics, Mapping) else None
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
                result["failures"].append(f"Missing or invalid metric: {key}")
        if result["failures"]:
            self.registry[(symbol, strategy)] = result
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
        self.registry[(symbol, strategy)] = result
        return result

    def is_approved(self, symbol: str, strategy: str) -> bool:
        return bool(self.registry.get((symbol, strategy), {}).get("approved", False))

    def get_registry_export(self) -> list[dict[str, Any]]:
        return list(self.registry.values())
