"""Frozen XAUUSD research rule shared by the demo setup and fold audit."""
import numpy as np
import pandas as pd

SYMBOL = "XAUUSD"
PROFILE = "demo_xau_ny_breakout"


def breakout(data):
    close = data.close
    ema = close.ewm(span=200, adjust=False, min_periods=200).mean()
    high = data.high.shift().rolling(4).max()
    low = data.low.shift().rolling(4).min()
    active = data.atr_ratio > 1.05
    long = active & (close > high) & (close > ema) & (ema.diff(5) > 0)
    short = active & (close < low) & (close < ema) & (ema.diff(5) < 0)
    side = np.select([long, short], [1, -1], default=0)
    return side, close.to_numpy() - side * data.atr14.to_numpy()


def entry_window(entry):
    return entry.weekday() < 5 and 13 * 60 + 15 <= entry.hour * 60 + entry.minute < 16 * 60

def broker_context(bars, info, tick, now):
    from decimal import ROUND_CEILING, ROUND_FLOOR
    from demo_orb import DemoUnavailable, _positive, _price_grid
    from research_lab import add_indicators

    frame = pd.DataFrame(bars)
    if len(frame) < 205 or not {"time","open","high","low","close"}.issubset(frame):
        raise DemoUnavailable("Insufficient completed XAUUSD bars")
    frame["time"] = pd.to_datetime(frame.time, unit="s", utc=True, errors="coerce")
    times = frame.time
    prices = frame[["open","high","low","close"]]
    if (times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing
            or not np.isfinite(prices.to_numpy()).all() or (prices<=0).any().any()
            or (frame.low>prices[["open","close"]].min(axis=1)).any()
            or (frame.high<prices[["open","close"]].max(axis=1)).any()):
        raise DemoUnavailable("Invalid XAUUSD M15 bars")
    bar = int(times.iloc[-1].timestamp())
    entry_time = pd.Timestamp(bar+900,unit="s",tz="UTC")
    if not 900 <= now-bar < 1800 or not entry_window(entry_time):
        raise DemoUnavailable("Outside XAUUSD entry window or stale M15 bar")
    if not (times.diff().iloc[-4:].dt.total_seconds()==900).all():
        raise DemoUnavailable("Incomplete XAUUSD breakout lookback")
    if not 0 <= now-tick.time < 60 or info.trade_stops_level < 0:
        raise DemoUnavailable("Stale XAUUSD quote or invalid broker stop level")
    data = add_indicators(frame)
    sides, stops = breakout(data)
    side = int(sides[-1])
    if not side:
        raise DemoUnavailable("No completed XAUUSD breakout")
    entry = tick.ask if side>0 else tick.bid
    stop = _price_grid(stops[-1],info.trade_tick_size,ROUND_FLOOR if side>0 else ROUND_CEILING)
    risk = side*(entry-stop)
    target = _price_grid(entry+side*3*risk,info.trade_tick_size,ROUND_CEILING if side>0 else ROUND_FLOOR)
    quote = tick.bid if side>0 else tick.ask
    minimum = max(info.point*info.trade_stops_level,info.trade_tick_size)
    if not _positive(stop,risk,target) or min(side*(quote-stop),side*(target-quote))<minimum:
        raise DemoUnavailable("XAUUSD levels violate broker stop geometry")
    return {"symbol":SYMBOL,"strategy":"XAUUSD NY 4-bar breakout","side":"BUY" if side>0 else "SELL",
            "bar_time":bar,"last_completed_bar_utc":times.iloc[-1].isoformat(),
            "setup":{"entry_reference":entry,"stop":stop,"target":target,"risk_reward":abs(target-entry)/risk}}
