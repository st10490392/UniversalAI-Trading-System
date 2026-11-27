"""Read-only broker preflight and setup generation for the demo Rolling ORB."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from math import isfinite

import pandas as pd

from research_lab import _causal_spreads
from walkforward_standard import Variant, signal_components, stop_levels

SYMBOL = "Range Break 100 Index"
PROFILE = "demo_rolling_orb"
RISK_PCT = 0.25
VARIANT = Variant(3.0, True, True)


class DemoUnavailable(ValueError):
    """A safe, user-facing reason to return WAIT."""


def _positive(*values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) and v > 0 for v in values)


def demo_broker_snapshot(mt5, account_id, server, symbol=SYMBOL):
    """Verify the connected terminal against the licensed, configured account."""
    terminal = mt5.terminal_info()
    account = mt5.account_info()
    if terminal is None or not terminal.connected or account is None:
        raise DemoUnavailable("An active connected MT5 terminal is required for broker constraints")
    if (account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO or str(account.login) != account_id
            or account.server != server):
        raise DemoUnavailable("Connected MT5 account must match the configured demo login and server")
    info = mt5.symbol_info(symbol)
    if info is None or info.name != symbol:
        raise DemoUnavailable(f"{symbol} is unavailable on the connected broker")
    if info.trade_mode != getattr(mt5, "SYMBOL_TRADE_MODE_FULL", 4):
        raise DemoUnavailable(f"{symbol} is restricted on this account")
    if not _positive(info.point, info.trade_tick_size, info.trade_contract_size,
                     info.volume_min, info.volume_max, info.volume_step) or info.volume_max < info.volume_min:
        raise DemoUnavailable("Broker contract, point, tick size or volume specification is invalid")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or not _positive(tick.bid, tick.ask) or tick.ask < tick.bid:
        raise DemoUnavailable("Broker bid/ask is unavailable or invalid")
    return account, info, tick


def demo_lots(capital, loss_per_lot, minimum, maximum, step, risk_pct=RISK_PCT):
    """Round down on the broker's minimum-anchored volume grid."""
    if not _positive(capital, loss_per_lot, minimum, maximum, step, risk_pct) or maximum < minimum:
        raise DemoUnavailable("Cannot calculate demo risk lots from broker data")
    budget = Decimal(str(capital)) * Decimal(str(risk_pct)) / 100
    affordable = min(Decimal(str(maximum)), budget / Decimal(str(loss_per_lot)))
    floor, increment = Decimal(str(minimum)), Decimal(str(step))
    if affordable < floor:
        raise DemoUnavailable(f"Broker minimum lot would exceed {risk_pct:g}% demo risk")
    return float(floor + ((affordable - floor) / increment).to_integral_value(rounding=ROUND_FLOOR) * increment)


def _price_grid(price, tick_size, rounding):
    step = Decimal(str(tick_size))
    return float((Decimal(str(price)) / step).to_integral_value(rounding=rounding) * step)


def rolling_orb_context(bars, info, tick):
    """Reuse the walk-forward signal rules on completed M15 bars."""
    if bars is None or len(bars) < 205:
        raise DemoUnavailable("Insufficient completed M15 bars for Rolling ORB and EMA-200")
    frame = pd.DataFrame(bars).rename(columns={"spread": "spread_points"})
    frame["time"] = pd.to_datetime(frame.time, unit="s", utc=True, errors="coerce")
    if frame.time.isna().any() or not frame.time.is_monotonic_increasing or frame.time.duplicated().any():
        raise DemoUnavailable("Invalid MT5 bar timestamps")
    frame["point_size"] = info.point
    spreads = _causal_spreads(frame)
    mode = "session_orb_retest_rolling"
    components = signal_components(frame, mode, spreads)
    sides, stops = stop_levels(frame, components, VARIANT)
    side = int(sides[-1])
    if not side:
        raise DemoUnavailable("No completed Rolling ORB setup")
    entry = float(tick.ask if side > 0 else tick.bid)
    stop = _price_grid(stops[-1], info.trade_tick_size, ROUND_FLOOR if side > 0 else ROUND_CEILING)
    risk = side * (entry - stop)
    target = _price_grid(entry + side * VARIANT.rr * risk, info.trade_tick_size,
                         ROUND_CEILING if side > 0 else ROUND_FLOOR)
    min_stop = max(info.trade_tick_size, info.trade_stops_level * info.point)
    # Stops are validated against the side of the quote used to close a position.
    close_quote = tick.bid if side > 0 else tick.ask
    if not _positive(stop, target, risk) or min(side * (close_quote-stop), side * (target-close_quote)) < min_stop:
        raise DemoUnavailable("Rolling ORB stop or target violates broker price geometry")
    return {
        "symbol": SYMBOL, "strategy": "Rolling ORB", "side": "BUY" if side > 0 else "SELL",
        "setup": {"rule": "rolling_orb_atr_ema200", "entry_reference": entry, "stop": stop,
                  "target": target, "risk_reward": abs(target-entry)/risk},
        "quant_metrics": {"atr14": float(components[2].iloc[-1]), "stop_points": risk/info.point,
                          "stop_ticks": risk/info.trade_tick_size},
        "last_completed_bar_utc": frame.time.iloc[-1].isoformat(),
        "bar_time": int(frame.time.iloc[-1].timestamp()),
    }


def check_demo_volume(mt5, account, info, context, risk_pct=RISK_PCT):
    setup = context["setup"]
    action = mt5.ORDER_TYPE_BUY if context["side"] == "BUY" else mt5.ORDER_TYPE_SELL
    loss = mt5.order_calc_profit(action, context.get("symbol", SYMBOL), info.volume_min, setup["entry_reference"], setup["stop"])
    if loss is None or not _positive(-loss):
        raise DemoUnavailable("MT5 cannot calculate account-currency stop loss for this symbol")
    if not _positive(account.balance, account.equity):
        raise DemoUnavailable("Demo balance or equity is unavailable")
    return demo_lots(min(account.balance, account.equity), -loss/info.volume_min,
                     info.volume_min, info.volume_max, info.volume_step, risk_pct)
