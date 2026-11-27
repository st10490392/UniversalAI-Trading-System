from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import brain
import demo_orb


def broker():
    info = NS(name=demo_orb.SYMBOL, point=.1, trade_tick_size=.1, trade_contract_size=1.,
              trade_stops_level=2, volume_min=.01, volume_max=10., volume_step=.01, trade_mode=4)
    account = NS(login=42, server="Test-Demo", trade_mode=0, equity=10000., balance=12000.)
    mt5 = Mock(ACCOUNT_TRADE_MODE_DEMO=0, SYMBOL_TRADE_MODE_FULL=4, TIMEFRAME_M15=15,
               ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1)
    mt5.terminal_info.return_value = NS(connected=True)
    mt5.account_info.return_value = account
    mt5.symbol_info.return_value = info
    mt5.symbol_info_tick.return_value = NS(ask=100.2, bid=100.)
    mt5.order_calc_profit.return_value = -.20
    return mt5, account, info


def test_demo_broker_constraint_and_volume_validation():
    mt5, account, info = broker()
    assert demo_orb.demo_broker_snapshot(mt5, "42", "Test-Demo")[:2] == (account, info)
    assert demo_orb.check_demo_volume(mt5, account, info, {
        "side": "BUY", "setup": {"entry_reference": 100., "stop": 96.},
    }) == 1.25
    assert demo_orb.demo_lots(1000., 18., .02, 10., .03) == .11
    with pytest.raises(demo_orb.DemoUnavailable, match="minimum lot"):
        demo_orb.demo_lots(10., 100., .01, 10., .01)
    with pytest.raises(demo_orb.DemoUnavailable, match="demo login and server"):
        demo_orb.demo_broker_snapshot(mt5, "43", "Test-Demo")
    mt5.terminal_info.return_value.connected = False
    with pytest.raises(demo_orb.DemoUnavailable, match="active connected MT5"):
        demo_orb.demo_broker_snapshot(mt5, "42", "Test-Demo")
    mt5.terminal_info.return_value.connected = True
    mt5.symbol_info.return_value = None
    with pytest.raises(demo_orb.DemoUnavailable, match="unavailable"):
        demo_orb.demo_broker_snapshot(mt5, "42", "Test-Demo")
    mt5.symbol_info.return_value = info
    info.volume_step = 0.
    with pytest.raises(demo_orb.DemoUnavailable, match="specification"):
        demo_orb.demo_broker_snapshot(mt5, "42", "Test-Demo")
    mt5.order_send.assert_not_called()


def test_rolling_orb_atr_ema200_maps_completed_bars_to_3r():
    mt5, _, info = broker()
    trend = np.linspace(80., 99., 220)
    opening = [100.] * 6
    closes = trend.tolist() + opening + [100.6, 100.1, 100.7]
    frame = pd.DataFrame({
        "time": np.arange(len(closes)) * 900 + 1700000000,
        "open": closes, "high": np.array(closes) + .15,
        "low": np.array(closes) - .15, "close": closes, "spread": 2,
    })
    frame.loc[220:225, ["low", "high"]] = [99.6, 100.4]
    frame.loc[len(frame)-1, ["open", "low"]] = [100.1, 100.0]
    tick = NS(ask=100.9, bid=100.7)
    context = demo_orb.rolling_orb_context(frame, info, tick)
    assert context["symbol"] == demo_orb.SYMBOL and context["side"] == "BUY"
    assert context["strategy"] == "Rolling ORB" and context["setup"]["risk_reward"] >= 3.
    assert context["bar_time"] == frame.time.iloc[-1]
    assert context["quant_metrics"]["stop_points"] > 0
    with pytest.raises(demo_orb.DemoUnavailable, match="Insufficient"):
        demo_orb.rolling_orb_context(frame.iloc[-100:], info, mt5.symbol_info_tick.return_value)


def test_demo_evaluator_flag_account_mode_and_validator(monkeypatch):
    import sys

    mt5, account, _ = broker()
    monkeypatch.setitem(sys.modules, "MetaTrader5", mt5)
    broker_account = {"account_type": "broker", "risk_pct": .25, "broker_server": account.server}
    monkeypatch.delenv("ENABLE_DEMO_ROLLING_ORB", raising=False)
    monkeypatch.setattr(brain, "_demo_orb_cache", {})
    llm = Mock(return_value={"command": "BUY", "sl": 96., "tp": 112., "confidence": .9,
                             "setup": "Rolling ORB", "reason": "Mock verified"})
    monkeypatch.setattr(brain, "evaluate_llm_signal", llm)
    assert brain.evaluate_demo_orb("42", demo_orb.SYMBOL, broker_account)["command"] == "WAIT"
    mt5.initialize.assert_not_called()
    monkeypatch.setenv("ENABLE_DEMO_ROLLING_ORB", "1")
    for wrong in ({"risk_pct": .5}, {"account_type": "prop"}):
        assert brain.evaluate_demo_orb("42", demo_orb.SYMBOL, {**broker_account, **wrong})["command"] == "WAIT"
    assert brain.evaluate_demo_orb("42", "XAUUSD", broker_account)["command"] == "WAIT"
    mt5.initialize.assert_not_called()
    for mode in (1, 2):
        account.trade_mode = mode
        result = brain.evaluate_demo_orb("42", demo_orb.SYMBOL, broker_account)
        assert result["command"] == "WAIT" and "configured demo" in result["reason"]
    mt5.copy_rates_from_pos.assert_not_called()
    llm.assert_not_called()
    account.trade_mode = 0
    monkeypatch.setattr(demo_orb, "rolling_orb_context", Mock(return_value={
        "symbol": demo_orb.SYMBOL, "strategy": "Rolling ORB", "side": "BUY",
        "setup": {"entry_reference": 100., "stop": 96., "target": 112.}, "bar_time": int(brain.time.time())-901,
    }))
    assert brain.validate_signal(llm.return_value) == llm.return_value
    assert brain._valid_signal_risk(llm.return_value, 100., "BUY")
    signal = brain.evaluate_demo_orb("42", demo_orb.SYMBOL, broker_account)
    assert signal["command"] == signal["decision"] == "BUY"
    assert signal["execution_profile"] == demo_orb.PROFILE and signal["entry"] == 100.
    assert signal["demo_preflight_lots"] == 1.25
    assert mt5.copy_rates_from_pos.call_args.args == (demo_orb.SYMBOL, 15, 1, 500)
    mt5.order_send.assert_not_called()



def test_ea_poll_requires_explicit_demo_profile(monkeypatch):
    import sqlite3

    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        monkeypatch.setattr(brain, "_connect", lambda: connection)
        monkeypatch.setattr(brain, "_ea_authorized", lambda *args: True)
        brain.init_db()
        connection.execute("""INSERT INTO accounts
            (account_id, broker_server, account_type, risk_pct, max_daily_dd_pct, max_total_dd_pct, active)
            VALUES ('42', 'Test-Demo', 'broker', .25, 5, 10, 1)""")
        normal = Mock(return_value=None)
        demo = Mock(return_value=brain._wait_signal("Offline"))
        monkeypatch.setattr(brain, "analyze_symbol", normal)
        monkeypatch.setattr(brain, "evaluate_demo_orb", demo)
        args = ("42", demo_orb.SYMBOL, 10000., 10000., "mock-key", "mock-device")
        assert brain.ea_poll(*args)["command"] == "WAIT"
        normal.assert_called_once()
        demo.assert_not_called()
        wait = brain.ea_poll(*args, demo_orb_profile=True)
        demo.assert_called_once()
        assert normal.call_count == 1
        assert wait["reason"] == "Offline" and wait["risk_pct"] == .25
        assert wait["sl"] == wait["tp"] == wait["confidence"] == 0.
