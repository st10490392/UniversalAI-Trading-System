import numpy as np
import pandas as pd
import pytest

import brain
from backtest import _resample_h1, _walk_exit, run_backtest


def test_quant_gate_and_directional_z_filter():
    n=100;c=np.linspace(100,110,n)
    df=pd.DataFrame({"open":c,"high":c+.2,"low":c-.2,"close":c})
    m=brain.quant_metrics(df)
    assert m["ready"] and m["gate"]
    assert not brain.quant_allows({**m,"zscore":2.01},"BUY")
    assert not brain.quant_allows({**m,"zscore":-2.01},"SELL")
    assert brain.quant_allows({**m,"zscore":0.0},"BUY")


def test_bullish_sweep_followed_by_fvg():
    d=pd.DataFrame({"open":[10.]*26,"high":[11.]*26,"low":[9.]*26,"close":[10.]*26})
    d.loc[23,["open","high","low","close"]]=[9.5,10.1,8.0,9.5]
    d.loc[24,["open","high","low","close"]]=[10.2,10.4,10.0,10.3]
    d.loc[25,["open","high","low","close"]]=[10.7,11.0,10.5,10.8]
    setup=brain.detect_smc(d,"BULLISH")
    assert setup and setup["side"]=="BUY"
    assert setup["sweep_bar"]==23 and setup["fvg_bar"]==25


@pytest.mark.parametrize("side,values,expected",[
    ("BUY",{"decision":"BUY","confidence":.75,"entry":10,"sl":9,"tp":12.5},True),
    ("BUY",{"decision":"BUY","confidence":.74,"entry":10,"sl":9,"tp":12.5},False),
    ("SELL",{"decision":"SELL","confidence":.9,"entry":10,"sl":11,"tp":7.5},True),
    ("BUY",{"decision":"SELL","confidence":.9,"entry":10,"sl":9,"tp":12.5},False),
    ("BUY",{"decision":"BUY","confidence":.9,"entry":10,"sl":9,"tp":12.4},False),
])
def test_decision_validation(side,values,expected):
    assert brain.validate_decision(values,side) is expected


def test_synthetic_detection_and_utc_sessions():
    from datetime import datetime,timezone
    assert brain.is_synthetic("Volatility 75 Index")
    assert not brain.is_synthetic("EURUSD")
    assert brain.in_kill_zone(datetime(2025,1,1,8,0,tzinfo=timezone.utc))
    assert not brain.in_kill_zone(datetime(2025,1,1,11,0,tzinfo=timezone.utc))


def test_m15_export_times_resample_to_completed_h1_bars():
    times=pd.date_range("2025-01-01T09:00:00Z",periods=8,freq="15min")
    close=np.arange(8,dtype=float)+10
    df=pd.DataFrame({"time":times,"open":close,"high":close+1,"low":close-1,"close":close})
    hourly=_resample_h1(df)
    assert hourly.iloc[0].time==pd.Timestamp("2025-01-01T10:00:00Z")
    assert hourly.iloc[0].open==10
    assert hourly.iloc[0].close==13


def test_api_license_binding_admin_and_immediate_poll_revoke(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(brain,"DB_PATH",tmp_path/"accounts.db")
    monkeypatch.setenv("ADMIN_PIN","admin-test-pin")
    brain.init_db()
    client=TestClient(brain.app)
    assert client.get("/manifest.webmanifest").status_code==200
    assert client.get("/sw.js").status_code==200
    assert client.get("/icon.svg").status_code==200
    admin={"X-Admin-PIN":"admin-test-pin"}
    assert client.get("/api/admin/licenses").status_code==401
    created=client.post("/api/admin/licenses/create",json={"owner_name":"Demo test"},headers=admin)
    assert created.status_code==200
    key=created.json()["license_key"]
    activation={"license_key":key,"account_id":"123","device_id":"device-uuid-001"}
    assert client.post("/api/activate",json=activation).status_code==200
    assert client.post("/api/activate",json={**activation,"device_id":"device-uuid-002"}).status_code==403
    assert client.post("/api/activate",json={**activation,"account_id":"456"}).status_code==403
    headers={"X-License-Key":key,"X-License-Account-ID":"123","X-Device-ID":"device-uuid-001"}
    data={"account_id":"123","broker_server":"Demo","account_type":"broker","risk_pct":1.0,"max_daily_dd_pct":4.0,"max_total_dd_pct":10,"active":True}
    assert client.post("/api/accounts",json=data,headers=headers).status_code==200
    assert client.get("/api/accounts/123/status",headers={**headers,"X-Device-ID":"device-uuid-002"}).status_code==403
    license_row=client.get("/api/admin/licenses",headers=admin).json()[0]
    assert license_row["bound_account_id"]=="123" and license_row["bound_device_id"]=="device-uuid-001"
    monkeypatch.setattr(brain,"analyze_symbol",lambda *args,**kwargs:None)
    poll=client.get("/api/ea/poll/123/EURUSD?balance=10000&equity=9900",headers={"X-License-Key":key})
    assert poll.status_code==200 and poll.json()["decision"]=="WAIT"
    assert poll.json()["max_daily_dd_pct"]==4
    assert client.get("/api/accounts/123/status",headers=headers).json()["account"]["active"]==1
    assert client.post(f"/api/admin/licenses/{key}/toggle",json={"is_active":False},headers=admin).status_code==200
    assert client.get("/api/ea/poll/123/EURUSD",headers={"X-License-Key":key}).json()=={"command":"UNAUTHORIZED"}


def test_quant_and_bias_series_latest_match_single_bar_functions():
    n=300;x=np.arange(n);close=100+x*.01+np.sin(x/3)
    d=pd.DataFrame({"open":close,"high":close+.3,"low":close-.3,"close":close})
    series=brain.quant_metrics_series(d).iloc[-1]
    latest=brain.quant_metrics(d)
    assert latest["atr_ratio"]==pytest.approx(series.atr_ratio)
    assert latest["zscore"]==pytest.approx(series.zscore)
    assert brain.market_bias_series(d).iloc[-1]==brain.market_bias(d)


def test_backtest_reports_empty_window_without_fake_trades():
    times=pd.date_range("2024-01-01",periods=500,freq="15min",tz="UTC")
    close=100+np.sin(np.arange(500)/4)*.1
    df=pd.DataFrame({"time":times,"open":close,"high":close+.05,"low":close-.05,"close":close})
    out=run_backtest(df)
    assert out["trade_count"]>=0
    assert out["total_return_pct"]==pytest.approx((out["ending_balance"]/10000-1)*100)
    assert isinstance(out["monthly_trade_frequency"],dict)


def test_backtest_exit_prices_include_adverse_slippage_and_spread():
    future=pd.DataFrame({"low":[99.5],"high":[100.5],"close":[100.],"spread_price":[.0002]})
    buy,_,_=_walk_exit(future,"BUY",100,99,102,None,.0001)
    sell,_,_=_walk_exit(future,"SELL",100,101,98,None,.0001)
    assert buy==pytest.approx(99.9999)
    assert sell==pytest.approx(100.0003)
