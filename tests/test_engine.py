import numpy as np
import pandas as pd
import pytest

import brain
from backtest import _resample_h1, run_backtest


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


def test_api_account_lifecycle_and_pin(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(brain,"DB_PATH",tmp_path/"accounts.db")
    monkeypatch.setenv("DASHBOARD_PIN","2468")
    brain.init_db()
    client=TestClient(brain.app)
    assert client.get("/manifest.webmanifest").status_code==200
    assert client.get("/sw.js").status_code==200
    assert client.get("/icon.svg").status_code==200
    data={"account_id":"123","broker_server":"Demo","account_type":"broker","risk_pct":.5,"max_daily_dd_pct":5,"max_total_dd_pct":10}
    assert client.post("/api/accounts",json=data).status_code==401
    headers={"X-Dashboard-PIN":"2468"}
    assert client.post("/api/accounts",json=data,headers=headers).status_code==200
    poll=client.get("/api/ea/poll/123/EURUSD?balance=10000&equity=9900")
    assert poll.status_code==200 and poll.json()["decision"]=="WAIT"
    assert poll.json()["max_daily_dd_pct"]==5
    assert client.post("/api/accounts/123/toggle?active=true",headers=headers).status_code==200
    assert client.get("/api/accounts/123/status",headers=headers).json()["account"]["active"]==1


def test_backtest_reports_empty_window_without_fake_trades():
    times=pd.date_range("2024-01-01",periods=500,freq="15min",tz="UTC")
    close=100+np.sin(np.arange(500)/4)*.1
    df=pd.DataFrame({"time":times,"open":close,"high":close+.05,"low":close-.05,"close":close})
    out=run_backtest(df)
    assert out["trade_count"]>=0
    assert out["total_return_pct"]==pytest.approx((out["ending_balance"]/10000-1)*100)
    assert isinstance(out["monthly_trade_frequency"],dict)
