import json
from datetime import datetime,timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import paper_control as core


@pytest.fixture
def clients(tmp_path,monkeypatch):
    clock=datetime(2026,11,25,14,tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(core,"time",SimpleNamespace(time=lambda:clock))
    monkeypatch.setenv("PAPER_STATE_DB",str(tmp_path/"state.db"))
    registry={}
    for account in ("a","b"):
        registry[account]=dict(profile="prop_firm_core",starting_balance=10000,
                               control_token_env=account+"c",feed_token_env=account+"f")
        for role in "cf": monkeypatch.setenv(account+role,account+role)
    monkeypatch.setenv("PAPER_ACCOUNTS_JSON",json.dumps(registry))
    return TestClient(core.app),clock,registry


def token(value):
    return {"Authorization":"Bearer "+value}


def test_floating_breach_is_atomic_and_account_scoped(clients):
    client,now,registry=clients
    base="/accounts/a"
    mark=dict(symbol="XAUUSD",timestamp=now,bid=100.,ask=100.,cash_per_price_unit=1.,
              volume_min=.01,volume_step=.01,volume_max=100.)
    intent=dict(id="1",symbol="XAUUSD",strategy="Session ORB",side="BUY",stop=99.,target=103.)
    with core.state("a",registry["a"]) as s:
        s["bias"]["XAUUSD"]={"direction":"BULLISH","as_of":now}
    assert client.get("/accounts/b/status",headers=token("ac")).status_code==401
    assert client.post(base+"/quotes",headers=token("ac"),json=mark).status_code==401
    assert client.post(base+"/quotes",headers=token("af"),json=mark).is_success
    def enter(): return client.post(base+"/paper-intents",headers=token("af"),json=intent).json()
    assert enter()["command"]=="WAIT"
    cal=dict(start=now-3600,end=now+3600,events=[])
    client.post(base+"/calendar",headers=token("af"),json=cal)
    assert enter()["paper_mode"]
    mark.update(bid=94.,ask=94.)
    assert client.post(base+"/quotes",headers=token("af"),json=mark).json()["locked"]=="DAILY_LOSS"
    snapshot=client.get(base+"/status",headers=token("ac")).json()
    assert snapshot["positions"]==[] and snapshot["balance"]==9700
    assert len(client.get(base+"/history",headers=token("ac")).json())==1
    assert client.get("/accounts/b/status",headers=token("bc")).json()["balance"]==10000
    assert client.patch(base+"/settings",headers=token("ac"),json={"max_dd_cap_pct":2}).is_success
    assert client.get(base+"/status",headers=token("ac")).json()["locked"]=="MAX_DRAWDOWN"
    assert client.post("/accounts/b/commands/FLATTEN_ALL_AND_PAUSE",headers=token("bc")).json()["locked"]=="EMERGENCY"


def test_calendar_boundaries_and_hot_reload(clients):
    client,now,_=clients
    cal=dict(start=now-10000,end=now+10000,events=[dict(timestamp=now,currency="EUR",impact="High")])
    for offset in (-900,900): assert core.news_reason(cal,now+offset)
    assert core.news_reason(cal,now+901) is None
    patch=dict(risk_pct=.25,setups=[dict(symbol="Germany 40",strategy="EMA VWAP",enabled=False)])
    assert client.patch("/accounts/a/settings",headers=token("ac"),json=patch).is_success
    cfg=client.get("/accounts/a/status",headers=token("ac")).json()["config"]
    assert cfg["risk_pct"]==.25 and not cfg["setups"][2]["enabled"]
    assert client.get("/market-bias?account=a",headers=token("ac")).json()["XAUUSD"]["stale"]

def test_partial_fill_preserves_equity_and_does_not_repeat(clients):
    client,now,registry=clients
    q=dict(symbol="XAUUSD",timestamp=now,bid=101.,ask=101.,cash_per_price_unit=1.,
           volume_min=.01,volume_step=.01,volume_max=100.)
    with core.state("a",registry["a"]) as s:
        s["quotes"]["XAUUSD"]={**q,"bid":100.,"ask":100.}
        s["positions"].append(dict(id="p",symbol="XAUUSD",strategy="EMA VWAP",side="BUY",
                                   entry=100.,stop=99.,target=101.,volume=2.,opened_at=now,exit_rule="partialtrail"))
    for _ in range(2):
        response=client.post("/accounts/a/quotes",headers=token("af"),json=q)
        assert response.json()["equity"]==10002
    with core.state("a",registry["a"]) as s:
        assert s["balance"]==10001 and s["positions"][0]["volume"]==1
        assert [v["reason"] for v in s["history"]]==["PARTIAL_1R"]


def test_offline_fallback_preserves_known_news():
    s={"config":{"news":{"offline_fallback":"allow_paper"}},"calendar":None}
    assert core.calendar_pause(s,1000) is None
    s["calendar"]={"start":950,"end":1050,"events":[{"timestamp":1010,"currency":"USD","impact":"HIGH"}]}
    assert core.calendar_pause(s,1000)

def test_closed_bar_orb_drives_signal_without_future_bars():
    import pandas as pd
    from paper_signals import decisions

    now=int(pd.Timestamp("2026-11-25 07:20",tz="UTC").timestamp())
    def frame(seconds,count):
        return pd.DataFrame({"time":range(now//seconds*seconds-count*seconds,now//seconds*seconds,seconds),
                             "open":[100.]*count,"close":[100.]*count,"low":[99.]*count,
                             "high":[101.]*count,"tick_volume":[10]*count})
    frames={"H1":frame(3600,220),"M5":frame(300,200),"M15":frame(900,200)}
    frames["M5"].loc[199,["high","close"]]=[103.,102.]
    setup={"symbol":"XAUUSD","strategy":"Session ORB","enabled":True,"profile":"1.5R/15m"}
    result,_=decisions(frames,"XAUUSD",[setup],now)
    assert result==[(setup,1,100.,now)]
    frames["M5"].loc[200]=[now,100.,100.,80.,150.,10]
    assert decisions(frames,"XAUUSD",[setup],now)[0]==result
