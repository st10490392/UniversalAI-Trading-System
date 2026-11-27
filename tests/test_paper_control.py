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
