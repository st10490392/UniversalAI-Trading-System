import time
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
import paper_control as core
from mt5_live_adapter import Adapter,resolve_symbol


@pytest.fixture
def api(tmp_path,monkeypatch):
    monkeypatch.setenv("PAPER_STATE_DB",str(tmp_path/"control.db"))
    monkeypatch.setenv("OWNER_MASTER_KEY","owner")
    monkeypatch.setenv("PAPER_ACCOUNTS_JSON",'{"desk":{"profile":"prop_firm_core","starting_balance":10000,"feed_token_env":"FEED"}}')
    monkeypatch.setenv("FEED","feed")
    return TestClient(core.app)


def headers(key,device="phone"):
    return {"Authorization":"Bearer "+key,"X-Device-ID":device}


def issue(api,scope):
    reply=api.post("/admin/licenses",headers=headers("owner"),json={"account":"desk","scope":scope})
    assert reply.status_code==200
    return reply.json()


def test_signal_license_and_queue(api):
    reg=issue(api,"SIGNALS_ONLY"); h=headers(reg["license_key"])
    assert api.get("/market-bias?account=desk",headers=h).status_code==403
    assert api.post("/licenses/activate",headers=h,json={"device_id":"phone"}).is_success
    assert api.post("/licenses/activate",headers=h,json={"device_id":"other"}).status_code==403
    assert api.get("/market-bias?account=desk",headers=h).is_success
    assert api.get("/accounts/desk/alerts",headers=h).is_success
    assert api.get("/accounts/desk/status",headers=h).status_code==403
    assert api.post("/notifications/register-device",headers=h,json={"token":"mock-device"}).is_success
    with core.state("desk",{"profile":"prop_firm_core","starting_balance":10000}) as s:
        s.setdefault("alerts",[]).append({"id":"setup","symbol":"XAUUSD","side":"BUY"})
    assert api.get("/notifications/events",headers=h).json()[0]["alert"]["id"]=="setup"
    with core.licenses.store() as db:
        assert db.execute("SELECT count(*) FROM push_delivery").fetchone()[0]==1
    assert api.post("/admin/licenses/"+reg["id"]+"/revoke",headers=headers("owner")).is_success
    assert api.get("/notifications/events",headers=h).status_code==401


def test_ea_registration_and_owner_reset(api):
    reg=issue(api,"EA_ONLY");h=headers(reg["license_key"])
    binding={"device_id":"phone","mt5_account_id":"123","mt5_server":"Deriv-Demo"}
    assert api.post("/licenses/activate",headers=h,json=binding).is_success
    assert api.get("/accounts/desk/alerts",headers=h).status_code==403
    assert api.get("/market-bias?account=desk",headers=h).status_code==403
    data=dict(mt5_account_id="123",mt5_server="Deriv-Demo",risk_mode="percent",risk_value=.25,
              daily_loss_cap=2,max_dd_cap=5,demo_live_execution=True)
    assert api.post("/ea/register-and-start",headers=h,json={**data,"mt5_account_id":"321"}).status_code==403
    assert api.post("/ea/register-and-start",headers=h,json=data).is_success
    url="/accounts/desk/execution-plan"
    assert api.get(url,headers=headers("feed")).json()["enabled"]
    assert api.post("/admin/licenses/"+reg["id"]+"/reset",headers=headers("owner")).is_success
    assert not api.get(url,headers=headers("feed")).json()["enabled"]
    assert api.get("/accounts/desk/status",headers=h).status_code==403


def test_mt5_real_gate_symbol_mapping_and_idempotent_entry(tmp_path,monkeypatch):
    monkeypatch.delenv("ALLOW_REAL_ACCOUNTS",raising=False)
    m=Mock(ACCOUNT_TRADE_MODE_DEMO=0,ORDER_TYPE_BUY=0,ORDER_TYPE_SELL=1,
           TRADE_ACTION_DEAL=1,ORDER_FILLING_IOC=1,ORDER_FILLING_FOK=0,
           TRADE_RETCODE_DONE=10009,TRADE_RETCODE_DONE_PARTIAL=10010)
    info=NS(name="Gold.m",trade_tick_size=.01,volume_min=.01,volume_step=.01,volume_max=10.,filling_mode=2)
    m.symbol_info.side_effect=lambda name:info if name=="Gold.m" else None
    m.symbols_get.return_value=[info]
    assert resolve_symbol(m,"XAUUSD")=="Gold.m"
    a=NS(server="Deriv-Demo",login=123,trade_mode=2,equity=10000.,balance=10000.)
    m.account_info.return_value=a
    adapter=Adapter(m,"123","Deriv-Demo",tmp_path/"journal")
    with pytest.raises(ValueError): adapter.account()
    m.order_send.assert_not_called()
    a.trade_mode=0
    m.positions_get.return_value=[]
    m.symbol_info_tick.return_value=NS(time=int(time.time()),ask=100.,bid=99.99)
    m.order_calc_profit.return_value=-100.
    m.order_check.return_value=NS(retcode=0)
    m.order_send.return_value=NS(retcode=10009)
    plan=dict(enabled=True,risk=dict(max_dd_cap_pct=5,daily_loss_cap_pct=2,risk_pct=.5),
              positions=[dict(id="abc",opened_at=time.time(),symbol="XAUUSD",side="BUY",stop=99.,target=103.,volume=1.)])
    adapter.sync(plan);adapter.sync(plan)
    req=m.order_send.call_args.args[0]
    assert req["volume"]==.5 and req["sl"]==99 and req["tp"]==103
    assert m.order_send.call_count==1
