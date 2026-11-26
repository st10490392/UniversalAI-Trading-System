import numpy as np
import pandas as pd
import pytest

import brain
from backtest import _resample_h1, _smc_side_at, _walk_exit, run_backtest
from tools.export_mt5_history import classify_symbol
import research_lab


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
    assert _smc_side_at(d.high.to_numpy(),d.low.to_numpy(),d.close.to_numpy(),25,"BULLISH")==setup["side"]


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
    assert brain.is_synthetic("Range Break 200 Index")


@pytest.mark.parametrize("symbol,family",[
    ("Volatility 150 (1s) Index","Volatility"),
    ("Boom 900 Index","Boom/Crash"),
    ("Skew Step Index 5 Up","Step"),
    ("Jump 100 Index","Jump"),
    ("Range Break 200 Index","Range Break"),
    ("EURUSD.a","Major FX"),
    ("XAUUSD.a","Metals"),
    ("US Tech 100","NASDAQ"),
    ("Germany 40","GER30"),
    ("Wall Street 30","US30"),
])
def test_market_watch_family_matching(symbol,family):
    assert family in classify_symbol(symbol)


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
    poll=client.get("/api/ea/poll/123/EURUSD",headers={"X-License-Key":key,"X-Account-Balance":"10000","X-Account-Equity":"9900"})
    assert poll.status_code==200 and poll.json()["decision"]=="WAIT"
    assert poll.json()["max_daily_dd_pct"]==4
    assert client.get("/api/accounts/123/status",headers=headers).json()["account"]["active"]==1
    state=client.get("/api/accounts/123/status",headers=headers).json()["state"]
    assert state["balance"]==10000 and state["equity"]==9900
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


def test_backtest_fills_next_open_and_uses_exported_point_size(monkeypatch):
    from types import SimpleNamespace
    import backtest as bt
    n=420;times=pd.date_range("2025-01-01T00:00:00Z",periods=n,freq="15min")
    close=np.full(n,100.);op=close.copy();high=close+.2;low=close-.2
    close[300]=99.5;op[301]=100.;high[301]=100.5;low[301]=100.
    high[302]=104.;low[302]=100.
    df=pd.DataFrame({"time":times,"open":op,"high":high,"low":low,"close":close,"point_size":.01,"spread_points":1.,"tick_size":.01,"tick_value":1.})
    quant=SimpleNamespace(ready=pd.Series(np.ones(n,dtype=bool)),gate=pd.Series(np.ones(n,dtype=bool)),zscore=pd.Series(np.zeros(n)),atr14=pd.Series(np.ones(n)))
    monkeypatch.setattr(bt,"quant_metrics_series",lambda _d:quant)
    monkeypatch.setattr(bt,"market_bias_series",lambda _d:pd.Series(["BULLISH"]*len(_d)))
    monkeypatch.setattr(bt,"_smc_side_at",lambda h,l,c,i,bias:"BUY" if i==300 else None)
    out=bt.run_backtest(df,symbol="Volatility 75 Index",slippage_points=2,spread_points=20,point=.1,commission_per_lot=0,commission_points=0)
    assert out["trade_count"]==1
    trade=out["trades"][0]
    assert trade["entry_time"]==times[301]
    assert trade["entry"]==pytest.approx(100.03)
    assert trade["outcome"]=="win"


def test_causal_spreads_backtester_does_not_fill_zeros_from_future():
    from backtest import _causal_spread_points
    assert _causal_spread_points(np.array([0.,4.,0.,8.,0.]),20).tolist()==[20.,4.,4.,8.,6.]


@pytest.mark.parametrize("symbol,asset",[
    ("EURUSD.a","forex"),("XAUUSD","forex"),("USTEC.cash","index"),
    ("Volatility 75 Index","synthetic"),("Range Break 200 Index","synthetic"),
])
def test_research_asset_class_taxonomy(symbol,asset):
    assert research_lab._asset(symbol)==asset


def test_strategy_c_is_explicitly_mean_reversion_and_causal_indicators():
    n=400;times=pd.date_range("2020-01-01",periods=n,freq="15min",tz="UTC")
    close=100+np.sin(np.arange(n)/5)*.15
    d=pd.DataFrame({"time":times,"open":close,"high":close+.05,"low":close-.05,"close":close,"tick_volume":100})
    f=research_lab.add_indicators(d)
    assert {"atr_ratio","zscore","er20","squeeze","vwap","body_ratio","upper_wick_ratio","lower_wick_ratio"}.issubset(f.columns)
    side,_,_,codes=research_lab.build_signals(f,"Volatility 75 Index","C")
    assert len(side)==n and set(np.unique(side)).issubset({-1,0,1})
    assert all("mean_reversion" in code for code in codes[side!=0])


def test_synthetic_smc_signal_has_no_forex_session_gate():
    times=pd.date_range("2024-01-01T00:00:00Z",periods=130,freq="15min")
    close=100+np.arange(130)*.05
    d=pd.DataFrame({"time":times,"open":close,"high":close+.02,"low":close-.02,"close":close,"tick_volume":100,"asset_class":"synthetic"})
    prior_low=d.low.iloc[80:100].min()
    d.loc[100,"low"]=prior_low-.5;d.loc[100,"close"]=prior_low+.02;d.loc[100,"open"]=prior_low-.1;d.loc[100,"high"]=prior_low+.05
    d.loc[102,"low"]=d.high.iloc[100]+.1;d.loc[102,"close"]=d.low.iloc[102]+.05;d.loc[102,"open"]=d.close.iloc[102]-.02;d.loc[102,"high"]=d.close.iloc[102]+.1
    f=research_lab.add_indicators(d)
    side,_,_,codes=research_lab.build_signals(f,"Volatility 75 Index","B")
    assert side[102]==1
    assert codes[102]=="synthetic_sweep_fvg_directional_bias"


def test_unapproved_symbol_is_never_sent_to_mt5_or_groq(monkeypatch,tmp_path):
    monkeypatch.setattr(brain,"APPROVED_PATH",tmp_path/"approved_strategies.json")
    def forbidden(*args,**kwargs):raise AssertionError("unapproved symbol must fail closed before live scan")
    monkeypatch.setattr(brain,"_mt5_rates",forbidden)
    monkeypatch.setattr(brain,"groq_decision",forbidden)
    assert brain.analyze_symbol("EURUSD") is None


def test_approved_live_candidate_passes_exact_setup_and_quant_to_qwen(monkeypatch,tmp_path):
    path=tmp_path/"approved.json"
    path.write_text(__import__("json").dumps({"approved":[{"symbol":"EURUSD","asset_class":"forex","strategy":"Trend Momentum"}]}),encoding="utf-8")
    monkeypatch.setattr(brain,"APPROVED_PATH",path)
    times=pd.date_range("2025-01-01",periods=80,freq="15min",tz="UTC")
    close=1.1+np.arange(80)*.0001
    rates=pd.DataFrame({"time":(times.astype("int64")//10**9).astype(int),"open":close,"high":close+.0001,"low":close-.0001,"close":close,"tick_volume":100})
    monkeypatch.setattr(brain,"_mt5_rates",lambda *a,**k:rates.copy())
    setup={"rule":"ema_cross_rsi_atr_expansion","entry_reference":1.2,"stop":1.19,"target":1.225,"risk_reward":2.5}
    quant={"atr14":.01,"atr_ratio":1.2,"zscore":.4,"er20":.4,"rsi":60}
    monkeypatch.setattr(brain,"live_candidates",lambda symbol,data,names,model_dir:[{"strategy":"Trend Momentum","side":"BUY","setup":setup,"quant":quant}])
    seen={}
    def qwen(context):seen.update(context);return {"decision":"BUY","entry":1.2,"sl":1.19,"tp":1.225,"confidence":.9,"reason":"sanity check passed"}
    monkeypatch.setattr(brain,"groq_decision",qwen)
    out=brain.analyze_symbol("EURUSD")
    assert out["decision"]=="BUY" and out["strategy"]=="Trend Momentum" and out["breakeven_at_r"]==1.2
    assert seen["setup"]==setup and seen["quant_metrics"]==quant


def test_approved_strategies_api_requires_license_and_returns_pairs(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(brain,"DB_PATH",tmp_path/"accounts.db")
    monkeypatch.setattr(brain,"APPROVED_PATH",tmp_path/"approved.json")
    monkeypatch.setenv("ADMIN_PIN","admin-test-pin")
    brain.init_db();client=TestClient(brain.app);admin={"X-Admin-PIN":"admin-test-pin"}
    key=client.post("/api/admin/licenses/create",json={"owner_name":"Dashboard test"},headers=admin).json()["license_key"]
    activation={"license_key":key,"account_id":"321","device_id":"device-uuid-321"}
    assert client.post("/api/activate",json=activation).status_code==200
    headers={"X-License-Key":key,"X-License-Account-ID":"321","X-Device-ID":"device-uuid-321"}
    assert client.get("/api/strategies").status_code==403
    approved={"generated_at":"2025-01-01T00:00:00Z","approved":[{"symbol":"EURUSD","asset_class":"forex","strategy":research_lab.STRATEGIES["A"],"oos":{"profit_factor":1.5,"trades":90},"monte_carlo":{"probability_of_ruin_pct":2.0}}]}
    brain.APPROVED_PATH.write_text(__import__("json").dumps(approved),encoding="utf-8")
    response=client.get("/api/strategies",headers=headers)
    assert response.status_code==200 and response.json()["approved"][0]["strategy"]=="Trend Momentum"


def test_monte_carlo_uses_requested_risk_and_repeatable_seed():
    trades=[{"time":f"2024-01-{i+1:02d}T00:00:00+00:00","r":.2} for i in range(20)]
    a=research_lab._monte_carlo(trades,iterations=100,risk_pct=.75,seed=4)
    b=research_lab._monte_carlo(trades,iterations=100,risk_pct=.75,seed=4)
    assert a==b and a["risk_pct"]==.75 and a["probability_of_ruin_pct"]==0


def _dated_r_trades(values, *, same_day=False):
    base=pd.Timestamp("2025-01-01T00:00:00Z")
    return [{"time":(base+pd.Timedelta(days=(0 if same_day else i),minutes=(i if same_day else 0))).isoformat(),"r":float(r)} for i,r in enumerate(values)]


def test_monte_carlo_separates_prop_limits_from_equity_zero():
    daily=research_lab._monte_carlo(_dated_r_trades([-1]*7,same_day=True),iterations=20,risk_pct=.75,seed=11)
    assert daily["probability_daily_limit_breach_pct"]==100
    assert daily["probability_total_limit_breach_pct"]==0
    assert daily["probability_either_limit_breach_pct"]==100
    assert daily["probability_equity_zero_pct"]==0
    assert daily["p95_max_drawdown_pct"]==pytest.approx(5.137,abs=.01)

    total=research_lab._monte_carlo(_dated_r_trades([-1]*15),iterations=20,risk_pct=.75,seed=11)
    assert total["probability_daily_limit_breach_pct"]==0
    assert total["probability_total_limit_breach_pct"]==100
    assert total["probability_either_limit_breach_pct"]==100
    assert total["probability_equity_zero_pct"]==0
    assert total["p95_max_drawdown_pct"]==pytest.approx(10.003,abs=.02)


def test_monte_carlo_account_zero_is_not_the_prop_ruin_definition():
    out=research_lab._monte_carlo(_dated_r_trades([-200]),iterations=1,risk_pct=.75,seed=3)
    assert out["probability_daily_limit_breach_pct"]==100
    assert out["probability_total_limit_breach_pct"]==100
    assert out["probability_equity_zero_pct"]==100
    assert out["probability_of_ruin_pct"]==100
    positive=research_lab._monte_carlo(_dated_r_trades([1]),iterations=1,risk_pct=.75,seed=3)
    assert positive["probability_of_ruin_pct"]==0 and positive["probability_equity_zero_pct"]==0


def test_causal_spread_fallback_uses_past_only():
    d=pd.DataFrame({"spread_points":[0,4,0,8,0]})
    assert research_lab._causal_spreads(d,20).tolist()==[20,4,4,8,6]


def test_research_short_timeout_exits_at_ask():
    d=pd.DataFrame({"time":pd.date_range("2025-01-01",periods=3,freq="15min",tz="UTC"),"open":[100.,100.,100.],"high":[100.5,100.5,100.5],"low":[99.5,99.5,99.5],"close":[100.,100.,100.],"point_size":[1.,1.,1.],"spread_points":[5.,5.,5.],"tick_size":[0.,0.,0.],"tick_value":[0.,0.,0.]})
    trade=research_lab._trade_one(d,0,-1,2.,np.nan,breakeven=False,commission_per_lot=0.,slippage_points=0.,commission_points=0.,end_idx=3,horizon=1)
    assert trade["exit"]==105.
    assert trade["r"]==pytest.approx(-2.5)


def test_research_point_commission_fallback_is_round_turn_once():
    d=pd.DataFrame({"time":pd.date_range("2025-01-01",periods=3,freq="15min",tz="UTC"),"open":[100.,100.,100.],"high":[100.5,100.5,100.5],"low":[99.5,99.5,99.5],"close":[100.,100.,100.],"point_size":[1.,1.,1.],"spread_points":[0.,0.,0.],"tick_size":[0.,0.,0.],"tick_value":[0.,0.,0.]})
    trade=research_lab._trade_one(d,0,1,10.,np.nan,breakeven=False,commission_per_lot=0.,slippage_points=0.,commission_points=2.,end_idx=3,horizon=1)
    assert trade["r"]==pytest.approx(-.2)
