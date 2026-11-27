"""Account-scoped, SQLite-serialized paper execution. All prices are bid/ask."""
import threading
from contextlib import asynccontextmanager
import account_licenses as licenses
import json
import math
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parent
load_dotenv(ROOT/".env",override=False)
@asynccontextmanager
async def lifespan(app):
    with licenses.store(): pass
    stop=threading.Event()
    project=os.getenv("FCM_PROJECT_ID")
    def dispatch():
        while not stop.is_set():
            try:
                if licenses.deliver_one(project): continue
            except Exception:
                import logging
                logging.getLogger(__name__).warning("FCM delivery pending; check credentials and project")
            stop.wait(3)
    worker=threading.Thread(target=dispatch,daemon=True) if project else None
    if worker: worker.start()
    try: yield
    finally:
        stop.set()
        if worker: worker.join(timeout=12)

app=FastAPI(title="UniversalAI account control",lifespan=lifespan)
app.include_router(licenses.router)

@app.get("/health")
def health():
    with licenses.store() as db: db.execute("SELECT 1")
    return {"status":"ok"}



class Payload(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)


class Quote(Payload):
    symbol:str
    timestamp:float
    bid:float=Field(gt=0)
    ask:float=Field(gt=0)
    cash_per_price_unit:float=Field(gt=0)
    volume_min:float=Field(gt=0)
    volume_step:float=Field(gt=0)
    volume_max:float=Field(gt=0)

    @model_validator(mode="after")
    def valid(self):
        if self.ask<self.bid or self.volume_max<self.volume_min:
            raise ValueError("invalid broker quote or volume grid")
        return self


class Intent(Payload):
    id:str=Field(min_length=1,max_length=160)
    symbol:str
    strategy:str
    side:Literal["BUY","SELL"]
    stop:float=Field(gt=0)
    target:float=Field(gt=0)


class Toggle(Payload):
    symbol:str
    strategy:str
    enabled:bool


class Settings(Payload):
    ai_gate_enforce:bool|None=None
    risk_pct:float|None=Field(default=None,gt=0,le=.75)
    daily_loss_cap_pct:float|None=Field(default=None,gt=0,le=2.5)
    max_dd_cap_pct:float|None=Field(default=None,gt=0,le=7.5)
    risk_amount_usd:float|None=Field(default=None,gt=0)
    daily_loss_cap_usd:float|None=Field(default=None,gt=0)
    setups:list[Toggle]=Field(default_factory=list)


class Event(Payload):
    timestamp:float
    currency:str
    impact:str


class Calendar(Payload):
    start:float
    end:float
    events:list[Event]


def news_reason(calendar,now):
    if not calendar or calendar["start"]>now-900 or calendar["end"]<now+900:
        return "Calendar coverage unavailable"
    for event in calendar["events"]:
        if (event["currency"].upper() in {"USD","EUR"} and event["impact"].lower()=="high"
                and abs(event["timestamp"]-now)<=900):
            return "High-impact "+event["currency"]+" news window"
    return None


def calendar_pause(s,now):
    policy=s["config"].get("news",{})
    if not policy.get("enabled",True): return None
    reason=news_reason(s["calendar"],now)
    if reason=="Calendar coverage unavailable" and policy.get("offline_fallback")=="allow_paper":
        # Even an incomplete cache may contain a known event in the guard interval.
        events=(s["calendar"] or {}).get("events",[])
        return news_reason(dict(start=now-901,end=now+901,events=events),now)
    return reason


def authorize(role,scope="EA_ONLY"):
    def check(account:str,authorization:str=Header(default=""),x_device_id:str=Header(default="")):
        registry=json.loads(os.getenv("PAPER_ACCOUNTS_JSON","{}"))
        cfg=registry.get(account,{})
        if role=="control" and authorization.startswith("Bearer uai_"):
            licenses.licensed(authorization,x_device_id,scope,account)
            if not cfg: raise HTTPException(403,"Unknown engine account")
            return cfg
        token=os.getenv(cfg.get(role+"_token_env",""),"")
        tokens=[os.getenv(v.get(k,""),"") for v in registry.values() for k in ("control_token_env","feed_token_env")]
        if not token or tokens.count(token)!=1 or not secrets.compare_digest(authorization.encode(),("Bearer "+token).encode()):
            raise HTTPException(401,"Invalid account token")
        return cfg
    return check


control=authorize("control")
feed=authorize("feed")
signals=authorize("control","SIGNALS_ONLY")


@contextmanager
def state(account,cfg,*,include_db=False):
    from control_storage import transaction
    with transaction("engine:"+account) as db:
        row=db.execute("SELECT state FROM engines WHERE account=?",(account,)).fetchone()
        if row:
            s=json.loads(row[0])
            if s["profile"]!=cfg["profile"]: raise HTTPException(409,"Account profile binding changed")
        else:
            profile=json.loads(Path(cfg.get("config",ROOT/"three_setup_portfolio.json")).read_text())
            if profile.get("paper_mode") is not True or profile.get("profile")!=cfg["profile"]:
                raise HTTPException(409,"Invalid paper profile")
            balance=float(cfg["starting_balance"])
            if not math.isfinite(balance) or balance<=0: raise HTTPException(409,"Invalid paper capital")
            s=dict(profile=cfg["profile"],config=profile,balance=balance,peak=balance,day_open=balance,
                   day="",locked=None,positions=[],history=[],quotes={},bias={},calendar=None,used={})
        count=len(s.get("alerts",[]))
        yield (s,db) if include_db else s
        for alert in s.get("alerts",[])[count:]: licenses.enqueue(db,account,alert)
        db.execute("INSERT INTO engines VALUES(?,?) ON CONFLICT(account) DO UPDATE SET state=excluded.state",(account,json.dumps(s,allow_nan=False)))


def pnl(s,p):
    q=s["quotes"][p["symbol"]]
    price=q["bid"] if p["side"]=="BUY" else q["ask"]
    direction=1 if p["side"]=="BUY" else -1
    return direction*(price-p["entry"])*q["cash_per_price_unit"]*p["volume"]


def equity(s):
    return s["balance"]+sum(pnl(s,p) for p in s["positions"])


def close(s,p,reason,now):
    gain=pnl(s,p)
    s["balance"]+=gain
    s["history"].append(dict(**p,closed_at=now,net_pnl=gain,reason=reason,mark_time=s["quotes"][p["symbol"]]["timestamp"]))
    s["positions"].remove(p)


def flatten(s,reason,now):
    s["locked"]=reason
    for p in s["positions"][:]: close(s,p,reason,now)


def risk(s,now):
    value=equity(s)
    s["peak"]=max(s["peak"],value)
    daily=max(0.,100*(1-value/s["day_open"]))
    dd=max(0.,100*(1-value/s["peak"]))
    if dd>=s["config"]["max_dd_cap_pct"]-1e-10:
        flatten(s,"MAX_DRAWDOWN",now)
    elif (s["day_open"]-value>=s["config"]["daily_loss_cap_usd"] if "daily_loss_cap_usd" in s["config"] else daily>=s["config"]["daily_loss_cap_pct"]-1e-10) and not s["locked"]:
        flatten(s,"DAILY_LOSS",now)
    return daily,dd


def rollover(s,now):
    day=datetime.fromtimestamp(now,timezone.utc).date().isoformat()
    if day!=s["day"]:
        s.update(day=day,day_open=equity(s),used={})
        if s["locked"]=="DAILY_LOSS": s["locked"]=None


@app.post("/accounts/{account}/quotes")
def quote(account:str,q:Quote,cfg=Depends(feed)):
    now=time.time()
    if not 0<=now-q.timestamp<=30: raise HTTPException(422,"Stale or future quote")
    with state(account,cfg) as s:
        tracked={x["symbol"] for x in s["config"]["setups"]}
        if q.symbol not in tracked: raise HTTPException(422,"Untracked symbol")
        if q.timestamp<s["quotes"].get(q.symbol,{}).get("timestamp",0): raise HTTPException(409,"Older quote")
        rollover(s,now)
        s["quotes"][q.symbol]=q.model_dump()
        risk(s,now)
        for p in s["positions"][:]:
            if p["symbol"]!=q.symbol: continue
            window=next(v["entry_utc"] for v in s["config"]["setups"] if
                        (v["symbol"],v["strategy"])==(p["symbol"],p["strategy"]))
            clock=datetime.fromtimestamp(now,timezone.utc)
            opened=datetime.fromtimestamp(p["opened_at"],timezone.utc)
            if not s["config"].get("always_open") and (clock.date()!=opened.date() or clock.strftime("%H:%M")>=window[1] or now>=p.get("session_end",float("inf"))):
                close(s,p,"SESSION_END",now)
                continue
            buy=p["side"]=="BUY"
            if (q.bid<=p["stop"] if buy else q.ask>=p["stop"]): close(s,p,"STOP",now)
            elif p.get("exit_rule")=="partialtrail" and not p.get("partial") and (q.bid>=p["target"] if buy else q.ask<=p["target"]):
                p["volume"]/=2
                gain=pnl(s,p)
                s["balance"]+=gain
                s["history"].append(dict(**p,closed_at=now,net_pnl=gain,reason="PARTIAL_1R",mark_time=q.timestamp))
                p["partial"]=True
                p["partial_at"]=now
            elif p.get("exit_rule") not in ("partialtrail","be") and (q.bid>=p["target"] if buy else q.ask<=p["target"]):
                close(s,p,"TARGET",now)
        return {"equity":equity(s),"locked":s["locked"]}


@app.post("/accounts/{account}/calendar")
def calendar(account:str,data:Calendar,cfg=Depends(feed)):
    if data.end<=data.start: raise HTTPException(422,"Invalid calendar interval")
    with state(account,cfg) as s: s["calendar"]=data.model_dump()
    return {"accepted":True}


@app.post("/accounts/{account}/paper-intents")
def enter(account:str,intent:Intent,cfg=Depends(feed)):
    now=time.time()
    with state(account,cfg) as s:
        return admit(s,intent,now)


def admit(s,intent,now,rule=None,ai=None):
    rollover(s,now)
    risk(s,now)
    c=s["config"]; ps=s["positions"]
    setup=next((v for v in c["setups"] if v["symbol"]==intent.symbol and v["strategy"]==intent.strategy),None)
    clock=datetime.fromtimestamp(now,timezone.utc)
    q=s["quotes"].get(intent.symbol)
    reason=s["locked"] or calendar_pause(s,now)
    if not setup or not setup["enabled"]: reason="Setup disabled"
    elif not c.get("always_open") and (clock.weekday()>4 or not setup["entry_utc"][0]<=clock.strftime("%H:%M")<setup["entry_utc"][1]): reason="Outside setup session"
    if not q or any(now-s["quotes"][sym]["timestamp"]>30 for sym in {intent.symbol}|{p["symbol"] for p in ps}): reason="Quote unavailable or stale"
    if len(ps)>=c["max_open"] or any(p["symbol"]==intent.symbol or p["strategy"]==intent.strategy for p in ps): reason="Exposure limit"
    if any(p["id"]==intent.id for p in ps+s["history"]): reason="Intent already processed"
    session="NY" if clock.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M")>="09:30" and intent.symbol=="XAUUSD" else "London"
    key=json.dumps([intent.symbol,intent.strategy,session])
    if setup and s["used"].get(key,0)>=setup["max_trades_per_session"]: reason="Session limit"
    bias=s["bias"].get(intent.symbol,{})
    if now-bias.get("as_of",0)>300 or bias.get("direction")!=("BULLISH" if intent.side=="BUY" else "BEARISH"): reason="Missing or conflicting market bias"
    if setup and setup.get("id") in {"XAUUSD_ORB","XAUUSD_EMA_VWAP","GER40_EMA_VWAP"} and c.get("ai_gate_enforce",False) and (not ai or ai["decision"]!="APPROVE"):
        reason="AI veto or unavailable"
    if reason: return {"command":"WAIT","reason":reason}
    buy=intent.side=="BUY"; entry=q["ask" if buy else "bid"]
    if not (intent.stop<q["bid"]<=entry<intent.target if buy else intent.target<entry<=q["ask"]<intent.stop):
        return {"command":"WAIT","reason":"Invalid protective levels"}
    budget=c.get("risk_amount_usd",min(equity(s),s["balance"])*c["risk_pct"]/100)
    raw=min(q["volume_max"],budget/(abs(entry-intent.stop)*q["cash_per_price_unit"]))
    if raw<q["volume_min"]: return {"command":"WAIT","reason":"Below minimum volume"}
    volume=q["volume_min"]+math.floor((raw-q["volume_min"])/q["volume_step"])*q["volume_step"]
    position=dict(**intent.model_dump(),entry=entry,volume=volume,opened_at=now,exit_rule=rule)
    if intent.symbol=="XAUUSD" and session=="London":
        position["session_end"]=clock.astimezone(ZoneInfo("America/New_York")).replace(hour=9,minute=30,second=0,microsecond=0).timestamp()
    if ai: position["ai"]=ai
    ps.append(position)
    s.setdefault("alerts",[]).append(dict(position,setup_id=setup.get("id"),execution_mode=c.get("execution_mode","paper")))
    s["used"][key]=s["used"].get(key,0)+1
    risk(s,now)
    return {"paper_mode":True,"id":intent.id,"locked":s["locked"]}


@app.get("/accounts/{account}/status")
def status(account:str,cfg=Depends(control)):
    now=time.time()
    with state(account,cfg) as s:
        rollover(s,now)
        daily,dd=risk(s,now)
        return dict(paper_mode=True,broker_execution_requested=bool(s.get("registration",{}).get("enabled")),profile=s["profile"],config=s["config"],balance=s["balance"],equity=equity(s),
                    floating_pnl=equity(s)-s["balance"],daily_loss_pct=daily,drawdown_pct=dd,locked=s["locked"],
                    news_pause=calendar_pause(s,now),positions=s["positions"],
                    quote_times={k:v["timestamp"] for k,v in s["quotes"].items()})


@app.get("/accounts/{account}/history")
def history(account:str,cfg=Depends(control)):
    with state(account,cfg) as s: return s["history"]


@app.patch("/accounts/{account}/settings")
def settings(account:str,patch:Settings,cfg=Depends(control)):
    with state(account,cfg) as s:
        rollover(s,time.time())
        for key,value in patch.model_dump(exclude_none=True,exclude={"setups"}).items():
            if key.endswith("_usd") and key not in s["config"]: raise HTTPException(422,"Account does not use USD limits")
            s["config"][key]=value
            if key=="max_dd_cap_pct": s["config"]["max_drawdown_cap_pct"]=value
        for toggle in patch.setups:
            setup=next((v for v in s["config"]["setups"] if (v["symbol"],v["strategy"])==(toggle.symbol,toggle.strategy)),None)
            if not setup: raise HTTPException(422,"Unknown setup")
            setup["enabled"]=toggle.enabled
        risk(s,time.time())
        return s["config"]


@app.post("/accounts/{account}/commands/FLATTEN_ALL_AND_PAUSE")
def emergency(account:str,cfg=Depends(control)):
    with state(account,cfg) as s: flatten(s,"EMERGENCY",time.time())
    return {"locked":"EMERGENCY","positions":[]}


class BiasBar(Payload):
    open:float|None=Field(default=None,gt=0)
    tick_volume:int|None=Field(default=None,ge=0)
    time:int=Field(gt=0)
    high:float=Field(gt=0)
    low:float=Field(gt=0)
    close:float=Field(gt=0)

    @model_validator(mode="after")
    def bounds(self):
        if not self.low<=self.close<=self.high: raise ValueError("Invalid candle")
        return self


class HistoryBars(Payload):
    M5:list[BiasBar]|None=Field(default=None,max_length=500)
    H1:list[BiasBar]|None=Field(default=None,max_length=500)
    H4:list[BiasBar]=Field(min_length=50,max_length=500)
    D1:list[BiasBar]=Field(min_length=20,max_length=500)
    M15:list[BiasBar]=Field(min_length=1,max_length=500)


def admit_reviewed_candidate(s,symbol,setup,side,stop,end,snapshot,review_id,assessment=None):
    now=time.time()
    q=s["quotes"].get(symbol)
    if not q or not 0<=now-end<=30 or s["bias"][symbol]["as_of"]!=snapshot["as_of"]: return
    rule=setup["profile"].split("/")[0]
    rr={"partialtrail":1.,"be":1.,"2R":2.,"1.5R":1.5}[rule]
    entry=q["ask"] if side==1 else q["bid"]
    target=entry+rr*(entry-stop)
    if target<=0: return
    intent=Intent(id=review_id,symbol=symbol,strategy=setup["strategy"],
                  side="BUY" if side==1 else "SELL",stop=stop,target=target)
    admit(s,intent,now,rule,assessment)


@app.post("/accounts/{account}/bars/{symbol}")
def update_bias(account:str,symbol:str,bars:HistoryBars,cfg=Depends(feed)):
    import pandas as pd
    from paper_bias import describe,commentary
    with state(account,cfg) as s:
        if symbol not in {v["symbol"] for v in s["config"]["setups"]}: raise HTTPException(422,"Unknown symbol")
    try:
        frames={k:pd.DataFrame(v) for k,v in bars.model_dump(exclude_none=True).items()}
        snapshot=describe(frames,time.time())
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    snapshot["symbol"]=symbol
    with state(account,cfg) as s:
        now=time.time()
        s["bias"][symbol]=snapshot
        if {"H1","M5"}<=frames.keys() and not s["config"].get("always_open"):
            from paper_signals import decisions
            try:
                candidates,last=decisions(frames,symbol,s["config"]["setups"],now)
            except (ValueError,KeyError,TypeError) as exc:
                raise HTTPException(422,"Invalid strategy bars") from exc
            for p in s["positions"]:
                if p["symbol"]!=symbol: continue
                buy=p["side"]=="BUY"
                if p.get("exit_rule")=="partialtrail" and p.get("partial") and "M15" in last:
                    end,price,atr,_,_=last["M15"]
                    if end>=p["partial_at"] and math.isfinite(atr):
                        p["stop"]=max(p["stop"],price-2*atr) if buy else min(p["stop"],price+2*atr)
                if p.get("exit_rule")=="be" and "M5" in last:
                    end,_,_,high,low=last["M5"]
                    if end-300>=int(p["opened_at"]//300)*300 and (high>=p["target"] if buy else low>=p["target"]):
                        p["stop"]=p["entry"]
            q=s["quotes"].get(symbol)
            pending=candidates
            atr={tf:values[2] for tf,values in last.items() if math.isfinite(values[2])}
        else:
            pending=[]; atr={}
        cal=s.get("calendar") or {}
        context={"market":snapshot.copy(),"atr14":atr,"news_pause":calendar_pause(s,now),
                 "calendar_covered":cal.get("start",now+1)<=now-900 and cal.get("end",0)>=now+900,
                 "upcoming_events":[e for e in cal.get("events",[]) if now-900<=e["timestamp"]<=now+14400
                                    and e["currency"].upper() in ("USD","EUR") and e["impact"].lower()=="high"]}
    # Model I/O must never hold the transaction used to mark equity or flatten.
    from signal_ai import review
    latest=None
    for setup,side,stop,end in pending:
        review_id=f"{symbol}/{setup['strategy']}/{end}"
        with state(account,cfg) as s:
            if any(row["id"]==review_id for row in s.get("ai_reviews",[])): continue
            enforced=s["config"].get("ai_gate_enforce",False)
            if not enforced:
                admit_reviewed_candidate(s,symbol,setup,side,stop,end,snapshot,review_id)
        assessment=review({**context,"candidate":{"id":setup.get("id"),"side":"BUY" if side==1 else "SELL",
                                                    "stop":stop,"signal_close":end,"profile":setup["profile"]}})
        latest=assessment
        with state(account,cfg) as s:
            if any(row["id"]==review_id for row in s.get("ai_reviews",[])): continue
            now=time.time()
            reviews=s.setdefault("ai_reviews",[])
            reviews.append(dict(id=review_id,at=now,**assessment))
            del reviews[:-200]
            if enforced:
                admit_reviewed_candidate(s,symbol,setup,side,stop,end,snapshot,review_id,assessment)
    note=latest or (review(context) if os.getenv("AI_API_KEY") else commentary(snapshot))
    if note:
        with state(account,cfg) as s:
            if s["bias"][symbol]["as_of"]==snapshot["as_of"]: s["bias"][symbol]["ai"]=note
    return {"accepted":True}


@app.get("/market-bias")
def market_bias(account:str,cfg=Depends(signals)):
    with state(account,cfg) as s:
        result={}
        for symbol in {v["symbol"] for v in s["config"]["setups"]}:
            snapshot=s["bias"].get(symbol)
            if snapshot and time.time()-snapshot["as_of"]<=300: result[symbol]=snapshot
            else: result[symbol]=dict(direction="NEUTRAL",stale=True,ai=None)
        return result


@app.get("/accounts/{account}/alerts")
def alerts(account:str,cfg=Depends(signals)):
    with state(account,cfg) as s: return s.get("alerts",[])


@app.post("/accounts/{account}/calendar/refresh")
def refresh_calendar(account:str,cfg=Depends(feed)):
    from paper_calendar import fetch_week
    import requests
    with state(account,cfg) as s:
        if not s["config"].get("news",{}).get("enabled",True): return {"status":"disabled"}
    try:
        data=fetch_week(time.time())
    except (requests.RequestException,ValueError,TypeError,KeyError):
        return {"status":"cached_or_offline"}
    with state(account,cfg) as s: s["calendar"]=data
    return {"status":"refreshed"}

@app.get("/accounts/{account}/feed-config")
def feed_config(account:str,cfg=Depends(feed)):
    with state(account,cfg) as s:
        c=s["config"]
        return {"symbols":sorted({v["symbol"] for v in c["setups"]}),"currency":c.get("account_currency")}

class Registration(Payload):
    mt5_account_id:str
    mt5_server:str
    risk_mode:Literal["percent","fixed_usd"]
    risk_value:float=Field(gt=0)
    daily_loss_cap:float=Field(gt=0)
    max_dd_cap:float=Field(gt=0,le=7.5)
    demo_live_execution:bool=False


@app.post("/ea/register-and-start")
def register_ea(data:Registration,authorization:str=Header(default=""),x_device_id:str=Header(default="")):
    from mt5_live_adapter import real_allowed
    r=licenses.licensed(authorization,x_device_id,"EA_ONLY")
    if (data.mt5_account_id,data.mt5_server)!=(r["login"],r["server"]):
        raise HTTPException(403,"Bound MT5 account mismatch")
    if data.demo_live_execution and "demo" not in data.mt5_server.lower() and not real_allowed():
        raise HTTPException(403,"Demo server required")
    cfg=json.loads(os.getenv("PAPER_ACCOUNTS_JSON","{}")).get(r["account"])
    if not cfg: raise HTTPException(403,"Engine account unavailable")
    with state(r["account"],cfg) as s:
        c=s["config"]
        if data.demo_live_execution and c.get("execution_mode")=="signal_only":
            raise HTTPException(403,"Signal-only profile")
        if s["locked"] or s["positions"]: raise HTTPException(409,"Engine locked or has open positions")
        if data.risk_mode=="fixed_usd":
            if c.get("account_currency")!="USD": raise HTTPException(422,"Fixed USD profile required")
            c.update(risk_amount_usd=data.risk_value,daily_loss_cap_usd=data.daily_loss_cap)
        else:
            if data.risk_value>.75 or data.daily_loss_cap>2.5 or "risk_amount_usd" in c:
                raise HTTPException(422,"Invalid percentage limits or profile")
            c.update(risk_pct=data.risk_value,daily_loss_cap_pct=data.daily_loss_cap)
        c.update(max_dd_cap_pct=data.max_dd_cap,max_drawdown_cap_pct=data.max_dd_cap)
        s["registration"]=dict(license_id=r["id"],generation=r["generation"],login=r["login"],server=r["server"],
                               enabled=data.demo_live_execution,started=time.time())
    return {"account":r["account"],"mode":"broker_requested" if data.demo_live_execution else "paper",
            "requires_connected_bridge":data.demo_live_execution}


@app.get("/accounts/{account}/execution-plan")
def execution_plan(account:str,cfg=Depends(feed)):
    with state(account,cfg,include_db=True) as (s,db):
        reg=s.get("registration")
        if not reg: return {"registered":False}
        valid=db.execute("""SELECT 1 FROM mobile_license WHERE id=? AND generation=? AND revoked=0
                AND (expires IS NULL OR expires>?) AND account=? AND login=? AND server=?""",
                (reg["license_id"],reg["generation"],time.time(),account,reg["login"],reg["server"])).fetchone()
        enabled=bool(valid and reg["enabled"] and not s["locked"])
        return {"registered":True,"enabled":enabled,"login":reg["login"],"server":reg["server"],
                "positions":s["positions"] if enabled else [],"risk":s["config"]}


@app.get("/accounts/{account}/ai-reviews")
def ai_reviews(account:str,cfg=Depends(signals)):
    with state(account,cfg) as s: return s.get("ai_reviews",[])
