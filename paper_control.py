"""Account-scoped, SQLite-serialized paper execution. All prices are bid/ask."""
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
app=FastAPI(title="UniversalAI paper account control")


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
    risk_pct:float|None=Field(default=None,gt=0,le=.75)
    daily_loss_cap_pct:float|None=Field(default=None,gt=0,le=2.5)
    max_dd_cap_pct:float|None=Field(default=None,gt=0,le=7.5)
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


def authorize(role):
    def check(account:str,authorization:str=Header(default="")):
        registry=json.loads(os.getenv("PAPER_ACCOUNTS_JSON","{}"))
        cfg=registry.get(account,{})
        token=os.getenv(cfg.get(role+"_token_env",""),"")
        tokens=[os.getenv(v.get(k,""),"") for v in registry.values() for k in ("control_token_env","feed_token_env")]
        if not token or tokens.count(token)!=1 or not secrets.compare_digest(authorization.encode(),("Bearer "+token).encode()):
            raise HTTPException(401,"Invalid account token")
        return cfg
    return check


control=authorize("control")
feed=authorize("feed")


@contextmanager
def state(account,cfg):
    db=sqlite3.connect(os.getenv("PAPER_STATE_DB",ROOT/"paper_state.db"),timeout=10)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS engines(account TEXT PRIMARY KEY, state TEXT NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
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
        yield s
        db.execute("INSERT OR REPLACE INTO engines VALUES(?,?)",(account,json.dumps(s,allow_nan=False)))
        db.commit()
    finally:
        db.close()


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
    elif daily>=s["config"]["daily_loss_cap_pct"]-1e-10 and not s["locked"]:
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
            if clock.date()!=opened.date() or clock.strftime("%H:%M")>=window[1]:
                close(s,p,"SESSION_END",now)
                continue
            buy=p["side"]=="BUY"
            if (q.bid<=p["stop"] if buy else q.ask>=p["stop"]): close(s,p,"STOP",now)
            elif (q.bid>=p["target"] if buy else q.ask<=p["target"]): close(s,p,"TARGET",now)
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
        rollover(s,now)
        risk(s,now)
        c=s["config"]; ps=s["positions"]
        setup=next((v for v in c["setups"] if v["symbol"]==intent.symbol and v["strategy"]==intent.strategy),None)
        clock=datetime.fromtimestamp(now,timezone.utc)
        q=s["quotes"].get(intent.symbol)
        reason=s["locked"] or news_reason(s["calendar"],now)
        if not setup or not setup["enabled"]: reason="Setup disabled"
        elif clock.weekday()>4 or not setup["entry_utc"][0]<=clock.strftime("%H:%M")<setup["entry_utc"][1]: reason="Outside setup session"
        if not q or any(now-s["quotes"][sym]["timestamp"]>30 for sym in {intent.symbol}|{p["symbol"] for p in ps}): reason="Quote unavailable or stale"
        if len(ps)>=c["max_open"] or any(p["symbol"]==intent.symbol or p["strategy"]==intent.strategy for p in ps): reason="Exposure limit"
        if any(p["id"]==intent.id for p in ps+s["history"]): reason="Intent already processed"
        session="NY" if clock.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M")>="09:30" and intent.symbol=="XAUUSD" else "London"
        key=json.dumps([intent.symbol,intent.strategy,session])
        if setup and s["used"].get(key,0)>=setup["max_trades_per_session"]: reason="Session limit"
        bias=s["bias"].get(intent.symbol,{})
        if now-bias.get("as_of",0)>300 or bias.get("direction")!=("BULLISH" if intent.side=="BUY" else "BEARISH"): reason="Missing or conflicting market bias"
        if reason: return {"command":"WAIT","reason":reason}
        buy=intent.side=="BUY"; entry=q["ask" if buy else "bid"]
        if not (intent.stop<q["bid"]<=entry<intent.target if buy else intent.target<entry<=q["ask"]<intent.stop):
            return {"command":"WAIT","reason":"Invalid protective levels"}
        budget=min(equity(s),s["balance"])*c["risk_pct"]/100
        raw=min(q["volume_max"],budget/(abs(entry-intent.stop)*q["cash_per_price_unit"]))
        if raw<q["volume_min"]: return {"command":"WAIT","reason":"Below minimum volume"}
        volume=q["volume_min"]+math.floor((raw-q["volume_min"])/q["volume_step"])*q["volume_step"]
        ps.append(dict(**intent.model_dump(),entry=entry,volume=volume,opened_at=now))
        s["used"][key]=s["used"].get(key,0)+1
        risk(s,now)
        return {"paper_mode":True,"id":intent.id,"locked":s["locked"]}


@app.get("/accounts/{account}/status")
def status(account:str,cfg=Depends(control)):
    now=time.time()
    with state(account,cfg) as s:
        rollover(s,now)
        daily,dd=risk(s,now)
        return dict(paper_mode=True,profile=s["profile"],config=s["config"],balance=s["balance"],equity=equity(s),
                    floating_pnl=equity(s)-s["balance"],daily_loss_pct=daily,drawdown_pct=dd,locked=s["locked"],
                    news_pause=news_reason(s["calendar"],now),positions=s["positions"],
                    quote_times={k:v["timestamp"] for k,v in s["quotes"].items()})


@app.get("/accounts/{account}/history")
def history(account:str,cfg=Depends(control)):
    with state(account,cfg) as s: return s["history"]


@app.patch("/accounts/{account}/settings")
def settings(account:str,patch:Settings,cfg=Depends(control)):
    with state(account,cfg) as s:
        rollover(s,time.time())
        for key,value in patch.model_dump(exclude_none=True,exclude={"setups"}).items():
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
    time:int=Field(gt=0)
    high:float=Field(gt=0)
    low:float=Field(gt=0)
    close:float=Field(gt=0)

    @model_validator(mode="after")
    def bounds(self):
        if not self.low<=self.close<=self.high: raise ValueError("Invalid candle")
        return self


class HistoryBars(Payload):
    H4:list[BiasBar]=Field(min_length=50,max_length=500)
    D1:list[BiasBar]=Field(min_length=20,max_length=500)
    M15:list[BiasBar]=Field(min_length=1,max_length=500)


@app.post("/accounts/{account}/bars/{symbol}")
def update_bias(account:str,symbol:str,bars:HistoryBars,cfg=Depends(feed)):
    import pandas as pd
    from paper_bias import describe,commentary
    with state(account,cfg) as s:
        if symbol not in {v["symbol"] for v in s["config"]["setups"]}: raise HTTPException(422,"Unknown symbol")
    try:
        snapshot=describe({k:pd.DataFrame(v) for k,v in bars.model_dump().items()},time.time())
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    snapshot["symbol"]=symbol
    with state(account,cfg) as s: s["bias"][symbol]=snapshot
    note=commentary(snapshot)
    if note:
        with state(account,cfg) as s:
            if s["bias"][symbol]["as_of"]==snapshot["as_of"]: s["bias"][symbol]["ai"]=note
    return {"accepted":True}


@app.get("/market-bias")
def market_bias(account:str,cfg=Depends(control)):
    with state(account,cfg) as s:
        result={}
        for symbol in {v["symbol"] for v in s["config"]["setups"]}:
            snapshot=s["bias"].get(symbol)
            if snapshot and time.time()-snapshot["as_of"]<=300: result[symbol]=snapshot
            else: result[symbol]=dict(direction="NEUTRAL",stale=True,ai=None)
        return result
