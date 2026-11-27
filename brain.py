"""UniversalAI: broker-independent quant/SMC engine and FastAPI service.

MT5 and the LLM providers are optional adapters. This service never places orders; execution
and a second drawdown circuit breaker belong to the EA. See README.md for contracts.
"""
from __future__ import annotations
import json, logging, math, os, random, re, secrets, sqlite3, threading, time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from research_lab import STRATEGIES as RESEARCH_STRATEGIES, live_candidates
from strategies import SIGNAL_SCHEMA, _make_wait, validate_signal
import demo_orb
import xau_ny_breakout as gold

ROOT=Path(__file__).resolve().parent
load_dotenv(ROOT/".env",override=False)
DB_PATH=Path(os.getenv("UNIVERSALAI_DB",ROOT/"accounts.db"))
SYMBOLS=[s.strip() for s in os.getenv("MT5_SYMBOLS","EURUSD,XAUUSD,Volatility 75 Index,Boom 1000 Index,Crash 1000 Index").split(",") if s.strip()]
LLM_PROVIDERS=(
    ("GROQ","https://api.groq.com/openai/v1","openai/gpt-oss-20b"),
    ("GEMINI","https://generativelanguage.googleapis.com/v1beta/openai/","gemini-3.1-flash-lite"),
)
_provider_cooldowns:dict[str,float]={}
_cooldown_lock=threading.Lock()
LEVEL_TOLERANCE=float(os.getenv("UNIVERSALAI_LEVEL_TOLERANCE","0.10"))
_THINK_RE=re.compile(r"<think>.*?</think>",re.S|re.I)
LOG=logging.getLogger("universalai")
@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield

app=FastAPI(title="UniversalAI Trading API",version="1.0.0",lifespan=lifespan)
_scanner_lock=threading.Lock()
_analysis_cache:dict[str,tuple[int,tuple[str,...],dict[str,Any]|None]]={}
APPROVED_PATH=Path(os.getenv("UNIVERSALAI_APPROVED_PATH",ROOT/"approved_strategies.json"))

def _connect():
    c=sqlite3.connect(DB_PATH,timeout=10); c.row_factory=sqlite3.Row; c.execute("PRAGMA journal_mode=WAL"); return c

def init_db():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    with _connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS accounts(account_id TEXT PRIMARY KEY,broker_server TEXT NOT NULL,account_type TEXT NOT NULL CHECK(account_type IN ('prop','broker')),risk_pct REAL NOT NULL CHECK(risk_pct>0 AND risk_pct<=5),max_daily_dd_pct REAL NOT NULL CHECK(max_daily_dd_pct>0 AND max_daily_dd_pct<=100),max_total_dd_pct REAL NOT NULL CHECK(max_total_dd_pct>0 AND max_total_dd_pct<=100),active INTEGER NOT NULL DEFAULT 0,daily_start_balance REAL NOT NULL DEFAULT 0,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS trade_logs(id INTEGER PRIMARY KEY AUTOINCREMENT,account_id TEXT NOT NULL,symbol TEXT NOT NULL,signal_id TEXT NOT NULL,decision TEXT NOT NULL,confidence REAL NOT NULL DEFAULT 0,reason TEXT NOT NULL DEFAULT '',signal_json TEXT NOT NULL DEFAULT '{}',created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,UNIQUE(account_id,signal_id),FOREIGN KEY(account_id) REFERENCES accounts(account_id));
        CREATE TABLE IF NOT EXISTS account_state(account_id TEXT PRIMARY KEY,balance REAL NOT NULL DEFAULT 0,equity REAL NOT NULL DEFAULT 0,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS licenses(license_key TEXT PRIMARY KEY,owner_name TEXT NOT NULL,bound_account_id TEXT NULL,bound_device_id TEXT NULL,expires_at TEXT NULL,is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0,1)),created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        """)

class AccountConfig(BaseModel):
    account_id:str=Field(min_length=1,max_length=40)
    broker_server:str=Field(min_length=1,max_length=120)
    account_type:Literal["prop","broker"]
    risk_pct:float=Field(gt=0,le=5)
    max_daily_dd_pct:float=Field(gt=0,le=100)
    max_total_dd_pct:float=Field(gt=0,le=100)
    active:bool=False
    daily_start_balance:float=Field(default=0,ge=0)

class ActivationRequest(BaseModel):
    license_key:str=Field(min_length=1,max_length=32)
    account_id:str=Field(min_length=1,max_length=40)
    device_id:str=Field(min_length=8,max_length=128)

class LicenseCreate(BaseModel):
    owner_name:str=Field(min_length=1,max_length=120)
    bound_account_id:str|None=Field(default=None,max_length=40)
    expiry_days:int|None=Field(default=None,ge=1,le=3650)

class LicenseToggle(BaseModel):
    is_active:bool

class LicenseContext(BaseModel):
    account_id:str
    license_key:str
    device_id:str

def _expired(raw:str)->bool:
    try:exp=datetime.fromisoformat(raw)
    except (TypeError,ValueError):return True
    if exp.tzinfo is None:exp=exp.replace(tzinfo=timezone.utc)
    return exp<=datetime.now(timezone.utc)

def _valid_license(license_key:str,account_id:str,device_id:str|None=None,bind:bool=False)->bool:
    key=license_key.strip().upper()
    if not key or not account_id:return False
    if device_id is not None and len(device_id.strip())<8:return False
    with _connect() as c:
        if bind:c.execute("BEGIN IMMEDIATE")
        row=c.execute("SELECT * FROM licenses WHERE license_key=?",(key,)).fetchone()
        if not row or not row["is_active"]:return False
        if row["expires_at"] and _expired(row["expires_at"]):return False
        if row["bound_account_id"] and row["bound_account_id"]!=account_id:return False
        if device_id is not None:
            if row["bound_device_id"] and row["bound_device_id"]!=device_id:return False
            if bind:
                c.execute("UPDATE licenses SET bound_account_id=COALESCE(bound_account_id,?),bound_device_id=COALESCE(bound_device_id,?) WHERE license_key=?",(account_id,device_id,key))
        elif not row["bound_account_id"] or not row["bound_device_id"]:
            return False
        return True

def _ea_authorized(key:str,account_id:str,device_id:str)->bool:
    """EA polls: key must be active/unexpired/bound. If the EA sends X-Device-ID it must match the claimed device;
    set UNIVERSALAI_REQUIRE_EA_DEVICE=1 to make the header mandatory."""
    if not _valid_license(key,account_id):return False
    dev=device_id.strip()
    if dev:return _valid_license(key,account_id,dev)
    return os.getenv("UNIVERSALAI_REQUIRE_EA_DEVICE","0")!="1"

def require_dashboard_license(x_license_key:str=Header(default=""),x_license_account_id:str=Header(default=""),x_device_id:str=Header(default=""))->LicenseContext:
    if not _valid_license(x_license_key,x_license_account_id,x_device_id,bind=True):raise HTTPException(403,"License activation is invalid or bound to another account/device")
    return LicenseContext(account_id=x_license_account_id,license_key=x_license_key.upper(),device_id=x_device_id)

def require_admin(x_admin_pin:str=Header(default="")):
    pin=os.getenv("ADMIN_PIN","")
    if not pin:raise HTTPException(503,"ADMIN_PIN is not configured")
    if not secrets.compare_digest(x_admin_pin,pin):raise HTTPException(401,"Invalid admin PIN")

def _frame(data:Any)->pd.DataFrame:
    d=data.copy() if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    aliases={"tick_volume":"volume","real_volume":"volume"}
    d=d.rename(columns={k:v for k,v in aliases.items() if k in d and v not in d})
    if not {"open","high","low","close"}.issubset(d.columns): raise ValueError("OHLC columns required")
    return d.reset_index(drop=True)

def quant_metrics_series(data:Any)->pd.DataFrame:
    """Vectorized gate metrics for every row, using only current and earlier bars."""
    d=_frame(data); h,l,c=d.high.astype(float),d.low.astype(float),d.close.astype(float); prev=c.shift(1)
    tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    a14=tr.rolling(14,min_periods=14).mean(); a50=tr.rolling(50,min_periods=50).mean()
    mean=c.rolling(50,min_periods=50).mean(); std=c.rolling(50,min_periods=50).std(ddof=0)
    er=c.diff(20).abs()/c.diff().abs().rolling(20,min_periods=20).sum().replace(0,np.nan)
    ratio=a14/a50.replace(0,np.nan); z=(c-mean)/std.replace(0,np.nan); z=z.fillna(0.)
    ready=a14.notna()&a50.notna()&mean.notna()&std.notna()&er.notna()&(a50>0)
    return pd.DataFrame({"ready":ready,"atr14":a14,"atr50":a50,"atr_ratio":ratio.fillna(0.),"zscore":z,"efficiency_ratio":er.fillna(0.),"gate":ready&(ratio>.90)&(er>.25)},index=d.index)

def quant_metrics(data:Any)->dict[str,Any]:
    """ATR14/ATR50, 50-close z-score, and Kaufman ER20 for the latest row."""
    m=quant_metrics_series(data).iloc[-1]
    return {"ready":bool(m.ready),"atr14":float(m.atr14) if pd.notna(m.atr14) else 0.,"atr50":float(m.atr50) if pd.notna(m.atr50) else 0.,"atr_ratio":float(m.atr_ratio),"zscore":float(m.zscore),"efficiency_ratio":float(m.efficiency_ratio),"gate":bool(m.gate)}

def quant_allows(metrics:dict[str,Any],side:str)->bool:
    if not metrics.get("ready") or not metrics.get("gate"): return False
    z=float(metrics["zscore"]); return not ((side=="BUY" and z>2.) or (side=="SELL" and z< -2.))

def market_bias(data:Any)->str:
    """H1 EMA20/50 aligned with the two latest confirmed fractal HH/HL or LH/LL."""
    d=_frame(data)
    if len(d)<55:return "NEUTRAL"
    c=d.close.astype(float); e20=c.ewm(span=20,adjust=False).mean().iloc[-1]; e50=c.ewm(span=50,adjust=False).mean().iloc[-1]
    h,l=d.high.astype(float),d.low.astype(float)
    ph=h[(h.shift(2)<h)&(h.shift(1)<h)&(h.shift(-1)<h)&(h.shift(-2)<h)].dropna().tail(2)
    pl=l[(l.shift(2)>l)&(l.shift(1)>l)&(l.shift(-1)>l)&(l.shift(-2)>l)].dropna().tail(2)
    if len(ph)<2 or len(pl)<2:return "NEUTRAL"
    if e20>e50 and ph.iloc[-1]>ph.iloc[-2] and pl.iloc[-1]>pl.iloc[-2]:return "BULLISH"
    if e20<e50 and ph.iloc[-1]<ph.iloc[-2] and pl.iloc[-1]<pl.iloc[-2]:return "BEARISH"
    return "NEUTRAL"

def market_bias_series(data:Any)->pd.Series:
    """Vectorized H1 EMA/swing bias at each bar; pivots become known after 2 bars."""
    d=_frame(data); c=d.close.astype(float); h,l=d.high.astype(float),d.low.astype(float)
    e20=c.ewm(span=20,adjust=False).mean();e50=c.ewm(span=50,adjust=False).mean()
    ph=(h.shift(2)<h)&(h.shift(1)<h)&(h.shift(-1)<h)&(h.shift(-2)<h)
    pl=(l.shift(2)>l)&(l.shift(1)>l)&(l.shift(-1)>l)&(l.shift(-2)>l)
    ch=h.where(ph).shift(2);cl=l.where(pl).shift(2)
    last_h=ch.ffill();last_l=cl.ffill()
    higher=(ch>last_h.shift(1)).astype("boolean").where(ch.notna()).ffill().fillna(False).astype(bool)
    lower=(ch<last_h.shift(1)).astype("boolean").where(ch.notna()).ffill().fillna(False).astype(bool)
    higher_l=(cl>last_l.shift(1)).astype("boolean").where(cl.notna()).ffill().fillna(False).astype(bool)
    lower_l=(cl<last_l.shift(1)).astype("boolean").where(cl.notna()).ffill().fillna(False).astype(bool)
    bullish=(e20>e50)&higher&higher_l;bearish=(e20<e50)&lower&lower_l
    return pd.Series(np.select([bullish,bearish],["BULLISH","BEARISH"],default="NEUTRAL"),index=d.index)

def detect_smc(data_m15:Any,bias:str)->dict[str,Any]|None:
    """Recent 20-bar liquidity sweep plus a three-candle FVG in H1 bias.

    The sweep may occur in the four bars before the completed FVG. Requiring a
    sweep and FVG on the same bar is geometrically incompatible with the prior
    20-bar extreme because the FVG's first candle is inside that lookback.
    """
    d=_frame(data_m15)
    if len(d)<25 or bias not in ("BULLISH","BEARISH"):return None
    i=len(d)-1; up=float(d.low.iloc[i])>float(d.high.iloc[i-2]); down=float(d.high.iloc[i])<float(d.low.iloc[i-2])
    if not (up if bias=="BULLISH" else down):return None
    for j in range(i-2,max(19,i-6),-1):
        prev=d.iloc[j-20:j]; b=d.iloc[j]
        if bias=="BULLISH" and b.low<prev.low.min() and b.close>prev.low.min():
            return {"side":"BUY","sweep_level":float(prev.low.min()),"fvg_low":float(d.high.iloc[i-2]),"fvg_high":float(d.low.iloc[i]),"sweep_bar":j,"fvg_bar":i}
        if bias=="BEARISH" and b.high>prev.high.max() and b.close<prev.high.max():
            return {"side":"SELL","sweep_level":float(prev.high.max()),"fvg_low":float(d.high.iloc[i]),"fvg_high":float(d.low.iloc[i-2]),"sweep_bar":j,"fvg_bar":i}
    return None

def is_synthetic(symbol:str)->bool:
    s=symbol.lower();compact="".join(ch for ch in s if ch.isalnum())
    return any(x in s for x in ("volatility","boom","crash","step index","skew step","jump","synthetic")) or "rangebreak" in compact

def in_kill_zone(now:datetime|None=None)->bool:
    """UTC London (07-10) and New York (12-16); broker DST calendars are not inferred."""
    now=now or datetime.now(timezone.utc); m=now.hour*60+now.minute
    return 420<=m<600 or 720<=m<960

def _reject_constant(name:str)->float:raise ValueError("non-finite JSON constant "+name)

def _finite(v:Any)->float:
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)):raise ValueError("non-finite or non-numeric field")
    return float(v)

def _unique_json_object(pairs:list[tuple[str,Any]])->dict[str,Any]:
    result=dict(pairs)
    if len(result)!=len(pairs):raise ValueError("duplicate JSON field")
    return result

def _parse_model_json(content:Any)->dict[str,Any]:
    """Strict parse: drop complete <think> blocks, reject unterminated ones, NaN/Infinity, trailing text and non-objects."""
    if not isinstance(content,str):raise ValueError("empty model content")
    text=_THINK_RE.sub("",content).strip()
    if re.search(r"</?think>",text,re.I):raise ValueError("unterminated think block")
    fenced=re.fullmatch(r"```(?:json)?\s*(.*?)\s*```",text,re.I|re.S)
    if fenced:text=fenced.group(1)
    out=json.loads(text,parse_constant=_reject_constant,object_pairs_hook=_unique_json_object)
    if not isinstance(out,dict):raise ValueError("model JSON is not an object")
    return out

def _wait_signal(reason:str)->dict[str,Any]:
    return {"command":"WAIT","decision":"WAIT","entry":0.,"sl":0.,"tp":0.,"confidence":0.,"reason":reason}

def _retry_after_seconds(header:str)->float:
    try:
        try:
            seconds=float(header)
        except ValueError:
            seconds=(parsedate_to_datetime(header)-datetime.now(timezone.utc)).total_seconds()
    except (ValueError, TypeError, OverflowError):
        return 30.
    return min(300.,max(0.,seconds)) if math.isfinite(seconds) else 30.


def _signal_messages(context:dict[str,Any])->list[dict[str,str]]:
    properties={key:{"type":"number" if kind is float else "string"} for key,kind in SIGNAL_SCHEMA.items()}
    properties["command"]["enum"]=["BUY","SELL","WAIT"]
    schema={"type":"object","properties":properties,"required":list(SIGNAL_SCHEMA),"additionalProperties":False}
    instructions=(
        "You are a signal validator for a supplied deterministic setup. Output exactly one JSON object "
        "matching this schema: "+json.dumps(schema)+". "
        "Use finite JSON numbers for sl, tp and confidence. setup and reason must be nonempty strings. "
        "Use the supplied strategy as the setup label when present. "
        "For BUY or SELL, keep the supplied side and copy setup.stop to sl and setup.target to tp. "
        "The entry reference is supplied in setup.entry_reference and is never an output key. "
        "All three prices must be positive. BUY requires sl < entry reference < tp. "
        "SELL requires tp < entry reference < sl. Reward divided by risk must be at least 2.5, "
        "and actionable confidence must be between 0.75 and 1 inclusive. "
        "If the setup is incomplete, invalid or uncertain, return WAIT with sl, tp and confidence all 0.0 "
        "and explain why in reason. Do not include any other keys or text."
    )
    return [{"role":"system","content":instructions},{"role":"user","content":json.dumps(context)}]


def evaluate_llm_signal(context:dict[str,Any])->dict[str,Any]:
    """Return the exact strategy signal schema. EA compatibility is handled by the caller."""
    setup_label=str(context.get("strategy") or "signal_evaluation")
    def wait(reason:str)->dict[str,Any]:
        return _make_wait(reason,setup_label)

    messages=_signal_messages(context)
    try:
        token_budget=int(os.getenv("LLM_MAX_COMPLETION_TOKENS","1024"))
        if token_budget<=0:raise ValueError
    except ValueError:
        return wait("LLM_MAX_COMPLETION_TOKENS must be a positive integer")
    failures=[]
    for provider,default_url,default_model in LLM_PROVIDERS:
        key=os.getenv(provider+"_API_KEY","").strip()
        if not key:
            failures.append(provider+"_API_KEY is not configured")
            continue
        with _cooldown_lock:
            cooling_down=_provider_cooldowns.get(provider,0.)>time.monotonic()
        if cooling_down:
            failures.append(f"{provider} is cooling down after HTTP 429")
            continue
        model=os.getenv(provider+"_MODEL",default_model)
        base_url=os.getenv(provider+"_BASE_URL",default_url).rstrip("/")
        body={"model":model,"messages":messages,"temperature":.1,"response_format":{"type":"json_object"}}
        if provider=="GROQ":
            body["max_completion_tokens"]=token_budget
            if "qwen" in model.lower():body["reasoning_format"]="hidden"
            if model.startswith("openai/gpt-oss"):body["reasoning_effort"]="low"
        else:
            body["max_tokens"]=4096
        try:
            response=requests.post(base_url+"/chat/completions",timeout=(3.05,20),headers={"Authorization":"Bearer "+key},json=body)
            response.raise_for_status()
        except requests.RequestException as exc:
            status=exc.response.status_code if exc.response is not None else None
            if status==429:
                delay=_retry_after_seconds(exc.response.headers.get("retry-after","30"))
                with _cooldown_lock:
                    _provider_cooldowns[provider]=max(_provider_cooldowns.get(provider,0.),time.monotonic()+delay)
            failures.append(f"{provider} HTTP {status}" if status is not None else f"{provider} {type(exc).__name__}")
            continue
        try:
            choice=response.json()["choices"][0]
            if choice["finish_reason"]!="stop":return wait(f"{provider} returned incomplete output")
            result=_parse_model_json(choice["message"]["content"])
            for field,kind in SIGNAL_SCHEMA.items():
                if kind is float:result[field]=_finite(result[field])
            if validate_signal(result)!=result:raise ValueError("invalid signal schema")
        except (ValueError,TypeError,IndexError,KeyError,OverflowError):
            return wait(f"{provider} returned invalid signal JSON, schema, or SL/TP geometry")
        result["reason"]=result["reason"][:1000].rstrip() or "Provider signal"
        if result["command"]=="WAIT":return result
        try:
            setup=context["setup"]
            entry=_finite(setup["entry_reference"])
        except (KeyError,TypeError,ValueError,OverflowError):
            return wait("Missing or invalid setup entry reference")
        if not _valid_signal_risk(result,entry,context.get("side")):
            return wait(f"{provider} signal failed side, confidence, or SL/TP risk validation")
        if not levels_match_setup({**result,"entry":entry},setup):
            return wait(f"{provider} signal levels differ from the approved setup")
        return result
    return wait("; ".join(failures))


def _valid_signal_risk(signal:dict[str,Any],entry:float,side:str|None)->bool:
    sl,tp,confidence=(signal[k] for k in ("sl","tp","confidence"))
    if signal["command"]!=side or not all(math.isfinite(v) for v in (entry,sl,tp,confidence)):
        return False
    if min(entry,sl,tp)<=0 or not .75<=confidence<=1:return False
    directional=(side=="BUY" and sl<entry<tp) or (side=="SELL" and tp<entry<sl)
    if not directional:return False
    reward_risk=abs(tp-entry)/abs(entry-sl)
    return math.isfinite(reward_risk) and reward_risk>=2.5


def validate_decision(result:dict[str,Any],side:str)->bool:
    try:
        entry=float(result["entry"])
        signal={key:float(result[key]) for key in ("sl","tp","confidence")}
    except (KeyError,TypeError,ValueError,OverflowError):
        return False
    return _valid_signal_risk({**signal,"command":result.get("decision")},entry,side)

def levels_match_setup(result:dict[str,Any],setup:Any)->bool:
    """The model may veto a setup, not re-price it: entry/SL/TP must sit within LEVEL_TOLERANCE x risk of the deterministic setup."""
    try:
        ref,stop,target=(float(setup[k]) for k in ("entry_reference","stop","target"))
        e,sl,tp=(float(result[k]) for k in ("entry","sl","tp"))
    except (KeyError,TypeError,ValueError):return False
    risk=abs(ref-stop)
    if not all(math.isfinite(v) for v in (ref,stop,target,e,sl,tp)) or risk<=0:return False
    tol=risk*LEVEL_TOLERANCE
    return abs(e-ref)<=tol and abs(sl-stop)<=tol and abs(tp-target)<=tol

def _mt5_rates(symbol:str,timeframe:str,count:int=300)->pd.DataFrame|None:
    try:import MetaTrader5 as mt5
    except ImportError:LOG.warning("MetaTrader5 package unavailable");return None
    if not mt5.initialize():LOG.warning("MT5 initialize failed: %s",mt5.last_error());return None
    tf=mt5.TIMEFRAME_M15 if timeframe=="M15" else mt5.TIMEFRAME_H1
    raw=mt5.copy_rates_from_pos(symbol,tf,1,count)
    if raw is None or len(raw)==0:LOG.warning("No %s rates for %s: %s",timeframe,symbol,mt5.last_error());return None
    return pd.DataFrame(raw)

def analyze_symbol(symbol:str,account_id:str|None=None,now:datetime|None=None)->dict[str,Any]|None:
    try:
        approval_doc=json.loads(APPROVED_PATH.read_text(encoding="utf-8"))
    except (OSError,ValueError):
        approval_doc={"approved":[]}
    approved=approval_doc.get("approved") if isinstance(approval_doc,dict) else None
    records=[r for r in (approved if isinstance(approved,list) else []) if isinstance(r,dict) and r.get("symbol")==symbol]
    strategy_names=tuple(sorted({str(r.get("strategy")) for r in records if r.get("strategy") in RESEARCH_STRATEGIES.values()}))
    # Fail closed until this exact symbol has passed the OOS/Monte Carlo gate.
    if not strategy_names:return None
    m15=_mt5_rates(symbol,"M15")
    if m15 is None:return None
    if m15.empty or "time" not in m15:
        return _wait_signal("Missing M15 timestamps")
    stamps=pd.to_datetime(m15.time,unit="s" if pd.api.types.is_numeric_dtype(m15.time) else None,utc=True,errors="coerce")
    if stamps.isna().any() or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
        return _wait_signal("Invalid M15 timestamps")
    bar_time=int(stamps.iloc[-1].timestamp())
    age=(now.timestamp() if now else time.time())-bar_time
    if not 900 <= age < 1800:
        return _wait_signal("Stale or incomplete M15 bar")
    price_columns=["open","high","low","close"]
    if not set(price_columns).issubset(m15):
        return _wait_signal("Missing M15 OHLC")
    prices=m15[price_columns].apply(pd.to_numeric,errors="coerce")
    bounds=prices[["open","close"]]
    if (not np.isfinite(prices.to_numpy()).all() or (prices<=0).any().any()
            or (prices.low>bounds.min(axis=1)).any() or (prices.high<bounds.max(axis=1)).any()):
        return _wait_signal("Invalid M15 OHLC")
    m15=m15.assign(time=stamps,**{key:prices[key] for key in price_columns})
    signature=strategy_names+(str(APPROVED_PATH.stat().st_mtime_ns) if APPROVED_PATH.exists() else "missing",)
    cached=_analysis_cache.get(symbol)
    if cached and cached[0]==bar_time and cached[1]==signature:
        result=cached[2]
        if result and account_id and result.get("decision") in ("BUY","SELL"):
            with _connect() as c:c.execute("INSERT OR IGNORE INTO trade_logs(account_id,symbol,signal_id,decision,confidence,reason,signal_json) VALUES(?,?,?,?,?,?,?)",(account_id,symbol,result["signal_id"],result["decision"],result["confidence"],result["reason"],json.dumps(result)))
        return result
    m15["time"]=pd.to_datetime(m15.time,unit="s",utc=True) if not pd.api.types.is_datetime64_any_dtype(m15.time) else m15.time
    m15["asset_class"]=str(records[0].get("asset_class",""))
    try:
        candidates=live_candidates(symbol,m15,list(strategy_names),ROOT/"data"/"research_2020_2025"/"models")
    except Exception:
        LOG.exception("Approved strategy evaluation failed for %s",symbol)
        candidates=[]
    decision=None
    for candidate in candidates:
        side=candidate["side"]
        context={"symbol":symbol,"side":side,"strategy":candidate["strategy"],"setup":candidate["setup"],"quant_metrics":candidate["quant"],"asset_class":m15["asset_class"].iloc[-1],"last_completed_bar_utc":str(m15.time.iloc[-1])}
        signal=evaluate_llm_signal(context)
        if signal["command"]=="WAIT":
            proposed=_wait_signal(signal["reason"])
        else:
            proposed={**signal,"decision":signal["command"],"entry":float(candidate["setup"]["entry_reference"])}
        proposed.update({"symbol":symbol,"strategy":candidate["strategy"],"breakeven_at_r":1.2 if candidate["strategy"]==RESEARCH_STRATEGIES["A"] else 0.0,"bias":side,"quant":candidate["quant"],"setup":candidate["setup"],"bar_time":bar_time,"signal_id":f"{symbol}-{bar_time}-{candidate['strategy'].replace(' ','_')}","reason":proposed.get("reason","")})
        decision=proposed
        if decision["decision"] in ("BUY","SELL"):break
    if decision is None and candidates==[]:
        decision=None
    _analysis_cache[symbol]=(bar_time,signature,decision)
    if account_id:
        if decision:
            with _connect() as c:c.execute("INSERT OR IGNORE INTO trade_logs(account_id,symbol,signal_id,decision,confidence,reason,signal_json) VALUES(?,?,?,?,?,?,?)",(account_id,symbol,decision["signal_id"],decision["decision"],decision["confidence"],decision["reason"],json.dumps(decision)))
    return decision

@app.get("/")
def dashboard():return FileResponse(ROOT/"index.html")

@app.get("/manifest.webmanifest")
def pwa_manifest():return FileResponse(ROOT/"manifest.webmanifest",media_type="application/manifest+json")

@app.get("/sw.js")
def service_worker():return FileResponse(ROOT/"sw.js",media_type="application/javascript")

@app.get("/icon.svg")
def pwa_icon():return FileResponse(ROOT/"icon.svg",media_type="image/svg+xml")

@app.get("/api/health")
def health():return {"status":"ok","groq_configured":bool(os.getenv("GROQ_API_KEY"))}

@app.get("/api/strategies")
def approved_strategies(auth:LicenseContext=Depends(require_dashboard_license)):
    """Expose only OOS/Monte-Carlo-qualified strategies, never raw account data."""
    try:doc=json.loads(APPROVED_PATH.read_text(encoding="utf-8"))
    except (OSError,ValueError):doc={"approved":[]}
    items=[]
    for row in doc.get("approved",[]):
        if row.get("strategy") not in RESEARCH_STRATEGIES.values():continue
        items.append({"symbol":row.get("symbol"),"asset_class":row.get("asset_class"),"strategy":row.get("strategy"),"oos":row.get("oos",{}),"monte_carlo":row.get("monte_carlo",{})})
    return {"approved":items,"generated_at":doc.get("generated_at")}

@app.post("/api/activate")
def activate(request:ActivationRequest):
    if not _valid_license(request.license_key,request.account_id,request.device_id,bind=True):raise HTTPException(403,"License activation rejected")
    return {"activated":True,"account_id":request.account_id}

@app.post("/api/admin/licenses/create")
def create_license(request:LicenseCreate,_admin:None=Depends(require_admin)):
    alphabet="ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        raw="".join(secrets.choice(alphabet) for _ in range(12));key="UAI-"+raw[:4]+"-"+raw[4:8]+"-"+raw[8:]
        expires=(datetime.now(timezone.utc)+pd.Timedelta(days=request.expiry_days)).isoformat() if request.expiry_days else None
        try:
            with _connect() as c:c.execute("INSERT INTO licenses(license_key,owner_name,bound_account_id,expires_at) VALUES(?,?,?,?)",(key,request.owner_name.strip(),request.bound_account_id,expires))
            return {"license_key":key,"owner_name":request.owner_name.strip(),"bound_account_id":request.bound_account_id,"expires_at":expires,"is_active":True}
        except sqlite3.IntegrityError:continue

@app.get("/api/admin/licenses")
def list_licenses(_admin:None=Depends(require_admin)):
    with _connect() as c:rows=c.execute("SELECT * FROM licenses ORDER BY created_at DESC").fetchall()
    now=datetime.now(timezone.utc)
    return [{**dict(r),"expired":bool(r["expires_at"] and datetime.fromisoformat(r["expires_at"])<=now)} for r in rows]

@app.post("/api/admin/licenses/{key}/toggle")
def toggle_license(key:str,request:LicenseToggle,_admin:None=Depends(require_admin)):
    with _connect() as c:cur=c.execute("UPDATE licenses SET is_active=? WHERE license_key=?",(int(request.is_active),key.upper()))
    if not cur.rowcount:raise HTTPException(404,"License not found")
    return {"license_key":key.upper(),"is_active":request.is_active}

@app.post("/api/admin/licenses/{key}/reset-device")
def reset_license_device(key:str,_admin:None=Depends(require_admin)):
    with _connect() as c:cur=c.execute("UPDATE licenses SET bound_device_id=NULL WHERE license_key=?",(key.upper(),))
    if not cur.rowcount:raise HTTPException(404,"License not found")
    return {"license_key":key.upper(),"device_reset":True}

@app.get("/api/accounts")
def accounts(auth:LicenseContext=Depends(require_dashboard_license)):
    with _connect() as c:
        r=c.execute("SELECT * FROM accounts WHERE account_id=?",(auth.account_id,)).fetchone()
        return [dict(r)] if r else []

@app.post("/api/accounts")
def save_account(config:AccountConfig,auth:LicenseContext=Depends(require_dashboard_license)):
    if config.account_id!=auth.account_id:raise HTTPException(403,"Account does not match the activated license")
    init_db(); values=config.model_dump()
    with _connect() as c:c.execute("""INSERT INTO accounts(account_id,broker_server,account_type,risk_pct,max_daily_dd_pct,max_total_dd_pct,active,daily_start_balance) VALUES(:account_id,:broker_server,:account_type,:risk_pct,:max_daily_dd_pct,:max_total_dd_pct,:active,:daily_start_balance) ON CONFLICT(account_id) DO UPDATE SET broker_server=excluded.broker_server,account_type=excluded.account_type,risk_pct=excluded.risk_pct,max_daily_dd_pct=excluded.max_daily_dd_pct,max_total_dd_pct=excluded.max_total_dd_pct,active=excluded.active,daily_start_balance=excluded.daily_start_balance,updated_at=CURRENT_TIMESTAMP""",values)
    return {"saved":True,"account_id":config.account_id}

@app.post("/api/accounts/{account_id}/toggle")
def toggle(account_id:str,active:bool,auth:LicenseContext=Depends(require_dashboard_license)):
    if account_id!=auth.account_id:raise HTTPException(403,"Account does not match the activated license")
    init_db()
    with _connect() as c:
        cur=c.execute("UPDATE accounts SET active=?,daily_start_balance=CASE WHEN ?=1 THEN COALESCE((SELECT balance FROM account_state WHERE account_id=?),daily_start_balance) ELSE daily_start_balance END,updated_at=CURRENT_TIMESTAMP WHERE account_id=?",(int(active),int(active),account_id,account_id))
        if not cur.rowcount:raise HTTPException(404,"Account not found")
    return {"account_id":account_id,"active":active}

@app.get("/api/accounts/{account_id}/status")
def account_status(account_id:str,auth:LicenseContext=Depends(require_dashboard_license)):
    if account_id!=auth.account_id:raise HTTPException(403,"Account does not match the activated license")
    with _connect() as c:
        a=c.execute("SELECT * FROM accounts WHERE account_id=?",(account_id,)).fetchone();s=c.execute("SELECT * FROM account_state WHERE account_id=?",(account_id,)).fetchone()
    if not a:raise HTTPException(404,"Account not found")
    return {"account":dict(a),"state":dict(s) if s else {"balance":0,"equity":0}}

@app.get("/api/logs/{account_id}")
def logs(account_id:str,limit:int=Query(50,ge=1,le=200),auth:LicenseContext=Depends(require_dashboard_license)):
    if account_id!=auth.account_id:raise HTTPException(403,"Account does not match the activated license")
    with _connect() as c:return [dict(r) for r in c.execute("SELECT * FROM trade_logs WHERE account_id=? ORDER BY id DESC LIMIT ?",(account_id,limit))]

_demo_orb_cache:dict[tuple[str,str],tuple[int,dict[str,Any]]]={}


def evaluate_demo_orb(account_id:str,symbol:str,account:Any)->dict[str,Any]:
    """Dedicated opt-in path. The live approval registry is never consulted here."""
    if os.getenv("ENABLE_DEMO_ROLLING_ORB")!="1" or symbol!=demo_orb.SYMBOL:
        return _wait_signal("Demo Rolling ORB is disabled or the symbol is unsupported")
    if account["account_type"]!="broker" or account["risk_pct"]!=demo_orb.RISK_PCT:
        return _wait_signal("Demo Rolling ORB requires a broker account profile with exactly 0.25% risk")
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return _wait_signal("MT5 integration is unavailable; broker constraints cannot be queried offline")
    terminal_path=os.getenv("DEMO_MT5_TERMINAL_PATH")
    if not (mt5.initialize(terminal_path) if terminal_path else mt5.initialize()):
        return _wait_signal("Cannot connect to MT5 to verify demo mode and broker constraints")
    try:
        connected,info,tick=demo_orb.demo_broker_snapshot(mt5,account_id,account["broker_server"])
        rates=mt5.copy_rates_from_pos(symbol,mt5.TIMEFRAME_M15,1,500)
        context=demo_orb.rolling_orb_context(rates,info,tick)
        if not 900 <= time.time()-context["bar_time"] < 1800:
            return _wait_signal("Stale or incomplete demo M15 bar")
        lots=demo_orb.check_demo_volume(mt5,connected,info,context)
    except demo_orb.DemoUnavailable as exc:
        return _wait_signal(str(exc))
    finally:
        mt5.shutdown()

    bar=context["bar_time"]
    cache_key=(account_id,account["broker_server"])
    cached=_demo_orb_cache.get(cache_key)
    if cached and cached[0]==bar:
        result=cached[1]
        return {**result,"demo_preflight_lots":lots} if result["command"]!="WAIT" else result
    signal=evaluate_llm_signal(context)
    if signal["command"]=="WAIT":
        result=_wait_signal(signal["reason"])
    elif (signal["sl"],signal["tp"])!=(context["setup"]["stop"],context["setup"]["target"]):
        result=_wait_signal("Demo Rolling ORB levels must match the deterministic setup exactly")
    else:
        result={**signal,"decision":signal["command"],"entry":context["setup"]["entry_reference"],
                "symbol":symbol,"strategy":context["strategy"],"setup":context["setup"],"bar_time":bar,
                "signal_id":f"{symbol}-{bar}-demo-orb","execution_profile":demo_orb.PROFILE,
                "demo_preflight_lots":lots}
    _demo_orb_cache[cache_key]=(bar,result)
    return result


def evaluate_demo_gold(account_id, symbol, account):
    if os.getenv("ENABLE_DEMO_XAU_NY_BREAKOUT") != "1" or symbol != gold.SYMBOL:
        return _wait_signal("Demo XAUUSD breakout is disabled or symbol is unsupported")
    risk = account["risk_pct"]
    if account["account_type"] != "broker" or not .25 <= risk <= .50:
        return _wait_signal("Demo XAUUSD requires a broker profile with 0.25%-0.50% risk")
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return _wait_signal("MT5 integration is unavailable")
    path = os.getenv("DEMO_MT5_TERMINAL_PATH")
    if not (mt5.initialize(path) if path else mt5.initialize()):
        return _wait_signal("Cannot connect to MT5 demo for XAUUSD preflight")
    try:
        connected, info, tick = demo_orb.demo_broker_snapshot(mt5, account_id, account["broker_server"], symbol)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, 500)
        context = gold.broker_context(rates, info, tick, time.time())
        lots = demo_orb.check_demo_volume(mt5, connected, info, context, risk)
    except demo_orb.DemoUnavailable as exc:
        return _wait_signal(str(exc))
    finally:
        mt5.shutdown()
    setup = context["setup"]
    side, entry = context["side"], setup["entry_reference"]
    signal = validate_signal(dict(command=side, sl=setup["stop"], tp=setup["target"],
                                  confidence=1., setup=context["strategy"], reason="Completed deterministic breakout"))
    if not _valid_signal_risk(signal, entry, side):
        return _wait_signal("XAUUSD breakout failed signal risk validation")
    bar = context["bar_time"]
    return {**signal, "decision":side, "entry":entry, "symbol":symbol, "bar_time":bar,
            "signal_id":f"XAUUSD-{bar//86400}-demo-ny", "execution_profile":gold.PROFILE,
            "demo_preflight_lots":lots}


@app.get("/api/ea/poll/{account_id}/{symbol}")
def ea_poll(account_id:str,symbol:str,balance:float=Header(default=0,alias="X-Account-Balance"),equity:float=Header(default=0,alias="X-Account-Equity"),x_license_key:str=Header(default=""),x_device_id:str=Header(default=""),demo_orb_profile:bool=False,demo_xau_ny_breakout:bool=False):
    init_db()
    if not _ea_authorized(x_license_key,account_id,x_device_id):return {"command":"UNAUTHORIZED"}
    with _connect() as c:
        a=c.execute("SELECT * FROM accounts WHERE account_id=?",(account_id,)).fetchone()
        if not a:raise HTTPException(404,"Account not configured")
        c.execute("INSERT INTO account_state(account_id,balance,equity) VALUES(?,?,?) ON CONFLICT(account_id) DO UPDATE SET balance=excluded.balance,equity=excluded.equity,updated_at=CURRENT_TIMESTAMP",(account_id,balance,equity))
    if not a["active"]:return {**_wait_signal("Account stopped"),"risk_pct":a["risk_pct"],"max_daily_dd_pct":a["max_daily_dd_pct"],"max_total_dd_pct":a["max_total_dd_pct"]}
    with _scanner_lock:
        try:
            if demo_xau_ny_breakout:
                result = _wait_signal("Conflicting demo profiles") if demo_orb_profile else evaluate_demo_gold(account_id,symbol,a)
            else:
                result=evaluate_demo_orb(account_id,symbol,a) if demo_orb_profile else analyze_symbol(symbol,account_id)
        except Exception:LOG.exception("analyze_symbol failed for %s",symbol);result=None
    if not result or result["decision"] not in ("BUY","SELL"):return {**_wait_signal(result["reason"] if result else "No qualified setup"),"risk_pct":a["risk_pct"],"max_daily_dd_pct":a["max_daily_dd_pct"],"max_total_dd_pct":a["max_total_dd_pct"]}
    result=dict(result)  # never mutate the per-symbol cache shared across accounts/threads
    result.update({"execution_delay_seconds":random.randint(3,15) if a["account_type"]=="prop" else 0,"tp_offset_points":random.uniform(1,3) if a["account_type"]=="prop" else 0,"risk_pct":a["risk_pct"],"max_daily_dd_pct":a["max_daily_dd_pct"],"max_total_dd_pct":a["max_total_dd_pct"]})
    return result

def scan_loop():
    """Optional dashboard feed scanner; EA polling is an independent interface."""
    while True:
        try:
            with _connect() as c:ids=[r[0] for r in c.execute("SELECT account_id FROM accounts WHERE active=1")]
            for aid in ids:
                for sym in SYMBOLS:analyze_symbol(sym,aid)
        except Exception:LOG.exception("Scanner loop error")
        time.sleep(max(15,int(os.getenv("SCAN_INTERVAL_SECONDS","60"))))

if __name__=="__main__":
    import uvicorn
    init_db()
    if os.getenv("ENABLE_BACKGROUND_SCANNER","0")=="1":threading.Thread(target=scan_loop,daemon=True,name="market-scanner").start()
    uvicorn.run(app,host=os.getenv("HOST","127.0.0.1"),port=int(os.getenv("PORT","8000")),access_log=False)
