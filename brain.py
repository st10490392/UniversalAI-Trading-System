"""UniversalAI: broker-independent quant/SMC engine and FastAPI service.

MT5 and Groq are optional adapters. This service never places orders; execution
and a second drawdown circuit breaker belong to the EA. See README.md for contracts.
"""
from __future__ import annotations
import json, logging, os, random, secrets, sqlite3, threading, time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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

ROOT=Path(__file__).resolve().parent
load_dotenv(ROOT/".env",override=False)
DB_PATH=Path(os.getenv("UNIVERSALAI_DB",ROOT/"accounts.db"))
SYMBOLS=[s.strip() for s in os.getenv("MT5_SYMBOLS","EURUSD,XAUUSD,Volatility 75 Index,Boom 1000 Index,Crash 1000 Index").split(",") if s.strip()]
MODEL=os.getenv("GROQ_MODEL","qwen/qwen3-32b")
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

def _valid_license(license_key:str,account_id:str,device_id:str|None=None,bind:bool=False)->bool:
    key=license_key.strip().upper()
    if not key or not account_id:return False
    with _connect() as c:
        if bind:c.execute("BEGIN IMMEDIATE")
        row=c.execute("SELECT * FROM licenses WHERE license_key=?",(key,)).fetchone()
        if not row or not row["is_active"]:return False
        if row["expires_at"] and datetime.fromisoformat(row["expires_at"])<=datetime.now(timezone.utc):return False
        if row["bound_account_id"] and row["bound_account_id"]!=account_id:return False
        if device_id is not None:
            if row["bound_device_id"] and row["bound_device_id"]!=device_id:return False
            if bind:
                c.execute("UPDATE licenses SET bound_account_id=COALESCE(bound_account_id,?),bound_device_id=COALESCE(bound_device_id,?) WHERE license_key=?",(account_id,device_id,key))
        elif not row["bound_account_id"] or not row["bound_device_id"]:
            return False
        return True

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

def groq_decision(context:dict[str,Any])->dict[str,Any]:
    key=os.getenv("GROQ_API_KEY","").strip()
    if not key:raise RuntimeError("GROQ_API_KEY is not configured")
    schema={"decision":"BUY|SELL|WAIT","entry":"number","sl":"number","tp":"number","confidence":"number","reason":"string"}
    prompt="Return only a JSON object matching "+json.dumps(schema)+". Do not change setup side; uncertain means WAIT. Context: "+json.dumps(context)
    r=requests.post("https://api.groq.com/openai/v1/chat/completions",timeout=25,headers={"Authorization":"Bearer "+key},json={"model":MODEL,"messages":[{"role":"user","content":prompt}],"temperature":.1,"response_format":{"type":"json_object"}})
    r.raise_for_status(); out=json.loads(r.json()["choices"][0]["message"]["content"])
    if out.get("decision") not in ("BUY","SELL","WAIT"):raise ValueError("Invalid model decision")
    for k in ("entry","sl","tp","confidence"):out[k]=float(out[k])
    out["reason"]=str(out.get("reason",""))[:1000]; return out

def validate_decision(result:dict[str,Any],side:str)->bool:
    if result.get("decision")!=side or not .75<=result.get("confidence",0)<=1:return False
    e,sl,tp=result["entry"],result["sl"],result["tp"]
    directional=(side=="BUY" and sl<e<tp) or (side=="SELL" and tp<e<sl)
    return bool(directional and abs(e-sl)>0 and abs(tp-e)/abs(e-sl)>=2.5)

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
    records=[r for r in approval_doc.get("approved",[]) if r.get("symbol")==symbol]
    strategy_names=tuple(sorted({str(r.get("strategy")) for r in records if r.get("strategy") in RESEARCH_STRATEGIES.values()}))
    # Fail closed until this exact symbol has passed the OOS/Monte Carlo gate.
    if not strategy_names:return None
    m15=_mt5_rates(symbol,"M15")
    if m15 is None:return None
    bar_time=int(m15.time.iloc[-1]) if "time" in m15.columns else int(time.time()//900)
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
        context={"symbol":symbol,"strategy":candidate["strategy"],"setup":candidate["setup"],"quant_metrics":candidate["quant"],"asset_class":m15["asset_class"].iloc[-1],"last_completed_bar_utc":str(m15.time.iloc[-1])}
        try:
            proposed=groq_decision(context)
        except Exception as exc:
            LOG.exception("Decision generation failed for %s / %s",symbol,candidate["strategy"])
            proposed={"decision":"WAIT","entry":0.,"sl":0.,"tp":0.,"confidence":0.,"reason":"Decision unavailable: "+type(exc).__name__}
        if not validate_decision(proposed,side):proposed["decision"]="WAIT"
        proposed.update({"symbol":symbol,"strategy":candidate["strategy"],"breakeven_at_r":1.2 if candidate["strategy"]==RESEARCH_STRATEGIES["A"] else 0.0,"bias":side,"quant":candidate["quant"],"setup":candidate["setup"],"signal_id":f"{symbol}-{bar_time}-{candidate['strategy'].replace(' ','_')}","reason":proposed.get("reason","")})
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

@app.get("/api/ea/poll/{account_id}/{symbol}")
def ea_poll(account_id:str,symbol:str,balance:float=Header(default=0,alias="X-Account-Balance"),equity:float=Header(default=0,alias="X-Account-Equity"),x_license_key:str=Header(default="")):
    init_db()
    if not _valid_license(x_license_key,account_id):return {"command":"UNAUTHORIZED"}
    with _connect() as c:
        a=c.execute("SELECT * FROM accounts WHERE account_id=?",(account_id,)).fetchone()
        if not a:raise HTTPException(404,"Account not configured")
        c.execute("INSERT INTO account_state(account_id,balance,equity) VALUES(?,?,?) ON CONFLICT(account_id) DO UPDATE SET balance=excluded.balance,equity=excluded.equity,updated_at=CURRENT_TIMESTAMP",(account_id,balance,equity))
    if not a["active"]:return {"decision":"WAIT","reason":"Account stopped","risk_pct":a["risk_pct"],"max_daily_dd_pct":a["max_daily_dd_pct"],"max_total_dd_pct":a["max_total_dd_pct"]}
    with _scanner_lock:result=analyze_symbol(symbol,account_id)
    if not result or result["decision"] not in ("BUY","SELL"):return {"decision":"WAIT","reason":"No qualified setup","risk_pct":a["risk_pct"],"max_daily_dd_pct":a["max_daily_dd_pct"],"max_total_dd_pct":a["max_total_dd_pct"]}
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
