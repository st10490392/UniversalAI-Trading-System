"""UniversalAI: broker-independent quant/SMC engine and FastAPI service.

MT5 and Groq are optional adapters. This service never places orders; execution
and a second drawdown circuit breaker belong to the EA. See README.md for contracts.
"""
from __future__ import annotations
import json, logging, os, random, sqlite3, threading, time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parent
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
_analysis_cache:dict[str,tuple[int,dict[str,Any]|None]]={}

@app.middleware("http")
async def dashboard_pin(request:Request,call_next):
    if request.url.path.startswith("/api/") and not request.url.path.startswith("/api/ea/poll/") and request.url.path!="/api/health":
        pin=os.getenv("DASHBOARD_PIN","")
        if not pin:return JSONResponse({"detail":"DASHBOARD_PIN is not configured"},status_code=503)
        if request.headers.get("x-dashboard-pin","")!=pin:return JSONResponse({"detail":"Invalid dashboard PIN"},status_code=401)
    return await call_next(request)

def _connect():
    c=sqlite3.connect(DB_PATH,timeout=10); c.row_factory=sqlite3.Row; c.execute("PRAGMA journal_mode=WAL"); return c

def init_db():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    with _connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS accounts(account_id TEXT PRIMARY KEY,broker_server TEXT NOT NULL,account_type TEXT NOT NULL CHECK(account_type IN ('prop','broker')),risk_pct REAL NOT NULL CHECK(risk_pct>0 AND risk_pct<=5),max_daily_dd_pct REAL NOT NULL CHECK(max_daily_dd_pct>0 AND max_daily_dd_pct<=100),max_total_dd_pct REAL NOT NULL CHECK(max_total_dd_pct>0 AND max_total_dd_pct<=100),active INTEGER NOT NULL DEFAULT 0,daily_start_balance REAL NOT NULL DEFAULT 0,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS trade_logs(id INTEGER PRIMARY KEY AUTOINCREMENT,account_id TEXT NOT NULL,symbol TEXT NOT NULL,signal_id TEXT NOT NULL,decision TEXT NOT NULL,confidence REAL NOT NULL DEFAULT 0,reason TEXT NOT NULL DEFAULT '',signal_json TEXT NOT NULL DEFAULT '{}',created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,UNIQUE(account_id,signal_id),FOREIGN KEY(account_id) REFERENCES accounts(account_id));
        CREATE TABLE IF NOT EXISTS account_state(account_id TEXT PRIMARY KEY,balance REAL NOT NULL DEFAULT 0,equity REAL NOT NULL DEFAULT 0,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
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

def _frame(data:Any)->pd.DataFrame:
    d=data.copy() if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    aliases={"tick_volume":"volume","real_volume":"volume"}
    d=d.rename(columns={k:v for k,v in aliases.items() if k in d and v not in d})
    if not {"open","high","low","close"}.issubset(d.columns): raise ValueError("OHLC columns required")
    return d.reset_index(drop=True)

def quant_metrics(data:Any)->dict[str,Any]:
    """ATR14/ATR50, 50-close z-score, and Kaufman ER20, all on completed bars."""
    d=_frame(data); h,l,c=d.high.astype(float),d.low.astype(float),d.close.astype(float); prev=c.shift(1)
    tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    a14=tr.rolling(14,min_periods=14).mean(); a50=tr.rolling(50,min_periods=50).mean()
    mean=c.rolling(50,min_periods=50).mean(); std=c.rolling(50,min_periods=50).std(ddof=0)
    er=c.diff(20).abs()/c.diff().abs().rolling(20,min_periods=20).sum().replace(0,np.nan)
    vals=[a14.iloc[-1],a50.iloc[-1],mean.iloc[-1],std.iloc[-1],er.iloc[-1]]
    if any(pd.isna(v) for v in vals) or a50.iloc[-1]<=0: return {"ready":False,"atr14":0.,"atr50":0.,"atr_ratio":0.,"zscore":0.,"efficiency_ratio":0.,"gate":False}
    ratio=float(a14.iloc[-1]/a50.iloc[-1]); z=float((c.iloc[-1]-mean.iloc[-1])/std.iloc[-1]) if std.iloc[-1]>0 else 0.; efficiency=float(er.iloc[-1])
    return {"ready":True,"atr14":float(a14.iloc[-1]),"atr50":float(a50.iloc[-1]),"atr_ratio":ratio,"zscore":z,"efficiency_ratio":efficiency,"gate":bool(ratio>.90 and efficiency>.25)}

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
    s=symbol.lower(); return any(x in s for x in ("volatility","boom","crash","step index","jump","synthetic"))

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
    m15,h1=_mt5_rates(symbol,"M15"),_mt5_rates(symbol,"H1")
    if m15 is None or h1 is None:return None
    bar_time=int(m15.time.iloc[-1]) if "time" in m15.columns else int(time.time()//900)
    cached=_analysis_cache.get(symbol)
    if cached and cached[0]==bar_time:
        result=cached[1]
        if result and account_id and result.get("decision") in ("BUY","SELL"):
            with _connect() as c:c.execute("INSERT OR IGNORE INTO trade_logs(account_id,symbol,signal_id,decision,confidence,reason,signal_json) VALUES(?,?,?,?,?,?,?)",(account_id,symbol,result["signal_id"],result["decision"],result["confidence"],result["reason"],json.dumps(result)))
        return result
    bias=market_bias(h1); setup=detect_smc(m15,bias)
    if not setup or (not is_synthetic(symbol) and not in_kill_zone(now)):
        _analysis_cache[symbol]=(bar_time,None);return None
    metrics=quant_metrics(m15)
    if not quant_allows(metrics,setup["side"]):_analysis_cache[symbol]=(bar_time,None);return None
    try:decision=groq_decision({"symbol":symbol,"bias":bias,"setup":setup,"quant":metrics,"last_close":float(m15.close.iloc[-1])})
    except Exception as exc:
        LOG.exception("Decision generation failed for %s",symbol);decision={"decision":"WAIT","entry":0.,"sl":0.,"tp":0.,"confidence":0.,"reason":"Decision unavailable: "+type(exc).__name__}
    if not validate_decision(decision,setup["side"]):decision["decision"]="WAIT"
    decision.update({"symbol":symbol,"bias":bias,"quant":metrics,"setup":setup,"signal_id":f"{symbol}-{bar_time}"})
    _analysis_cache[symbol]=(bar_time,decision)
    if account_id:
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

@app.get("/api/accounts")
def accounts():
    with _connect() as c:return [dict(r) for r in c.execute("SELECT * FROM accounts ORDER BY account_id")]

@app.post("/api/accounts")
def save_account(config:AccountConfig):
    init_db(); values=config.model_dump()
    with _connect() as c:c.execute("""INSERT INTO accounts(account_id,broker_server,account_type,risk_pct,max_daily_dd_pct,max_total_dd_pct,active,daily_start_balance) VALUES(:account_id,:broker_server,:account_type,:risk_pct,:max_daily_dd_pct,:max_total_dd_pct,:active,:daily_start_balance) ON CONFLICT(account_id) DO UPDATE SET broker_server=excluded.broker_server,account_type=excluded.account_type,risk_pct=excluded.risk_pct,max_daily_dd_pct=excluded.max_daily_dd_pct,max_total_dd_pct=excluded.max_total_dd_pct,active=excluded.active,daily_start_balance=excluded.daily_start_balance,updated_at=CURRENT_TIMESTAMP""",values)
    return {"saved":True,"account_id":config.account_id}

@app.post("/api/accounts/{account_id}/toggle")
def toggle(account_id:str,active:bool):
    init_db()
    with _connect() as c:
        cur=c.execute("UPDATE accounts SET active=?,daily_start_balance=CASE WHEN ?=1 THEN COALESCE((SELECT balance FROM account_state WHERE account_id=?),daily_start_balance) ELSE daily_start_balance END,updated_at=CURRENT_TIMESTAMP WHERE account_id=?",(int(active),int(active),account_id,account_id))
        if not cur.rowcount:raise HTTPException(404,"Account not found")
    return {"account_id":account_id,"active":active}

@app.get("/api/accounts/{account_id}/status")
def account_status(account_id:str):
    with _connect() as c:
        a=c.execute("SELECT * FROM accounts WHERE account_id=?",(account_id,)).fetchone();s=c.execute("SELECT * FROM account_state WHERE account_id=?",(account_id,)).fetchone()
    if not a:raise HTTPException(404,"Account not found")
    return {"account":dict(a),"state":dict(s) if s else {"balance":0,"equity":0}}

@app.get("/api/logs/{account_id}")
def logs(account_id:str,limit:int=Query(50,ge=1,le=200)):
    with _connect() as c:return [dict(r) for r in c.execute("SELECT * FROM trade_logs WHERE account_id=? ORDER BY id DESC LIMIT ?",(account_id,limit))]

@app.get("/api/ea/poll/{account_id}/{symbol}")
def ea_poll(account_id:str,symbol:str,balance:float=0,equity:float=0):
    init_db()
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
    uvicorn.run(app,host=os.getenv("HOST","127.0.0.1"),port=int(os.getenv("PORT","8000")))
