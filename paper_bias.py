"""Closed-bar market direction and optional model commentary for the mobile feed."""
import json
import os
import time

import pandas as pd
import requests


def describe(frames,now):
    trends={}
    for name,period,seconds in (("H4",50,14400),("D1",20,86400)):
        f=frames[name]
        f=f[f.time+seconds<=now]
        if len(f)<period or not f.time.is_monotonic_increasing or f.time.duplicated().any():
            raise ValueError("Insufficient closed trend bars")
        c=float(f.close.iloc[-1]); ema=float(f.close.ewm(span=period,adjust=False).mean().iloc[-1])
        if not 0<c<float("inf") or not 0<ema<float("inf"): raise ValueError("Invalid trend prices")
        if now-f.time.iloc[-1]-seconds>seconds+3*86400: raise ValueError("Stale trend bars")
        trends[name]=dict(close=c,ema=ema,closed_at=int(f.time.iloc[-1])+seconds,
                          direction="BULLISH" if c>ema else "BEARISH" if c<ema else "NEUTRAL")
    direction=trends["H4"]["direction"] if trends["D1"]["direction"]==trends["H4"]["direction"] else "NEUTRAL"
    bars=frames["M15"]
    bars=bars[bars.time+900<=now].copy()
    if bars.empty or now-bars.time.iloc[-1]>=1800: raise ValueError("Stale intraday history")
    stamp=pd.to_datetime(bars.time,unit="s",utc=True)
    today=pd.Timestamp(now,unit="s",tz="UTC").date()
    levels={}
    for session,start,end in (("Asia",0,6),("London",7,13),("NY",13,17)):
        window=bars[(stamp.dt.date==today)&(stamp.dt.hour>=start)&(stamp.dt.hour<end)]
        if len(window):
            levels[session]=dict(high=float(window.high.max()),low=float(window.low.min()),bars=len(window))
    return dict(as_of=now,direction=direction,trends=trends,liquidity=levels,
                regime="aligned trend" if direction!="NEUTRAL" else "mixed trend",ai=None)


def commentary(snapshot):
    for prefix,url,model in (("GROQ","https://api.groq.com/openai/v1","openai/gpt-oss-20b"),
                             ("GEMINI","https://generativelanguage.googleapis.com/v1beta/openai","gemini-3.1-flash-lite")):
        key=os.getenv(prefix+"_API_KEY")
        if not key: continue
        body=dict(model=os.getenv(prefix+"_MODEL",model),response_format={"type":"json_object"},
                  messages=[dict(role="system",content="Describe observed market regime only. Return JSON with commentary (short text) and confidence (number 0-1, subjective)."),
                            dict(role="user",content=json.dumps(snapshot))])
        if prefix=="GROQ" and "gpt-oss" in body["model"]: body["reasoning_effort"]="low"
        try:
            r=requests.post(os.getenv(prefix+"_BASE_URL",url).rstrip("/")+"/chat/completions",json=body,
                            headers={"Authorization":"Bearer "+key},timeout=(3,8))
            r.raise_for_status()
            result=json.loads(r.json()["choices"][0]["message"]["content"])
            confidence=result["confidence"]; note=result["commentary"]
            if type(confidence) in (int,float) and 0<=confidence<=1 and isinstance(note,str) and note.strip():
                return dict(provider=prefix,commentary=note[:800],confidence=confidence)
        except (requests.RequestException,ValueError,KeyError,IndexError,TypeError): continue
    return None
