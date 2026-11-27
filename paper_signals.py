"""Compute approved entries and bar-close stop updates from completed broker bars."""
import math
import pandas as pd
from four_strategy_playbook import opening, pullback
from research_lab import add_indicators


def closed(frame,seconds,now):
    f=frame.loc[frame.time+seconds<=now].copy()
    if f.empty or f.time.duplicated().any() or not f.time.is_monotonic_increasing:
        raise ValueError("Missing or unordered completed bars")
    f.index=pd.to_datetime(f.time,unit="s",utc=True)
    return f


def decisions(frames,symbol,setups,now):
    """Return currently actionable signals; each timestamp is the signal close."""
    h=closed(frames["H1"],3600,now)
    if len(h)<200: return [],{}
    h=add_indicators(h)
    h.index+=pd.Timedelta(hours=1)
    h["hour_ema"]=h.close.ewm(span=200,adjust=False,min_periods=200).mean()
    clock=pd.Timestamp(now,unit="s",tz="UTC")
    start=clock.normalize()+pd.Timedelta(hours=7)
    ny=pd.Timestamp(str(clock.date())+" 09:30",tz="America/New_York").tz_convert("UTC")
    if symbol=="XAUUSD" and clock>=ny: start=ny
    candidates=[]; last={}
    for tf,seconds in (("M5",300),("M15",900)):
        f=closed(frames[tf],seconds,now)
        end=f.index[-1]+pd.Timedelta(seconds=seconds)
        if now-end.timestamp()>=seconds: continue
        f=add_indicators(f)
        last[tf]=(end.timestamp(),float(f.close.iloc[-1]),float(f.atr14.iloc[-1]),float(f.high.iloc[-1]),float(f.low.iloc[-1]))
        if now-end.timestamp()>30: continue
        ctimes=f.index+pd.Timedelta(seconds=seconds)
        f["hour_atr"]=h.atr14.reindex(ctimes,method="ffill").to_numpy()
        f["hour_ema"]=h.hour_ema.reindex(ctimes,method="ffill").to_numpy()
        g=f.loc[start:]
        if g.empty or g.index[0]!=start or (g.index.to_series().diff().dropna()!=pd.Timedelta(seconds=seconds)).any(): continue
        for setup in setups:
            if setup["symbol"]!=symbol or not setup["enabled"]: continue
            profile=setup["profile"]
            if tf=="M5" and setup["strategy"]=="Session ORB":
                n=int(profile.split("/")[1].rstrip("m"))//5
                signals=opening(g,n) if len(g)>n else []
            elif tf=="M15" and setup["strategy"]=="EMA VWAP":
                signals=pullback(g)
            else: continue
            for index,side,stop,_,_ in signals:
                if index==g.index[-1] and math.isfinite(stop):
                    candidates.append((setup,int(side),float(stop),int(end.timestamp())))
    return candidates,last
