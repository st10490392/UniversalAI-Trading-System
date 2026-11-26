"""Multi-strategy MT5 M15 research runner with chronological OOS evaluation.

Run: python research_lab.py [--data-dir data/history_2020_2025]
All entry decisions use completed bars and execute from the following bar's open.
The XGBoost model is fitted only on labelled 2020-2023 breakouts and predicts 2024-2025.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DEFAULT = ROOT / "data" / "history_2020_2025"
OUT_DEFAULT = ROOT / "data" / "research_2020_2025"
APPROVED_PATH = ROOT / "approved_strategies.json"
FEATURES = ("atr_ratio", "zscore", "rsi", "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "hour", "volatility_regime")
STRATEGIES = {
    "A": "Trend Momentum",
    "B": "ICT / SMC Session Engine",
    "C": "Quant Statistical Mean Reversion",
    "D": "Volatility Breakout + XGBoost Meta-Filter",
}


def _asset(symbol: str, tag: str = "") -> str:
    if tag in {"forex", "index", "synthetic"}:
        return tag
    compact = re.sub(r"[^a-z0-9]", "", symbol.lower())
    if any(k in compact for k in ("volatility", "boom", "crash", "stepindex", "skewstep", "jump", "rangebreak")):
        return "synthetic"
    return "index" if any(k in compact for k in ("us100", "nas100", "ustec", "nasdaq", "us30", "dj30", "wallstreet30", "ger40", "de40", "ger30", "de30", "germany", "dax")) else "forex"


def load_history(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    required = {"time", "open", "high", "low", "close"}
    if missing := required - set(d.columns):
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    d["time"] = pd.to_datetime(d.time, utc=True, errors="raise")
    for col in ("open", "high", "low", "close"):
        d[col] = pd.to_numeric(d[col], errors="raise")
    defaults = {"tick_volume": 1, "spread_points": 0, "point_size": 0.00001, "tick_size": 0.00001, "tick_value": 0.0, "asset_class": ""}
    for col, value in defaults.items():
        if col not in d:
            d[col] = value
    d["tick_volume"] = pd.to_numeric(d.tick_volume, errors="coerce").fillna(1).clip(lower=1)
    for col in ("spread_points", "point_size", "tick_size", "tick_value"):
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)
    d = d.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if len(d) < 300:
        raise ValueError("At least 300 M15 bars are needed for indicator warm-up")
    d.attrs["symbol"] = str(d.symbol.iloc[0]) if "symbol" in d and len(d) else Path(path).stem.split("_M15")[0]
    d.attrs["asset_class"] = _asset(d.attrs["symbol"], str(d.asset_class.iloc[0]) if len(d) else "")
    return d


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    if "volume" in d and "tick_volume" not in d:
        d["tick_volume"] = d.volume
    if "tick_volume" not in d:
        d["tick_volume"] = 1.0
    for col, value in (("point_size", 0.00001), ("spread_points", 0.0), ("tick_size", 0.00001), ("tick_value", 0.0)):
        if col not in d:
            d[col] = d["point"] if col == "point_size" and "point" in d else value
    if not pd.api.types.is_datetime64_any_dtype(d.time):
        d["time"] = pd.to_datetime(d.time, unit="s", utc=True)
    elif d.time.dt.tz is None:
        d["time"] = d.time.dt.tz_localize("UTC")
    c, h, l = (d[x].astype(float) for x in ("close", "high", "low"))
    prev = c.shift(1)
    tr = pd.concat((h - l, (h - prev).abs(), (l - prev).abs()), axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14, min_periods=14).mean()
    d["atr20"] = tr.rolling(20, min_periods=20).mean()
    d["atr50"] = tr.rolling(50, min_periods=50).mean()
    d["atr_ratio"] = (d.atr14 / d.atr50.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    d["ema_fast"] = c.ewm(span=12, adjust=False).mean()
    d["ema_slow"] = c.ewm(span=26, adjust=False).mean()
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    d["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)
    mean50 = c.rolling(50, min_periods=50).mean()
    std50 = c.rolling(50, min_periods=50).std(ddof=0)
    d["zscore"] = ((c - mean50) / std50.replace(0, np.nan)).fillna(0)
    er_den = delta.abs().rolling(20, min_periods=20).sum()
    d["er20"] = (c.diff(20).abs() / er_den.replace(0, np.nan)).fillna(0)
    bb_mid = c.rolling(20, min_periods=20).mean()
    bb_std = c.rolling(20, min_periods=20).std(ddof=0)
    d["bb_upper"], d["bb_lower"] = bb_mid + 2*bb_std, bb_mid - 2*bb_std
    kc_mid = c.ewm(span=20, adjust=False).mean()
    d["kc_upper"], d["kc_lower"] = kc_mid + 1.5*d.atr20, kc_mid - 1.5*d.atr20
    d["squeeze"] = (d.bb_upper < d.kc_upper) & (d.bb_lower > d.kc_lower)
    typical = (h + l + c) / 3
    d["vwap"] = (typical*d.tick_volume).groupby(d.time.dt.floor("D")).cumsum() / d.tick_volume.groupby(d.time.dt.floor("D")).cumsum()
    span = (h-l).replace(0, np.nan)
    d["body_ratio"] = ((c-d.open.astype(float)).abs()/span).fillna(0).clip(0, 1)
    d["upper_wick_ratio"] = ((h-pd.concat((c,d.open.astype(float)),axis=1).max(axis=1))/span).fillna(0).clip(0, 1)
    d["lower_wick_ratio"] = ((pd.concat((c,d.open.astype(float)),axis=1).min(axis=1)-l)/span).fillna(0).clip(0, 1)
    d["volatility_regime"] = (d.atr_ratio > 1.2).astype(float)
    d["hour"] = d.time.dt.hour.astype(float)
    return d


def build_signals(data: pd.DataFrame, symbol: str, strategy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return side, stop distance in price, optional target price, setup code per bar."""
    d = data if "atr14" in data else add_indicators(data)
    n = len(d); side = np.zeros(n, dtype=np.int8); stop = np.full(n, np.nan); target = np.full(n, np.nan); code = np.full(n, "", dtype=object)
    c, h, l = d.close.to_numpy(float), d.high.to_numpy(float), d.low.to_numpy(float)
    atr = d.atr14.to_numpy(float); er = d.er20.to_numpy(float); bias_up = (d.ema20 > d.ema50).to_numpy(); bias_dn = (d.ema20 < d.ema50).to_numpy()
    if strategy == "A":
        fast, slow = d.ema_fast.to_numpy(float), d.ema_slow.to_numpy(float); rsi=d.rsi.to_numpy(float); ratio=d.atr_ratio.to_numpy(float)
        up=(fast>slow)&(np.roll(fast,1)<=np.roll(slow,1))&(rsi>=55)&(ratio>1.0)
        dn=(fast<slow)&(np.roll(fast,1)>=np.roll(slow,1))&(rsi<=45)&(ratio>1.0)
        side[up]=1; side[dn]=-1; stop[:]=atr*1.5; code[up|dn]="ema_cross_rsi_atr_expansion"
    elif strategy == "B":
        asset=_asset(symbol, str(d.asset_class.iloc[0]) if "asset_class" in d else "")
        prev_hi=pd.Series(h).rolling(20,min_periods=20).max().shift(1).to_numpy(); prev_lo=pd.Series(l).rolling(20,min_periods=20).min().shift(1).to_numpy()
        fvg_up=l>np.roll(h,2); fvg_dn=h<np.roll(l,2)
        sweep_lo=(l<prev_lo)&(c>prev_lo); sweep_hi=(h>prev_hi)&(c<prev_hi)
        sweep_lo_recent=pd.Series(sweep_lo).rolling(4,min_periods=1).max().to_numpy().astype(bool)
        sweep_hi_recent=pd.Series(sweep_hi).rolling(4,min_periods=1).max().to_numpy().astype(bool)
        if asset == "synthetic":
            compact=re.sub(r"[^a-z0-9]","",symbol.lower())
            if "rangebreak" in compact or "jump" in compact:
                up=(c>prev_hi)&(d.atr_ratio.to_numpy(float)>1.05)&bias_up
                dn=(c<prev_lo)&(d.atr_ratio.to_numpy(float)>1.05)&bias_dn
                code[up|dn]="20_bar_range_breakout_atr_bias"
            elif "boom" in compact:
                up=sweep_lo_recent&fvg_up&bias_up; dn=np.zeros(n,dtype=bool)
                code[up]="boom_spike_aligned_bullish_sweep_fvg"
            elif "crash" in compact:
                dn=sweep_hi_recent&fvg_dn&bias_dn; up=np.zeros(n,dtype=bool)
                code[dn]="crash_spike_aligned_bearish_sweep_fvg"
            else:
                up=sweep_lo_recent&fvg_up&bias_up; dn=sweep_hi_recent&fvg_dn&bias_dn
                code[up|dn]="synthetic_sweep_fvg_directional_bias"
            side[up]=1; side[dn]=-1; stop[:]=atr*1.5
            return side,stop,target,code
        # Session comparisons are calculated in IANA time zones, including DST.
        ny=d.time.dt.tz_convert("America/New_York"); london=d.time.dt.tz_convert("Europe/London")
        ny_minutes=(ny.dt.hour*60+ny.dt.minute).to_numpy(); lon_minutes=(london.dt.hour*60+london.dt.minute).to_numpy()
        ny_day=ny.dt.strftime("%Y-%m-%d")
        session=((ny_minutes>=420)&(ny_minutes<600))|((ny_minutes>=600)&(ny_minutes<660))|((ny_minutes>=900)&(ny_minutes<960))|((lon_minutes>=480)&(lon_minutes<660))
        asian=(ny.dt.hour<7).to_numpy()
        day_hi=pd.Series(np.where(asian,h,np.nan)).groupby(ny_day).transform("max").to_numpy()
        day_lo=pd.Series(np.where(asian,l,np.nan)).groupby(ny_day).transform("min").to_numpy()
        structure_hi=pd.Series(h).rolling(3,min_periods=3).max().shift(1).to_numpy(); structure_lo=pd.Series(l).rolling(3,min_periods=3).min().shift(1).to_numpy()
        judas_sell=session&(ny_minutes>=420)&(ny_minutes<600)&(h>day_hi)&(c<day_hi)&(c<structure_lo)
        judas_buy=session&(ny_minutes>=420)&(ny_minutes<600)&(l<day_lo)&(c>day_lo)&(c>structure_hi)
        # A three-candle imbalance formed on the preceding bar and retested now.
        prior_bull=(np.roll(l,1)>np.roll(h,3)); prior_bear=(np.roll(h,1)<np.roll(l,3))
        retest_bull=prior_bull&(l<=np.roll(l,1))&(h>=np.roll(h,3))
        retest_bear=prior_bear&(h>=np.roll(h,1))&(l<=np.roll(l,3))
        silver=((ny_minutes>=600)&(ny_minutes<660))|((ny_minutes>=900)&(ny_minutes<960))
        bull_sb=session&silver&retest_bull&bias_up; bear_sb=session&silver&retest_bear&bias_dn
        up=judas_buy|bull_sb; dn=judas_sell|bear_sb
        code[judas_buy]="judas_sweep_mss_long"; code[judas_sell]="judas_sweep_mss_short"
        code[bull_sb]="silver_bullet_fvg_retest_long"; code[bear_sb]="silver_bullet_fvg_retest_short"
        side[up]=1; side[dn]=-1; stop[:]=atr*1.5
    elif strategy == "C":
        sq=d["squeeze"].shift(1,fill_value=False).to_numpy(bool); z=d.zscore.to_numpy(float); er=d.er20.to_numpy(float)
        up=sq&(z < -2.0)&(er<0.25)&(d.close.to_numpy(float)<d.bb_lower.to_numpy(float))
        dn=sq&(z > 2.0)&(er<0.25)&(d.close.to_numpy(float)>d.bb_upper.to_numpy(float))
        side[up]=1; side[dn]=-1; stop[:]=atr*1.5
        target[up|dn]=d.vwap.to_numpy(float)[up|dn]
        code[up|dn]="bb_keltner_squeeze_zscore_er_mean_reversion"
    elif strategy == "D":
        hi=pd.Series(h).rolling(20,min_periods=20).max().shift(1).to_numpy(); lo=pd.Series(l).rolling(20,min_periods=20).min().shift(1).to_numpy()
        expansion=d.atr_ratio.to_numpy(float)>1.1
        up=(c>hi)&expansion; dn=(c<lo)&expansion
        side[up]=1; side[dn]=-1; stop[:]=atr*1.25; code[up|dn]="20_bar_volatility_breakout_ml_candidate"
    else:
        raise ValueError(f"Unknown strategy family: {strategy}")
    invalid=~np.isfinite(stop)|(stop<=0)
    side[invalid]=0
    return side,stop,target,code


def _trade_one(d: pd.DataFrame, i: int, side: int, stop_dist: float, target_price: float, *, atr_multiple: float, breakeven: bool, commission_per_lot: float, slippage_points: float, commission_points: float, end_idx: int, horizon: int=96) -> dict[str, Any] | None:
    if i+1 >= end_idx or not np.isfinite(stop_dist) or stop_dist<=0:
        return None
    entry_i=i+1; point=max(float(d.point_size.iloc[entry_i]),0.0); spread=max(float(d.spread_points.iloc[entry_i]),0.0)*point
    slip=max(slippage_points,0.0)*point; tick_size=max(float(d.tick_size.iloc[entry_i]),0.0); tick_value=max(float(d.tick_value.iloc[entry_i]),0.0)
    base_open=float(d.open.iloc[entry_i]); entry=base_open+spread+slip if side>0 else base_open-slip
    risk=stop_dist; stop=entry-risk if side>0 else entry+risk
    if np.isfinite(target_price):
        target=float(target_price)
        if (side>0 and target<=entry) or (side<0 and target>=entry):
            return None
    else:
        target=entry+side*risk*2.5
    if commission_per_lot>0 and tick_size>0 and tick_value>0:
        risk_cash=max(risk/tick_size*tick_value,1e-9); cost_r=commission_per_lot/risk_cash
    else:
        cost_r=max(commission_points,0.0)*point*2/risk
    out_price=None; outcome="timeout"; last=min(end_idx,entry_i+horizon)
    for j in range(entry_i,last):
        op=float(d.open.iloc[j]); hi=float(d.high.iloc[j]); lo=float(d.low.iloc[j]); spr=max(float(d.spread_points.iloc[j]),0.0)*max(float(d.point_size.iloc[j]),0.0)
        if side>0:
            stop_hit=lo<=stop
            tp_hit=hi>=target
            if stop_hit:
                out_price=min(stop,op) if op<stop else stop
                out_price-=max(slippage_points,0.0)*max(float(d.point_size.iloc[j]),0.0); outcome="loss" if out_price<entry else "breakeven"; break
            if tp_hit:
                out_price=max(target,op) if op>target else target
                out_price-=max(slippage_points,0.0)*max(float(d.point_size.iloc[j]),0.0); outcome="win"; break
            # Activate the BE stop after evaluating this bar; OHLC cannot reveal
            # whether the favourable extreme preceded an adverse retrace.
            if breakeven and hi>=entry+1.2*risk: stop=max(stop,entry)
        else:
            ask_hi=hi+spr; ask_lo=lo+spr
            stop_hit=ask_hi>=stop; tp_hit=ask_lo<=target
            if stop_hit:
                fill=max(stop,op+spr) if op+spr>stop else stop
                out_price=fill+max(slippage_points,0.0)*max(float(d.point_size.iloc[j]),0.0); outcome="loss" if out_price>entry else "breakeven"; break
            if tp_hit:
                fill=min(target,op+spr) if op+spr<target else target
                out_price=fill+max(slippage_points,0.0)*max(float(d.point_size.iloc[j]),0.0); outcome="win"; break
            if breakeven and ask_lo<=entry-1.2*risk: stop=min(stop,entry)
    if out_price is None:
        j=max(entry_i,last-1)
        out_price=float(d.close.iloc[j])-(max(float(d.spread_points.iloc[j]),0.0)*max(float(d.point_size.iloc[j]),0.0) if side<0 else 0.0)-side*max(slippage_points,0.0)*max(float(d.point_size.iloc[j]),0.0)
    r=side*(out_price-entry)/risk-cost_r
    return {"time":d.time.iloc[entry_i].isoformat(),"r":float(r),"outcome":outcome,"entry":float(entry),"exit":float(out_price),"bars_held":int(j-entry_i+1)}


def simulate(data: pd.DataFrame, signals: tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray], *, start: str, end: str, risk_pct: float=.75, commission_per_lot: float=7.0, slippage_points: float=2.0, commission_points: float=1.0, breakeven: bool=False, horizon: int=96) -> list[dict[str,Any]]:
    side, stops, targets, _ = signals; times=data.time
    start_ts=pd.Timestamp(start,tz="UTC"); end_ts=pd.Timestamp(end,tz="UTC")
    valid=np.flatnonzero((side!=0)&(times>=start_ts).to_numpy()&(times<end_ts).to_numpy())
    end_idx=int(np.searchsorted(times.astype("int64").to_numpy(),end_ts.value,side="left"))
    trades=[]; next_free=0
    for i in valid:
        if i<next_free: continue
        trade=_trade_one(data,int(i),int(side[i]),float(stops[i]),float(targets[i]),atr_multiple=1.5,breakeven=breakeven,commission_per_lot=commission_per_lot,slippage_points=slippage_points,commission_points=commission_points,end_idx=end_idx,horizon=horizon)
        if trade:
            trade["signal_time"]=times.iloc[i].isoformat(); trades.append(trade)
            next_free=min(end_idx,int(i)+int(trade["bars_held"])+1)
    return trades


def _stats(trades: list[dict[str,Any]], risk_pct: float=.75) -> dict[str,Any]:
    eq=peak=1.0; maxdd=0.0; day_start=1.0; day_key=None; stopped_days=set(); stopped=False; month_ends={}; executed=[]
    for t in trades:
        r=float(t["r"])
        dt=pd.Timestamp(t["time"]); day=dt.date(); month=dt.strftime("%Y-%m")
        if day!=day_key: day_key=day; day_start=eq
        if stopped or day in stopped_days: continue
        executed.append(r)
        eq*=max(0.0,1.0+r*risk_pct/100.0); peak=max(peak,eq); dd=(peak-eq)/peak*100; maxdd=max(maxdd,dd)
        if (day_start-eq)/max(day_start,1e-12)*100>=5: stopped_days.add(day)
        if dd>=10: stopped=True
        month_ends[month]=eq
    rs=np.asarray(executed,dtype=float)
    wins=rs[rs>0].sum(); losses=-rs[rs<0].sum(); pf=float(wins/losses) if losses>0 else (None if wins>0 else 0.0)
    vals=list(month_ends.values()); mrets=[]; prev=1.0
    for value in vals: mrets.append(value/prev-1);prev=value
    return {"trades":len(executed),"win_rate_pct":float((rs>0).mean()*100) if len(rs) else 0.0,"profit_factor":pf,"total_return_pct":(eq-1)*100,"max_drawdown_pct":maxdd,"average_monthly_return_pct":float(np.mean(mrets)*100) if mrets else 0.0,"sharpe_ratio":float(np.mean(mrets)/np.std(mrets,ddof=1)*math.sqrt(12)) if len(mrets)>1 and np.std(mrets,ddof=1)>0 else 0.0}


def _feature_matrix(d: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    return d.loc[indices,list(FEATURES)].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)


def _fit_xgb(d: pd.DataFrame, raw: tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray], *, start: str, cutoff: str, args: argparse.Namespace, model_path: Path|None=None) -> tuple[Any|None,int,str]:
    try:
        import xgboost as xgb
    except ImportError:
        return None,0,"xgboost is not installed"
    sides,stops,targets,_=raw; times=d.time
    start_ts=pd.Timestamp(start,tz="UTC");cut_ts=pd.Timestamp(cutoff,tz="UTC")
    candidates=np.flatnonzero((sides!=0)&(times>=start_ts).to_numpy()&(times<cut_ts).to_numpy())
    end_idx=int(np.searchsorted(times.astype("int64").to_numpy(),cut_ts.value,side="left"))
    xi=[]; yi=[]
    for i in candidates:
        label=_trade_one(d,int(i),int(sides[i]),float(stops[i]),float(targets[i]),atr_multiple=1.25,breakeven=False,commission_per_lot=args.commission_per_lot,slippage_points=args.slippage_points,commission_points=args.commission_points,end_idx=end_idx,horizon=96)
        if label is None: continue
        xi.append(_feature_matrix(d,np.asarray([i]))[0]);yi.append(int(label["r"]>0))
    if len(yi)<40 or len(set(yi))<2:
        return None,len(yi),"fewer than 40 labelled training trades or one outcome class"
    train=xgb.DMatrix(np.asarray(xi),label=np.asarray(yi),feature_names=list(FEATURES))
    model=xgb.train({"max_depth":3,"eta":.04,"subsample":.8,"colsample_bytree":.85,"min_child_weight":5,"lambda":2,"objective":"binary:logistic","eval_metric":"logloss","nthread":2,"seed":42},train,num_boost_round=160)
    if model_path:
        model_path.parent.mkdir(parents=True,exist_ok=True); model.save_model(str(model_path))
    return model,len(yi),"trained on 2020-2023 only"


def _monte_carlo(trades: list[dict[str,Any]], *, iterations: int=1000, risk_pct: float=.75, seed: int=42) -> dict[str,float]:
    if not trades: return {"iterations":iterations,"risk_pct":risk_pct,"p95_max_drawdown_pct":0.0,"probability_of_ruin_pct":0.0,"average_monthly_return_pct":0.0,"sharpe_ratio":0.0}
    rs=np.asarray([t["r"] for t in trades],dtype=float); dates=[pd.Timestamp(t["time"]) for t in trades]
    counts=list(Counter(d.date() for d in dates).values()); rng=np.random.default_rng(seed); dds=[]; ruined=0; returns=[]; all_monthly=[]
    for _ in range(iterations):
        shuffled=rng.permutation(rs); eq=peak=1.0; maxdd=0.0; cursor=0; daily_stopped=False; total_stopped=False; month_values={}; current_month=None; month_start=1.0
        for day_n in counts:
            day_start=eq; day_ruin=False
            for r in shuffled[cursor:cursor+day_n]:
                cursor+=1
                if day_ruin or total_stopped: continue
                eq*=max(0.0,1+r*risk_pct/100.0); peak=max(peak,eq); dd=(peak-eq)/peak*100; maxdd=max(maxdd,dd)
                if (day_start-eq)/max(day_start,1e-12)*100>=5: day_ruin=True
                if dd>=10: total_stopped=True
            if day_ruin or total_stopped: daily_stopped=True
            # Reuse original chronological month lengths for a stable monthly-return estimate.
            idx=min(cursor-1,len(dates)-1); month=dates[idx].strftime("%Y-%m")
            if current_month is None: current_month=month;month_start=eq
            if month!=current_month:
                month_values[current_month]=eq/max(month_start,1e-12)-1;current_month=month;month_start=eq
        if current_month is not None:month_values[current_month]=eq/max(month_start,1e-12)-1
        dds.append(maxdd); returns.append((eq-1)*100); all_monthly.extend(x*100 for x in month_values.values())
        if daily_stopped or total_stopped: ruined+=1
    m=np.asarray(all_monthly,dtype=float)
    return {"iterations":iterations,"risk_pct":risk_pct,"p95_max_drawdown_pct":float(np.percentile(dds,95)),"probability_of_ruin_pct":ruined/iterations*100,"average_monthly_return_pct":float(m.mean()) if len(m) else 0.0,"sharpe_ratio":float(m.mean()/m.std(ddof=1)*math.sqrt(12)) if len(m)>1 and m.std(ddof=1)>0 else 0.0,"median_total_return_pct":float(np.median(returns))}


def live_candidates(symbol: str, data: pd.DataFrame, strategies: list[str], model_dir: Path|None=None) -> list[dict[str,Any]]:
    """Produce current completed-bar setups only for the passed approved strategy names."""
    d=add_indicators(data); found=[]
    for key,name in STRATEGIES.items():
        if name not in strategies: continue
        sig,stops,targets,codes=build_signals(d,symbol,key); i=len(d)-1
        if not sig[i]: continue
        if key=="D":
            try:
                import xgboost as xgb
                model_path=(model_dir or OUT_DEFAULT/"models")/(re.sub(r"[^A-Za-z0-9]+","_",symbol).strip("_")+"_D.json")
                model=xgb.Booster();model.load_model(str(model_path));prob=float(model.predict(xgb.DMatrix(_feature_matrix(d,np.asarray([i])),feature_names=list(FEATURES)))[0])
            except Exception:
                continue
            if prob<.55: continue
        side=int(sig[i]); entry=float(d.close.iloc[i]); risk=float(stops[i]); target=float(targets[i]) if np.isfinite(targets[i]) else entry+side*risk*2.5
        if (side>0 and target<=entry) or (side<0 and target>=entry): target=entry+side*risk*2.5
        q={k:float(d[k].iloc[i]) if pd.notna(d[k].iloc[i]) else 0.0 for k in ("atr14","atr_ratio","zscore","er20","rsi")}
        found.append({"strategy":name,"side":"BUY" if side>0 else "SELL","setup":{"rule":str(codes[i]),"entry_reference":entry,"stop":entry-side*risk,"target":target,"risk_reward":abs(target-entry)/risk},"quant":q})
    return found


def run(args: argparse.Namespace) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    files=sorted(args.data_dir.glob("*_M15_2020_2025.csv")); all_results=[]; approved=[]
    if args.symbols:
        requested=set(args.symbols)
        files=[p for p in files if p.stem.removesuffix("_M15_2020_2025").replace("_"," ") in requested or ("symbol" in pd.read_csv(p,nrows=1).columns and str(pd.read_csv(p,nrows=1).symbol.iloc[0]) in requested)]
    if not files: raise FileNotFoundError(f"No *_M15_2020_2025.csv files found in {args.data_dir}")
    model_dir=args.output_dir/"models"
    for ix,path in enumerate(files,1):
        try: rawdata=load_history(path)
        except Exception as exc: print(f"SKIP {path.name}: {type(exc).__name__}: {exc}");continue
        symbol=rawdata.attrs["symbol"];asset=rawdata.attrs["asset_class"];d=add_indicators(rawdata)
        print(f"[{ix}/{len(files)}] {symbol} [{asset}] {len(d):,} bars {d.time.iloc[0].date()}..{d.time.iloc[-1].date()}",flush=True)
        for key,name in STRATEGIES.items():
            if key=="D":
                raw=build_signals(d,symbol,"D"); model,train_n,train_note=_fit_xgb(d,raw,start="2020-01-01",cutoff="2024-01-01",args=args,model_path=model_dir/(re.sub(r"[^A-Za-z0-9]+","_",symbol).strip("_")+"_D.json"))
                if model is None:
                    result={"symbol":symbol,"asset_class":asset,"strategy":name,"strategy_id":key,"is":_stats(simulate(d,raw,start="2020-01-01",end="2024-01-01",commission_per_lot=args.commission_per_lot,slippage_points=args.slippage_points,commission_points=args.commission_points)),"oos":{"trades":0,"win_rate_pct":0.0,"profit_factor":0.0,"total_return_pct":0.0,"max_drawdown_pct":0.0,"average_monthly_return_pct":0.0,"sharpe_ratio":0.0},"training_trades":train_n,"model_status":train_note,"monte_carlo":None}
                else:
                    candidates=np.flatnonzero((raw[0]!=0)&(d.time>=pd.Timestamp("2024-01-01",tz="UTC")).to_numpy()&(d.time<pd.Timestamp("2026-01-01",tz="UTC")).to_numpy())
                    import xgboost as xgb
                    probs=model.predict(xgb.DMatrix(_feature_matrix(d,candidates),feature_names=list(FEATURES))) if len(candidates) else np.asarray([])
                    filtered=[int(i) for i,p in zip(candidates,probs) if p>=args.xgb_threshold]
                    gated=tuple(arr.copy() for arr in raw)
                    mask=np.ones(len(d),dtype=bool);mask[filtered]=False;gated[0][mask]=0
                    is_trades=simulate(d,raw,start="2020-01-01",end="2024-01-01",commission_per_lot=args.commission_per_lot,slippage_points=args.slippage_points,commission_points=args.commission_points)
                    oos_trades=simulate(d,gated,start="2024-01-01",end="2026-01-01",commission_per_lot=args.commission_per_lot,slippage_points=args.slippage_points,commission_points=args.commission_points)
                    result={"symbol":symbol,"asset_class":asset,"strategy":name,"strategy_id":key,"is":_stats(is_trades),"oos":_stats(oos_trades),"training_trades":train_n,"model_status":train_note,"monte_carlo":None}
            else:
                sig=build_signals(d,symbol,key)
                is_trades=simulate(d,sig,start="2020-01-01",end="2024-01-01",commission_per_lot=args.commission_per_lot,slippage_points=args.slippage_points,commission_points=args.commission_points,breakeven=(key=="A"))
                oos_trades=simulate(d,sig,start="2024-01-01",end="2026-01-01",commission_per_lot=args.commission_per_lot,slippage_points=args.slippage_points,commission_points=args.commission_points,breakeven=(key=="A"))
                result={"symbol":symbol,"asset_class":asset,"strategy":name,"strategy_id":key,"is":_stats(is_trades),"oos":_stats(oos_trades),"training_trades":None,"model_status":"rules only","monte_carlo":None}
            if result["oos"]["trades"]>=args.mc_min_trades and (result["oos"]["profit_factor"] or 0)>=args.mc_min_pf:
                result["monte_carlo"]=_monte_carlo(oos_trades,iterations=args.mc_iterations,risk_pct=args.mc_risk_pct,seed=args.seed+ix*100+ord(key))
                approved.append({"symbol":symbol,"asset_class":asset,"strategy":name,"strategy_id":key,"rule":_rule_spec(key),"oos":result["oos"],"monte_carlo":result["monte_carlo"],"model_path":str((model_dir/(re.sub(r"[^A-Za-z0-9]+","_",symbol).strip("_")+"_D.json")).relative_to(ROOT)) if key=="D" else None})
            all_results.append(result)
        # Avoid holding both source and indicator copies while processing many instruments.
        del d, rawdata
    return all_results,approved


def _rule_spec(key: str) -> dict[str,Any]:
    rules={
        "A":{"fast_ema":12,"slow_ema":26,"rsi_long_min":55,"rsi_short_max":45,"atr14_over_atr50_min":1.0,"initial_stop_atr":1.5,"target_r":2.5,"breakeven_at_r":1.2},
        "B":{"forex_index_sessions":{"london":"08:00-11:00 Europe/London","new_york":"07:00-10:00 America/New_York","silver_bullet":["10:00-11:00","15:00-16:00 America/New_York"]},"synthetic_sessions":None,"synthetic_setup":"structural sweep + 3-candle FVG + directional bias; Boom bullish spike, Crash bearish spike, Range Break/Jump 20-bar breakout"},
        "C":{"bollinger_period":20,"bollinger_std":2.0,"keltner_period":20,"keltner_atr":1.5,"zscore_period":50,"zscore_extreme":2.0,"efficiency_ratio_max":0.25,"target":"daily VWAP; 1.5 ATR stop"},
        "D":{"breakout_lookback":20,"atr_ratio_min":1.1,"model":"XGBoost binary classifier","train_period":"2020-01-01/2023-12-31","oos_period":"2024-01-01/2025-12-31","probability_threshold":0.55,"features":list(FEATURES)},
    }
    return rules[key]


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=DATA_DEFAULT)
    parser.add_argument("--symbols",nargs="*",help="optional exact symbols for a small validation run")
    parser.add_argument("--output-dir",type=Path,default=OUT_DEFAULT)
    parser.add_argument("--approved-path",type=Path,default=APPROVED_PATH)
    parser.add_argument("--commission-per-lot",type=float,default=7.0,help="round-turn account-currency commission per standard lot")
    parser.add_argument("--commission-points",type=float,default=1.0,help="fallback round-turn commission in symbol points if MT5 tick value is unavailable")
    parser.add_argument("--slippage-points",type=float,default=2.0)
    parser.add_argument("--mc-min-pf",type=float,default=1.25)
    parser.add_argument("--mc-min-trades",type=int,default=80)
    parser.add_argument("--mc-iterations",type=int,default=1000)
    parser.add_argument("--mc-risk-pct",type=float,default=.75)
    parser.add_argument("--max-ruin-pct",type=float,default=5.0,help="max Monte Carlo prop-ruin probability to approve a pair")
    parser.add_argument("--max-p95-drawdown-pct",type=float,default=10.0,help="max 95th-percentile Monte Carlo drawdown to approve a pair")
    parser.add_argument("--xgb-threshold",type=float,default=.55)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.mc_iterations<1 or args.mc_min_trades<1 or not (0<args.mc_risk_pct<=5):parser.error("invalid Monte Carlo configuration")
    try: results,approved=run(args)
    except Exception as exc: print(f"Research failed: {type(exc).__name__}: {exc}",file=sys.stderr); return 2
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"research_results.json").write_text(json.dumps(results,indent=2,allow_nan=False),encoding="utf-8")
    with (args.output_dir/"research_results.csv").open("w",newline="",encoding="utf-8") as f:
        fields=("symbol","asset_class","strategy","strategy_id","is_trades","is_pf","is_return_pct","oos_trades","oos_win_rate_pct","oos_profit_factor","oos_max_drawdown_pct","oos_total_return_pct","oos_avg_monthly_return_pct","oos_sharpe_ratio","mc_p95_max_drawdown_pct","mc_probability_of_ruin_pct")
        w=__import__("csv").DictWriter(f,fieldnames=fields);w.writeheader()
        for r in results:
            mc=r.get("monte_carlo") or {}; row={"symbol":r["symbol"],"asset_class":r["asset_class"],"strategy":r["strategy"],"strategy_id":r["strategy_id"],"is_trades":r["is"]["trades"],"is_pf":r["is"]["profit_factor"],"is_return_pct":r["is"]["total_return_pct"],"oos_trades":r["oos"]["trades"],"oos_win_rate_pct":r["oos"]["win_rate_pct"],"oos_profit_factor":r["oos"]["profit_factor"],"oos_max_drawdown_pct":r["oos"]["max_drawdown_pct"],"oos_total_return_pct":r["oos"]["total_return_pct"],"oos_avg_monthly_return_pct":r["oos"]["average_monthly_return_pct"],"oos_sharpe_ratio":r["oos"]["sharpe_ratio"],"mc_p95_max_drawdown_pct":mc.get("p95_max_drawdown_pct"),"mc_probability_of_ruin_pct":mc.get("probability_of_ruin_pct")};w.writerow(row)
    qualified=[r for r in results if r.get("monte_carlo") is not None]
    approved=[r for r in approved if r["monte_carlo"]["probability_of_ruin_pct"]<=args.max_ruin_pct and r["monte_carlo"]["p95_max_drawdown_pct"]<=args.max_p95_drawdown_pct]
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"period":{"in_sample":"2020-01-01/2023-12-31","out_of_sample":"2024-01-01/2025-12-31"},"approval_criteria":{"oos_profit_factor_min":args.mc_min_pf,"oos_trade_count_min":args.mc_min_trades,"monte_carlo_iterations":args.mc_iterations,"risk_per_trade_pct":args.mc_risk_pct,"prop_daily_drawdown_pct":5,"prop_total_drawdown_pct":10,"max_ruin_probability_pct":args.max_ruin_pct,"max_p95_drawdown_pct":args.max_p95_drawdown_pct},"monte_carlo_qualified":qualified,"approved":approved}
    args.approved_path.parent.mkdir(parents=True,exist_ok=True);args.approved_path.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    profitable=[r for r in results if r["oos"]["total_return_pct"]>0]
    profitable.sort(key=lambda r:(r["oos"]["total_return_pct"],r["oos"]["profit_factor"] or 0),reverse=True)
    print("\nRanked OOS leaderboard (2024-2025; profitable combinations, max 15)")
    print(f"{'#':>2} {'Symbol':<28} {'Strategy':<42} {'Trades':>7} {'WR %':>7} {'PF':>7} {'MaxDD %':>9} {'Return %':>10} {'MC95 DD %':>10} {'Ruin %':>8}")
    print("-"*139)
    for n,r in enumerate(profitable[:15],1):
        o=r["oos"];mc=r.get("monte_carlo") or {}; pf="INF" if o["profit_factor"] is None else f"{o['profit_factor']:.2f}"
        p95=f"{mc['p95_max_drawdown_pct']:.2f}" if "p95_max_drawdown_pct" in mc else "--"
        ruin=f"{mc['probability_of_ruin_pct']:.2f}" if "probability_of_ruin_pct" in mc else "--"
        print(f"{n:>2} {r['symbol']:<28.28} {r['strategy']:<42.42} {o['trades']:>7} {o['win_rate_pct']:>6.2f} {pf:>7} {o['max_drawdown_pct']:>8.2f} {o['total_return_pct']:>9.2f} {p95:>9} {ruin:>8}")
    print(f"\nSymbols={len({r['symbol'] for r in results})}; strategy-runs={len(results)}; Monte-Carlo candidates={len(qualified)}; approved after <= {args.max_ruin_pct:g}% ruin and <= {args.max_p95_drawdown_pct:g}% p95 DD={len(approved)}")
    print(f"Reports: {args.output_dir}; approved file: {args.approved_path}")
    if not profitable: print("No profitable OOS combinations were observed in the available history.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
