"""Fixed-grid, cost-stressed variant and portfolio research on non-synthetic M15 data."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from fx_stocks_all_strategies import adx14, prepare, regular_signals, signed, train_model
from fx_stocks_session_orb import EQUITIES, inventory
from research_lab import ROOT, _trade_one, build_signals, standard_trade_stats
from walkforward_standard import Variant, execution_columns, signal_components, stop_levels

OUT = ROOT / "data/research_2020_2025/strategy_audit_variants"
EXITS = {"1.5R": 1.5, "2R": 2., "3R": 3., "flat": None}


@dataclass(frozen=True)
class Market:
    label: str
    zone: str
    start: int
    entry_end: int
    flat: int


def markets(symbol):
    if symbol == "Germany 40":
        return [Market("Xetra cash", "Europe/Berlin", 540, 720, 1050)]
    if symbol in EQUITIES:
        return [Market("US cash", "America/New_York", 570, 720, 960)]
    return [
        Market("London local open", "Europe/London", 480, 660, 1440),
        Market("NY 08:30 ET", "America/New_York", 510, 660, 1440),
        Market("London killzone UTC", "UTC", 420, 600, 1260),
        Market("NY killzone UTC", "UTC", 780, 960, 1260),
    ]


def session_ranges(data, market):
    local = data.time.dt.tz_convert(market.zone)
    minute = (local.dt.hour*60 + local.dt.minute).to_numpy()
    result = {}
    for day, rows in data.groupby(local.dt.date, sort=False).indices.items():
        if day.weekday() >= 5:
            continue
        starts = rows[minute[rows] == market.start]
        entries = rows[minute[rows] < market.entry_end]
        if market.flat == 1440:
            closing = rows[data.time.iloc[rows].dt.hour.to_numpy()*60 + data.time.iloc[rows].dt.minute.to_numpy() == 1245]
        else:
            closing = rows[minute[rows] == market.flat - 15]
        if len(starts) and len(entries) and len(closing) and starts[0] < entries[-1] < closing[0]:
            result[day] = (int(starts[0]), int(entries[-1]+1), int(closing[0]+1))
    return result


def orb_choices(bounds, bars, retest, is_equity, indicators):
    lo, hi, _ = bounds
    o, h, l, c, ema, trend, atr, volume_ok, spread, valid, ts = indicators
    opening_end = lo+bars
    if opening_end >= hi-1 or not valid[lo:opening_end].all() or not np.all(np.diff(ts[lo:opening_end]) == 900):
        return []
    upper, lower = max(h[lo:opening_end]), min(l[lo:opening_end])
    bias = np.sign(c[opening_end-1]-o[lo])
    last_breakout = 0
    candidates = []
    for i in range(opening_end, hi-1):
        if not valid[i] or not valid[i+1] or ts[i]-ts[i-1] != 900 or ts[i+1]-ts[i] != 900:
            break
        fresh_break = int(c[i] > upper >= c[i-1]) - int(c[i] < lower <= c[i-1])
        if upper-lower < .5*atr[i]:
            continue
        if retest:
            side = last_breakout
            retested = ((side > 0 and l[i] <= upper < c[i] and c[i] > o[i])
                        or (side < 0 and h[i] >= lower > c[i] and c[i] < o[i]))
            if fresh_break:
                last_breakout = fresh_break
            if not retested:
                continue
        else:
            side = fresh_break
        if not side or (is_equity and not volume_ok[i]):
            continue
        if not is_equity and (side != bias or side*trend[i] <= 0 or side*(c[i]-ema[i]) <= 0):
            continue
        if retest:
            stop = min(l[i], upper)-spread[i] if side > 0 else max(h[i], lower)+spread[i]
        else:
            stop = lower-spread[i] if side > 0 else upper+spread[i]
        candidates.append((i, side, stop))
        if retest:
            last_breakout = 0
    return candidates


def raw_families(data, spreads, valid):
    """Produce completed-bar families before session gates are applied."""
    h,l,c,o = data.high,data.low,data.close,data.open
    spread = spreads*data.point_size
    base = regular_signals(data, spreads, valid)
    ema = c.ewm(span=200, adjust=False, min_periods=200).mean()
    trend = ema.diff(5)
    balanced = (adx14(data)<=20)&(data.atr_ratio<=1.2)
    activity = data.atr_ratio > 1.
    result = {name: { "core": values } for name, values in base.items()}
    for name in ("judas_sweep_mss","silver_bullet_fvg"):
        side, stop = base[name]
        result[name] = {"no_htf": (side,stop), "ema200": (np.where(side*trend>0,side,0),stop)}
    # The unfiltered legacy version preserves sweep/FVG order and quant gates.
    prior_lo, prior_hi = l.shift().rolling(20).min(), h.shift().rolling(20).max()
    down_sweep = ((l<prior_lo)&(c>prior_lo)).shift(2, fill_value=False).rolling(4).max().eq(1)
    up_sweep = ((h>prior_hi)&(c<prior_hi)).shift(2, fill_value=False).rolling(4).max().eq(1)
    liquid = (data.atr_ratio>.90)&(data.er20>.25)
    side = signed(down_sweep&(l>h.shift(2))&liquid&(data.zscore<=2),
                  up_sweep&(h<l.shift(2))&liquid&(data.zscore>=-2))
    wick = np.where(side>0,l,h+spread)
    side = np.where(abs(c-wick)<=data.atr14*3,side,0)
    result["legacy_sweep_fvg"] = {"h1_structure":base["legacy_sweep_fvg"], "no_htf":(side,wick)}
    for name, family in (("trend_momentum","A"),("volatility_breakout_xgb","D")):
        side, distance, _, _ = build_signals(data,data.attrs["symbol"],family)
        version = np.where((side*trend>0)&(side*(c-ema)>0)&activity,side,0)
        result[name]["ema200"] = (version, c.to_numpy()-version*distance)
    components=signal_components(data,"ema_trend_pullback",spreads)
    side,stop=stop_levels(data,components,Variant(2.,True,True))
    result["ema_trend_pullback"]["atr_buffer_ema200"]=(np.where(activity,side,0),stop)
    ema20=data.ema20
    result["ema_rsi_pullback"]={}
    for label, long_rsi, short_rsi in (("rsi50_reclaim",50,50),("rsi55_45_reclaim",55,45)):
        rsi_side=signed((l.shift(1)<=ema20.shift(1))&(c>ema20)&(c>o)&(data.rsi.shift(1)<=long_rsi)&(data.rsi>long_rsi)&(trend>0)&activity,
                        (h.shift(1)>=ema20.shift(1))&(c<ema20)&(c<o)&(data.rsi.shift(1)>=short_rsi)&(data.rsi<short_rsi)&(trend<0)&activity)
        result["ema_rsi_pullback"][label]=(rsi_side,np.where(rsi_side>0,l.shift(1)-spread,h.shift(1)+spread))
    side=signed((c>h.shift().rolling(4).max())&(c>ema)&(trend>0)&(data.atr_ratio>1.05),
                (c<l.shift().rolling(4).min())&(c<ema)&(trend<0)&(data.atr_ratio>1.05))
    result["intraday_range_break"]={"4_bar_close":(side,c.to_numpy()-side*data.atr14)}
    side=signed((c>h.shift().rolling(8).max())&(c>ema)&(trend>0)&(data.atr_ratio>1.05),
                (c<l.shift().rolling(8).min())&(c<ema)&(trend<0)&(data.atr_ratio>1.05))
    result["intraday_range_break"]["8_bar_close"]=(side,c.to_numpy()-side*data.atr14)
    return result,ema,trend,balanced


def run_trades(columns, opportunities, bounds, rr, slip, fee, cap, target=None, eligible=None):
    result=[]
    for choices, (_,_,end) in zip(opportunities,bounds):
        next_free=-1
        count=0
        for i,side,stop in choices:
            if i<next_free or (eligible is not None and i not in eligible):
                continue
            price=np.nan if target is None else target[i]
            if target is not None and not np.isfinite(price):
                continue
            trade=_trade_one(columns,i,side,stop,price,breakeven=False,commission_per_lot=7*fee,
                             slippage_points=slip,commission_points=fee,end_idx=end,absolute_stop=True,reward_risk=rr)
            if trade is not None:
                result.append(trade)
                next_free=trade["exit_index"]
                count+=1
                if count>=cap:
                    break
    return result


def scores(baseline, stress, months):
    base, adverse = map(standard_trade_stats,(baseline,stress))
    return {"oos_trades":base["trades"],"stress_trades":adverse["trades"],
            "monthly_trades":base["trades"]/months,"annual_trades":base["trades"]/months*12,
            "stress_monthly_trades":adverse["trades"]/months,"stress_annual_trades":adverse["trades"]/months*12,
            "win_rate_pct":base["win_rate_pct"],"stress_win_rate_pct":adverse["win_rate_pct"],
            "baseline_pf":base["profit_factor"],"stressed_pf":adverse["profit_factor"],
            "max_drawdown_pct":max(base["max_drawdown_pct"],adverse["max_drawdown_pct"]),
            "avg_net_r":base["average_r"],"stressed_avg_net_r":adverse["average_r"]}


def nonoverlap(*streams):
    next_free=-1
    for trade in sorted((t for stream in streams for t in stream),key=lambda t:t["entry_index"]):
        if trade["entry_index"]>next_free:
            yield trade
            next_free=trade["exit_index"]


def combined_check(trials, daily_index):
    """Select by validation only, then return matching standalone and mixed OOS streams."""
    eligible=[]
    for label,stages in trials.items():
        trades=stages[0][1]
        if len(trades)<8:
            continue
        net=np.array([t["r"] for t in trades])
        pf=net[net>0].sum()/max(abs(net[net<0].sum()),1e-12)
        if net.mean()>0 and pf>=1.15:
            series=pd.Series(net,index=[t["time"][:10] for t in trades]).groupby(level=0).sum().reindex(daily_index,fill_value=0)
            eligible.append((float(net.mean()),label,series))
    eligible.sort(key=lambda x:(-x[0],x[1]))
    if not eligible:
        return None
    _,first,sa=eligible[0]
    for _,second,sb in eligible[1:]:
        if first.split("|")[0]==second.split("|")[0]:
            continue
        if sa.std()==0 or sb.std()==0:
            continue
        corr=float(sa.corr(sb))
        if abs(corr)>.35 or (((sa!=0)&(sb!=0)).sum()<3):
            continue
        standalone=trials[first][1]
        combined=[list(nonoverlap(a,b)) for a,b in zip(standalone,trials[second][1])]
        return first,second,corr,standalone,combined
    return None


def evaluate_source(symbol,path):
    data,spreads,valid=prepare(path)
    fx=symbol not in EQUITIES
    families,ema,trend,balanced=raw_families(data,spreads,valid)
    spread=spreads*data.point_size
    indicators=tuple(v.to_numpy() for v in (data.open,data.high,data.low,data.close,ema,trend,
                     data.atr14,data.tick_volume>1.2*data.tick_volume.shift().rolling(20).median(),
                     spread,valid,(data.time-pd.Timestamp("1970-01-01",tz="UTC")).dt.total_seconds()))
    _,h,l,c,_,_,_,_,_,ok,ts=indicators
    execution=[(execution_columns(data,spreads*m),slip,fee) for m,slip,fee in ((1.,2.,1.),(2.,5.,1.5))]
    # One chronological partition shared across all sessions of this dataset.
    calendar=sorted(set(data.time.dt.date))
    cuts=[calendar[len(calendar)*q//5] for q in (2,3,4)]
    duration=(calendar[-1]-cuts[1]).days+1
    months=duration/30.4375
    slices=[{},{}]
    rows=[]
    for market in markets(symbol):
        by_day=session_ranges(data,market)
        days=list(by_day)
        bounds=list(by_day.values())
        split=[int(np.searchsorted(days,t)) for t in cuts]
        if not bounds:
            continue
        vwap=np.full(len(data),np.nan)
        typical=(h+l+c)/3
        volume=data.tick_volume.to_numpy()
        for lo,hi,_ in bounds:
            vwap[lo:hi]=np.cumsum(typical[lo:hi]*volume[lo:hi])/np.cumsum(volume[lo:hi])
        scoped={family:dict(variants) for family,variants in families.items()}
        side=signed(balanced&(data.low<=vwap)&(data.close>vwap)&(data.close>data.open)&(trend>0),
                    balanced&(data.high>=vwap)&(data.close<vwap)&(data.close<data.open)&(trend<0))
        reclaims = np.where(((side>0)&(data.close.shift(1)<vwap))|((side<0)&(data.close.shift(1)>vwap)),side,0)
        wick_stop = np.where(side>0,l-spread,h+spread)
        scoped["vwap_pullback"]={"balanced_touch":(side,wick_stop),"balanced_cross_reclaim":(reclaims,wick_stop)}
        side=signed((data.zscore< -1.5)&balanced&(c<vwap), (data.zscore>1.5)&balanced&(c>vwap))
        scoped["vwap_mean_reversion"]["balanced_vwap_fade"]=(side,c-side*data.atr14.to_numpy()*1.5)
        setups=[]
        for bars in (1,2):
            for retest in (False,True):
                setups.append(("session_orb",f"{bars*15}m_"+("retest" if retest else "close"),
                               [orb_choices(b,bars,retest,not fx,indicators) for b in bounds],bounds))
        for family,variants in scoped.items():
            sweep=family in {"judas_sweep_mss","legacy_sweep_fvg","silver_bullet_fvg"}
            limits=[]
            for lo,hi,end in bounds:
                # Sweep trades flatten by the end of the active window.
                close_bar=hi if sweep else end
                if family=="silver_bullet_fvg":
                    ny=data.time.iloc[lo:hi].dt.tz_convert("America/New_York")
                    active=np.flatnonzero(ny.dt.hour.to_numpy()==10)
                    limits.append((lo,hi,min(close_bar,lo+int(active[-1])+1) if len(active) else lo))
                else:
                    limits.append((lo,hi,close_bar))
            for variant,(sides,stops) in variants.items():
                choices=[]
                for lo,hi,end in limits:
                    choices.append([(int(i),int(sides[i]),float(stops[i])) for i in np.flatnonzero(sides[lo:min(hi,end)-1])+lo
                                    if end>lo+1 and ok[i] and ok[i+1] and ts[i+1]-ts[i]==900
                                    and (family!="silver_bullet_fvg" or data.time.iloc[i+1].tz_convert("America/New_York").hour==10)]
                                   if end>lo+1 else [])
                setups.append((family,variant,choices,limits))
        for family,variant,choices,limits in setups:
            models=[None,None]
            available=[True,True]
            if family=="volatility_breakout_xgb":
                for fold in range(2):
                    models[fold],_=train_model(data,execution[0][0],choices,limits,split[fold])
                    available[fold]=models[fold] is not None
            exits={**EXITS,**({"vwap":None} if family=="vwap_mean_reversion" else {})}
            caps=(1,2) if family in {"session_orb","intraday_range_break"} else (1,)
            for cap in caps:
                for exit_name,rr in exits.items():
                    key="|".join((family,variant,market.label,exit_name,str(cap)))
                    aggregate=[[],[]]
                    for fold in range(2):
                        train_end,val_end=split[fold:fold+2]
                        test_end=split[fold+2] if fold==0 else len(bounds)
                        if not available[fold]:
                            continue
                        stages=[]
                        for a,b in ((train_end,val_end),(val_end,test_end)):
                            stages.append([run_trades(columns,choices[a:b],limits[a:b],rr,slip,fee,cap,
                                                     vwap if exit_name=="vwap" else None,models[fold])
                                           for columns,slip,fee in execution])
                        slices[fold][key]=stages
                        for i,trades in enumerate(stages[1]):
                            aggregate[i].extend(trades)
                    rows.append({"symbol":symbol,"source":path.name,"strategy":family,"variant":variant,
                                 "session":market.label,"exit_profile":exit_name,"max_trades_per_session":cap,
                                 "folds_available":sum(available),"validation_members":"","daily_r_correlation":"",
                                 "oos_start":str(cuts[1]),"oos_end":str(calendar[-1]),"oos_calendar_months":months,
                                 **scores(*aggregate,months),"live_approval":False})
    use=[[],[]]
    anchor=[[],[]]
    picks=[]
    for fold,trials in enumerate(slices):
        first_day,last_day=cuts[fold:fold+2]
        validation_days=[str(d) for d in calendar if first_day<=d<last_day]
        chosen=combined_check(trials,validation_days)
        if chosen:
            a,b,corr,standalone,combined=chosen
            picks.append(f"fold {fold+1}: {a} + {b}")
            for i in range(2):
                use[i].extend(combined[i])
                anchor[i].extend(standalone[i])
            picks[-1]+="; correlation="+f"{corr:.3f}"
    for label,streams in (("validation_selected_primary",anchor),("validation_selected_combination",use)):
        rows.append({"symbol":symbol,"source":path.name,"strategy":"portfolio_check","variant":label,
                     "session":"validated sessions","exit_profile":"fold-selected","max_trades_per_session":"",
                     "folds_available":len(picks),"validation_members":" / ".join(picks),"daily_r_correlation":"",
                     "oos_start":str(cuts[1]),"oos_end":str(calendar[-1]),"oos_calendar_months":months,
                     **scores(*streams,months),"live_approval":False})
    return rows


def main():
    sources=list(inventory())
    with ProcessPoolExecutor(max_workers=min(3, os.cpu_count() or 1)) as pool:
        rows=[row for result in pool.map(evaluate_source, *(zip(*sources))) for row in result]
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"full_strategy_variants_report.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows, {len(sources)} input CSVs",flush=True)


if __name__=="__main__":
    main()
