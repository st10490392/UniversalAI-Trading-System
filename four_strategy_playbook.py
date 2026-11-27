"""Closed-M15 playbook research. No terminal or execution adapter."""
from collections import defaultdict
import numpy as np
import pandas as pd

from fx_stocks_all_strategies import prepare, adx14
from fx_stocks_session_orb import inventory
from research_lab import ROOT, standard_trade_stats, standard_qualification
from xau_ny_breakout import breakout

OUT = ROOT / "data/research_2020_2025/four_strategy_playbook"


def features(d, minutes=15):
    t = d.time
    h = d.set_index("time").resample("1h").agg({"high":"max","low":"min","close":"last","open":"count"})
    h = h[h.open == 60//minutes]
    prev = h.close.shift()
    tr = pd.concat([h.high-h.low,(h.high-prev).abs(),(h.low-prev).abs()],axis=1).max(axis=1)
    ht = pd.DataFrame({"hour_atr":tr.rolling(14).mean(),"hour_ema":h.close.ewm(span=200,min_periods=200,adjust=False).mean()})
    ht.index += pd.Timedelta(hours=1)
    aligned = ht.reindex(pd.DatetimeIndex(t)+pd.Timedelta(minutes=minutes),method="ffill")
    d = d.assign(hour_atr=aligned.hour_atr.to_numpy(),hour_ema=aligned.hour_ema.to_numpy(),adx=adx14(d))
    # A pivot at i-2 is confirmed only after both right-hand candles close.
    for col,fn in (("high","max"),("low","min")):
        x=d[col]
        pivot=x.shift(2).where(x.shift(2)==getattr(x.rolling(5),fn)())
        d["swing_"+col]=pivot.ffill().shift()
    d["middle"]=d.close.rolling(20).mean()
    d["flat_loose"]=((d.bb_upper-d.bb_upper.shift(4)).abs()<=.50*d.atr14)&((d.bb_lower-d.bb_lower.shift(4)).abs()<=.50*d.atr14)
    d["flat"]=((d.bb_upper-d.bb_upper.shift(4)).abs()<=.25*d.atr14)&((d.bb_lower-d.bb_lower.shift(4)).abs()<=.25*d.atr14)
    return d


def opening(d, n):
    top,bot=d.high.iloc[:n].max(),d.low.iloc[:n].min()
    width=top-bot
    if not .5*d.hour_atr.iloc[n-1]<=width<=1.5*d.hour_atr.iloc[n-1]:
        return []
    prev=d.close.shift()
    side=np.select([(prev<=top)&(d.close>top),(prev>=bot)&(d.close<bot)],[1,-1],0)
    return [(i,int(side[j]),(top+bot)/2,np.nan,np.nan) for j,i in enumerate(d.index) if j>=n and side[j]]


def pullback(d):
    pv=(d.high+d.low+d.close)/3*d.tick_volume
    vwap=pv.cumsum()/d.tick_volume.cumsum()
    signals=[]
    for side in (1,-1):
        wick,prior=(d.low,d.high) if side==1 else (d.high,d.low)
        band=(45,65) if side==1 else (35,55)
        ok=((d.close-d.hour_ema)*side>0)&((d.close-vwap)*side>0)&((d.ema20-d.ema50)*side>0)
        ok &= ((wick-d.ema20)*side<=0).shift(fill_value=False)&(((d.close-d.ema50)*side>=0).shift(fill_value=False))
        ok &= ((d.close-prior.shift())*side>0)&((d.close-d.open)*side>0)&d.rsi.between(*band)
        stop=wick.shift()-side*.2*d.atr14
        signals.extend((i,side,stop[i],np.nan,np.nan) for i in d.index[ok])
    return sorted(signals)


def asian(d):
    signals=[]
    for side in (1,-1):
        wick,band,opp=(d.low,d.bb_lower,d.bb_upper) if side==1 else (d.high,d.bb_upper,d.bb_lower)
        extreme=(d.rsi<30) if side==1 else (d.rsi>70)
        ok=(((wick-band)*side<0)&extreme).shift(fill_value=False)&d.close.between(d.bb_lower,d.bb_upper)
        ok &= ((d.close-d.open)*side>0)&(d.adx<25)&d.flat
        stop=wick.shift()-side*1.5*d.atr14
        signals.extend((i,side,stop[i],d.middle[i],opp[i]) for i in d.index[ok])
    return sorted(signals)


def sweep(d,top,bot):
    signals=[]
    for side,level,opposite in ((1,bot,top),(-1,top,bot)):
        wick=(d.low if side==1 else d.high).to_numpy()
        close=d.close.to_numpy()
        pivot=(d.swing_high if side==1 else d.swing_low).to_numpy()
        crossed=np.flatnonzero(((wick-level)*side<0)&(np.r_[True,(wick[:-1]-level)*side>=0]))
        for j in crossed:
            inside=next((k for k in range(j,min(j+3,len(d))) if (close[k]-level)*side>0),None)
            if inside is None: continue
            for k in range(inside,len(d)):
                if (close[k]-level)*side<=0: break
                if (close[k]-pivot[j])*side>0:
                    extreme=(min if side==1 else max)(wick[j:k+1])
                    signals.append((d.index[k],side,extreme-side*.2*d.atr14.iloc[k],np.nan,opposite))
                    break
    return sorted(signals)


def simulate(a,signal,end,mode,stress):
    i,s,stop,mid,opp=signal
    j=i+1
    point=a.point_size[j]
    slip=(5 if stress else 2)
    spread=a.spread_points
    m=2 if stress else 1
    entry=a.open[j]+(spread[j]*m if s==1 else 0)*point+s*slip*point
    risk=(entry-stop)*s
    if not np.isfinite(risk) or risk<=0: return None
    partial=mode in ("partialtrail","bands","mid1.5")
    first=mid if mode in ("bands","mid1.5") else entry+s*risk
    rr={"1.5R":1.5,"2R":2,"3R":3,"mid1.5":1.5}.get(mode,np.inf)
    target=opp if mode in ("bands","liquidity") else entry+s*rr*risk
    if (target-entry)*s<=0 or (partial and ((first-entry)*s<=0 or (target-first)*s<=0)): return None
    remaining=1.; result=0.
    for k in range(j,end+1):
        adj=spread[k]*m*a.point_size[k] if s<0 else 0.
        adverse=(a.low[k] if s>0 else a.high[k])+adj
        favorable=(a.high[k] if s>0 else a.low[k])+adj
        op=a.open[k]+adj
        cost=slip*a.point_size[k]/risk
        if (adverse-stop)*s<=0:
            result+=remaining*(min((stop-entry)*s,(op-entry)*s)/risk-cost)
            break
        if partial and remaining==1 and (favorable-first)*s>=0:
            result+=.5*((first-entry)*s/risk-cost)
            remaining=.5
        if (favorable-target)*s>=0:
            result+=remaining*((target-entry)*s/risk-cost)
            break
        if k==end:
            result+=remaining*((a.close[k]+adj-entry)*s/risk-cost)
            break
        if mode=="be" and (favorable-entry)*s>=risk: stop=entry
        if mode=="partialtrail" and remaining==.5 and np.isfinite(a.atr14[k]):
            stop=s*max(s*stop,s*(a.close[k]+adj)-2*a.atr14[k])
    commission=(10.5 if stress else 7)*a.tick_size[j]/a.tick_value[j] if a.tick_value[j]>0 and a.tick_size[j]>0 else (1.5 if stress else 1)*point
    return k,{"time":a.time[j].isoformat(),"r":result-commission/risk}


def completed_m15(d):
    from research_lab import add_indicators
    grouped=d.groupby(d.time.dt.floor("15min"))
    bars=grouped.agg(time=("time","first"),open=("open","first"),high=("high","max"),
                     low=("low","min"),close=("close","last"),tick_volume=("tick_volume","sum"))
    bars=bars.loc[grouped.size()==3].copy()
    bars.index=d.time.searchsorted(bars.time+pd.Timedelta(minutes=10))
    return features(add_indicators(bars))


def evaluate(symbol,path,minutes=15):
    from types import SimpleNamespace
    d,spreads,valid=prepare(path)
    d=features(d,minutes)
    higher=completed_m15(d) if minutes==5 else d
    d["spread_points"]=spreads
    a=SimpleNamespace(**{c:d[c].to_numpy() for c in ("open","close","high","low","spread_points","point_size","tick_size","tick_value","atr14")},time=d.time.array)
    trailing=SimpleNamespace(**vars(a))
    trailing.atr14=higher.atr14.reindex(d.index).to_numpy()
    days=list(d.groupby(d.time.dt.date).indices.items())
    cuts=[days[len(days)*k//5][0] for k in (2,3,4)]
    streams=defaultdict(list)
    prev=None
    gold_side,gold_stop=breakout(d)
    for day,idx in days:
        daily=d.iloc[idx]
        previous=prev
        prev=(daily.high.max(),daily.low.min())
        if day<cuts[0] or day.weekday()>=5: continue
        phase=sum(day>=cut for cut in cuts)-1
        ny=int(pd.Timestamp(str(day)+" 09:30",tz="America/New_York").tz_convert("UTC").hour)*60+30
        minute=(daily.time.dt.hour*60+daily.time.dt.minute)
        levels=[] if previous is None else [previous]
        router=[]
        for label,start,end in (("Asia",0,360),("London",480,ny),("NY",ny,ny+390),("Gold",780,1260)):
            g=daily[(minute>=start)&(minute<end)]
            if len(g)<2: continue
            execution=higher.loc[g.index[0]:g.index[-1]] if minutes==5 else g
            specs=[]
            if label=="Gold":
                if symbol!="XAUUSD" or minutes==5: continue
                sig=[(i,int(gold_side[i]),gold_stop[i],np.nan,np.nan) for i in g.index if gold_side[i] and minute[i]<945]
                specs.append(("Gold baseline","3R",1,sig))
            elif label=="Asia":
                for flatness in ((.25,.50) if minutes==5 else (.25,)):
                    flat=execution.flat if flatness==.25 else execution.flat_loose
                    sig=asian(execution.assign(flat=flat))
                    specs.extend(("Asian MR",mode+(f"/{flatness:.2f}" if minutes==5 else ""),1,sig) for mode in ("bands","mid1.5"))
            else:
                for width in (15,30):
                    n=width//minutes
                    sig=opening(g,n) if len(g)>=n and minute[g.index[0]]==start and minute[g.index[n-1]]==start+minutes*(n-1) else []
                    specs.extend(("Session ORB",f"{mode}/{width}m",cap,sig) for mode in ("1.5R","2R","be") for cap in (1,2))
                sig=pullback(execution)
                specs.extend(("EMA VWAP",mode,1,sig) for mode in ("2R","partialtrail"))
                sig=sorted(set(item for level in levels for item in sweep(g,*level)))
                specs.extend(("Sweep MSS",mode,1,sig) for mode in ("2R","3R","liquidity"))
            for family,profile,cap,signals in specs:
                key=(family,profile,cap)
                mode=profile.split("/")[0]
                for stress in (False,True):
                    count=0; busy=-1
                    for signal in signals:
                        i=signal[0]
                        if i<busy or i>=g.index[-1] or not valid[i] or not valid[i+1] or (a.time[i+1]-a.time[i]).total_seconds()!=minutes*60: continue
                        trade=simulate(trailing if minutes==5 and mode=="partialtrail" else a,signal,g.index[-1],mode,stress)
                        if trade is None: continue
                        k,t=trade
                        streams[key,phase,stress].append(t)
                        busy=k;count+=1
                        if count>=cap: break
                if (family,profile,cap) in (("Session ORB","2R/30m",1),("EMA VWAP","2R",1),("Asian MR","bands/0.25" if minutes==5 else "bands",1),("Sweep MSS","2R",1)):
                    for sig in signals:
                        i=sig[0]
                        if i>=g.index[-1] or not valid[i] or not valid[i+1] or (a.time[i+1]-a.time[i]).total_seconds()!=minutes*60: continue
                        if d.atr_ratio[i]>1.5 and d.adx[i]<25: continue
                        if family in ("Session ORB","EMA VWAP") and d.adx[i]<25: continue
                        if family=="Asian MR" and d.adx[i]>=25: continue
                        router.append((i,0 if family=="Sweep MSS" else 1,family,label,sig,g.index[-1],mode))
            if label in ("Asia","London") and len(g)==(end-start)//minutes: levels.append((g.high.max(),g.low.min()))
        for stress in (False,True):
            busy=-1;used=set()
            for i,_,family,label,sig,end,mode in sorted(router):
                if i<busy or (family,label) in used: continue
                trade=simulate(trailing if minutes==5 and mode=="partialtrail" else a,sig,end,mode,stress)
                if trade:
                    busy,t=trade
                    streams[("Router","fixed",1),phase,stress].append(t)
                    used.add((family,label))
    keys={k for k,_,_ in streams}
    keys.update((f,p,1) for f,p in (("Router","fixed"),("Asian MR","bands"),("Asian MR","mid1.5"),("EMA VWAP","2R"),("EMA VWAP","partialtrail"),("Sweep MSS","2R"),("Sweep MSS","3R"),("Sweep MSS","liquidity")))
    keys.update(("Session ORB",f"{m}/{n}m",c) for m in ("1.5R","2R","be") for n in (15,30) for c in (1,2))
    if minutes==5:
        keys.difference_update({("Asian MR","bands",1),("Asian MR","mid1.5",1)})
        keys.update(("Asian MR",f"{mode}/{threshold}",1) for mode in ("bands","mid1.5") for threshold in ("0.25","0.50"))
    def stats(trades):
        return standard_trade_stats(trades)
    def row(f,p,fold,b,s):
        x,y=stats(b),stats(s)
        return dict(gross_win=sum(max(0,t["r"]) for t in b),gross_loss=sum(max(0,-t["r"]) for t in b),stress_win=sum(max(0,t["r"]) for t in s),stress_loss=sum(max(0,-t["r"]) for t in s),symbol=symbol,source=path.name,strategy=f,profile=p,fold=fold,trades=x["trades"],win_pct=x["win_rate_pct"],baseline_pf=x["profit_factor"],stressed_pf=y["profit_factor"],max_dd_pct=max(x["max_drawdown_pct"],y["max_drawdown_pct"]),avg_net_r=x["average_r"],stress_trades=y["trades"],stress_avg_r=y["average_r"],passed=standard_qualification(x,y)["standard_pass"])
    rows=[]
    for family in sorted({k[0] for k in keys}):
        pool=[[],[]]
        group=sorted(k for k in keys if k[0]==family)
        for fold in (1,2):
            for k in group:
                rows.append(row(family,str(k[1:]),fold,streams[k,fold,False],streams[k,fold,True]))
            def rank(k):
                x=stats(streams[k,fold-1,True])
                return (x["trades"]>=10,x["average_r"] if x["trades"] else -np.inf)
            pick=max(group,key=rank)
            for stress in (False,True): pool[stress].extend(streams[pick,fold,stress])
            rows.append(row(family,"selected "+str(pick[1:]),fold,streams[pick,fold,False],streams[pick,fold,True]))
        rows.append(row(family,"selected","pooled",*pool))
    return rows


if __name__=="__main__":
    from concurrent.futures import ProcessPoolExecutor
    sources=list(inventory())
    with ProcessPoolExecutor(max_workers=4) as workers:
        rows=[r for batch in workers.map(evaluate,*zip(*sources)) for r in batch]
    OUT.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/"walkforward_four_strategies.csv",index=False)
    pooled=frame[frame.fold=="pooled"]
    notes="""# Four-strategy walk-forward research

M15 closes proxy for M5 execution. Entries use the next consecutive M15 open. No historical news calendar was available. London starts at 08:00 UTC and hands off to New York at 09:30 ET. NY closes at 16:00 ET. Asia closes at 06:00 UTC. Missing final bars use the last observed session close, a retrospective valuation assumption. All instruments use these benchmark windows, including index CFDs outside cash hours.

Validation uses the preceding 20% of observed dates to choose exits, ORB length and cap for each of two successive 20% OOS windows. Fixed rules require no fitted coefficients. Profiles with ten validation trades get priority, then stressed expectancy. Results are per archive. Fold rows include every profile and the prior-validation choice.

ORB uses midpoint stops and a stop moved to entry after a completed bar reaches 1R for the breakeven-only profile. Flat bands require both edges to change by at most 0.25 ATR over four bars. Sweep pivots use two bars on each side and are frozen before the sweep. VWAP uses session tick volume. Partial targets and band levels are fixed at signal time. Stops take priority on ambiguous bars. Trailing updates take effect next bar. Baseline uses recorded or causal fallback spread, two points of slippage per side and round-trip commission of 7 per lot or one point. Stress doubles spread, uses five points of slippage and 1.5 times commission. Drawdown is the worse closed-equity drawdown of both scenarios at 1% compounded stop risk.

Router uses fixed profiles, sweep priority, ADX 25 trend threshold and pauses when ATR14/ATR50 exceeds 1.5 with ADX below 25. Maximum one concurrent position. Portfolio drawdown across symbols was not estimated.

"""
    (OUT/"SUMMARY.md").write_text(notes+"```text\n"+pooled.round(4).to_string(index=False)+"\n```\n",encoding="utf-8")
