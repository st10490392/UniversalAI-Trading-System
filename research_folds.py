"""Two expanding walk-forward folds; select target on prior validation trades only."""
import numpy as np
import pandas as pd
from research_lab import _trade_one,standard_trade_stats,standard_qualification
from walkforward_standard import execution_columns


def evaluate(d,spreads,signals,fixed=False):
    days=sorted(set(d.time.dt.date))
    cuts=[d.time.searchsorted(pd.Timestamp(days[len(days)*n//5],tz="UTC")) for n in (2,3,4)]+[len(d)]
    columns=[execution_columns(d,spreads*m) for m in (1,2)]
    def run(start,end,rr,stress):
        trades=[];busy=-1;day=None;daily=0.
        for i,side,stop,limit in signals:
            if i<start or i+1>=end or i<busy: continue
            if (d.time.iat[i+1]-d.time.iat[i]).total_seconds()!=300: continue
            entry_day=d.time.iat[i+1].date()
            if entry_day!=day: day=entry_day;daily=0.
            if fixed and daily<=-20: continue
            t=_trade_one(columns[stress],i,side,stop,np.nan,breakeven=False,commission_per_lot=10.5 if stress else 7,
                         slippage_points=5 if stress else 2,commission_points=1.5 if stress else 1,
                         end_idx=min(end,limit),horizon=96,absolute_stop=True,reward_risk=rr)
            if t is None: continue
            if fixed:
                unit=abs(t["entry"]-stop)*d.tick_value.iat[i]/d.tick_size.iat[i]
                lots=np.floor(min(d.volume_max.iat[i],5/unit)/d.volume_step.iat[i]+1e-10)*d.volume_step.iat[i]
                if lots+1e-10<d.volume_min.iat[i]: continue
                t["r"]*=lots*unit/5
                daily+=t["r"]*5
            trades.append(t);busy=t["exit_index"]
        return trades
    def stats(ts):
        s=standard_trade_stats(ts)
        if fixed:
            equity=10000+np.cumsum([0.]+[t["r"]*5 for t in ts])
            s["max_drawdown_pct"]=float(max(100*(1-equity/np.maximum.accumulate(equity))))
        return s
    rows=[];pool=[[],[]]
    for fold in (0,1):
        def rank(rr):
            trades=run(cuts[fold],cuts[fold+1],rr,1)
            return (len(trades)>=10,np.mean([t["r"] for t in trades]) if trades else -np.inf)
        rr=max((2.5,3.),key=rank)
        streams=[run(cuts[fold+1],cuts[fold+2],rr,s) for s in (0,1)]
        for p,x in zip(pool,streams): p.extend(x)
        rows.append((fold+1,rr,streams))
    rows.append(("pooled",None,pool))
    for fold,rr,(b,s) in rows:
        x,y=stats(b),stats(s)
        yield dict(fold=fold,target=rr,trades=x["trades"],stress_trades=y["trades"],win=x["win_rate_pct"],pf=x["profit_factor"],
                   stressed_pf=y["profit_factor"],dd=max(x["max_drawdown_pct"],y["max_drawdown_pct"]),avg_r=y["average_r"],
                   passed=standard_qualification(x,y)["standard_pass"])
