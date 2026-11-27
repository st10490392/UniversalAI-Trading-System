"""Replay selected HTF ledger events using realized equity and UTC circuit breakers."""
import json
from itertools import product
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data/research_2020_2025/m5_session_htf_bias"
OUT=DATA/"three_setup_portfolio"


def replay(trades,config):
    equity=peak=day_open=1.
    day=None
    halted=dd_halted=False
    active=[]
    fills=[]
    blocked=0
    max_dd=0.
    setups={(x["symbol"],x["strategy"]):x["entry_utc"] for x in config["setups"] if x["enabled"]}
    def rollover(timestamp):
        nonlocal day,day_open,halted
        if timestamp[:10]!=day:
            day,day_open,halted=timestamp[:10],equity,False
    def settle(until):
        nonlocal equity,peak,halted,dd_halted,active,max_dd
        for stamp in sorted({t.exit for t,_ in active if t.exit<=until}):
            rollover(stamp)
            for t,budget in active:
                if t.exit==stamp:
                    pnl=budget*t.r
                    equity+=pnl
                    fills.append((stamp,pnl))
            active=[item for item in active if item[0].exit!=stamp]
            peak=max(peak,equity)
            dd=100*(1-equity/peak)
            max_dd=max(dd,max_dd)
            halted |= 100*(1-equity/day_open)>=config["daily_loss_cap_pct"]-1e-10
            dd_halted |= dd>=config["max_drawdown_cap_pct"]-1e-10
    for t in trades.sort_values(["time","symbol","strategy"]).itertuples():
        window=setups.get((t.symbol,t.strategy))
        if not window or not window[0]<=t.time[11:16]<window[1]: continue
        settle(t.time)
        rollover(t.time)
        if len(active)>=config["max_open"]: continue
        if sum(u.symbol==t.symbol for u,_ in active)>=config["max_per_symbol"]: continue
        if sum(u.strategy==t.strategy for u,_ in active)>=config["max_per_strategy"]: continue
        if halted: blocked+=1
        if halted or dd_halted: continue
        active.append((t,equity*config["risk_pct"]/100))
    settle("9999")
    f=pd.DataFrame(fills,columns=["exit","pnl"])
    monthly=f.groupby(f.exit.str[:7]).pnl.sum().reindex([f"2026-{m:02}" for m in range(5,12)],fill_value=0)
    returns=100*monthly/(1+monthly.cumsum()-monthly)
    return dict(trades=len(f),win_pct=100*(f.pnl>0).mean(),pf=f.pnl.clip(lower=0).sum()/-f.pnl.clip(upper=0).sum(),
                dd=max_dd,blocked=blocked,dd_halted=dd_halted,**returns.to_dict())


if __name__=="__main__":
    config=json.loads((ROOT/"three_setup_portfolio.json").read_text())
    ledger=pd.read_csv(DATA/"selected_trades.csv")
    results=[]
    for ny,risk,stress,breaker in product((False,True),(.5,.75),(False,True),(True,False)):
        config["risk_pct"]=risk
        config["daily_loss_cap_pct"]=2 if breaker else float("inf")
        config["setups"][1]["entry_utc"][0]="12:00" if ny else "07:00"
        metrics=replay(ledger[ledger.experiment&(ledger.stress==stress)],config)
        results.append(dict(ny=ny,risk=risk,stress=stress,breaker=breaker,**metrics))
    OUT.mkdir(exist_ok=True)
    frame=pd.DataFrame(results)
    frame.to_csv(OUT/"scenarios.csv",index=False)
    print(frame.round(3).to_string(index=False))
