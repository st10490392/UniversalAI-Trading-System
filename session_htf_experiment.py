"""Compare selected standalone strategies under shared account exposure limits."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
from four_strategy_playbook import ROOT, MAJORS, evaluate

OUT=ROOT/"data/research_2020_2025/m5_session_htf_bias"

def replay(path):
    symbol=path.stem.removesuffix("_M5").replace("_"," ")
    rows=[]; ledger=[]
    for experiment in (False,True):
        trades=[]
        results=evaluate(symbol,path,5,experiment,trades)
        for r in results: r["experiment"]=experiment
        rows.extend(results)
        ledger.extend(dict(t,experiment=experiment) for t in trades)
    return rows,ledger


def portfolio(d, curve_path=None):
    equity=peak=1.; dd=0.; active=[]; settled=[]
    def settle(until):
        nonlocal equity,peak,dd,active
        for stamp in sorted({t.exit for t,_ in active if t.exit<=until}):
            pnl=sum(budget*t.r for t,budget in active if t.exit==stamp)
            equity+=pnl
            peak=max(equity,peak)
            dd=max(dd,100*(1-equity/peak))
            settled.append((stamp,equity))
            active=[item for item in active if item[0].exit!=stamp]
    for t in d.sort_values(["time","symbol","strategy"]).itertuples():
        settle(t.time)
        if len(active)>=2 or any(other.symbol==t.symbol or
              (other.symbol in MAJORS and t.symbol in MAJORS) for other,_ in active): continue
        active.append((t,equity*.005))
    settle("9999")
    curve=pd.DataFrame(settled,columns=["time","equity"])
    if curve_path is not None:
        curve.to_csv(curve_path,index=False)
    ends=curve.groupby(curve.time.str[:7]).equity.last()
    ends=ends.reindex(months(d),method="ffill").fillna(1.)
    return (ends/ends.shift(fill_value=1)-1)*100,dd


def months(d):
    return pd.period_range(d.time.min()[:7],d.exit.max()[:7],freq="M").astype(str)


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        batches=list(pool.map(replay,sorted((ROOT/"data/m5").glob("*_M5.csv"))))
    pd.DataFrame([r for rows,_ in batches for r in rows]).to_csv(OUT/"walkforward.csv",index=False)
    trades=pd.DataFrame([t for _,batch in batches for t in batch])
    trades.to_csv(OUT/"selected_trades.csv",index=False)
    out={};draw={}
    for name,g in trades[trades.strategy!="Router"].groupby(["experiment","stress"]):
        out[name],draw[name]=portfolio(g,OUT/f"equity_{name[0]}_{name[1]}.csv")
    print(pd.DataFrame(out).round(4));print("DD",draw)
    gold=trades[(~trades.experiment)&(trades.strategy=="EMA VWAP")&(trades.symbol=="XAUUSD")]
    for stress,g in gold.groupby("stress"):
        print("stress",stress)
        for col in (g.time.str[:7],g.session):
            print(g.groupby(col).r.agg(count="size",r="sum",half=lambda x:100*((1+.005*x).prod()-1),one=lambda x:100*((1+.01*x).prod()-1)).round(6))
    pd.DataFrame(out).to_csv(OUT/"portfolio_monthly.csv")
