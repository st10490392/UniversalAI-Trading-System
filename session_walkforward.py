"""Sweep and reversal benchmarks on closed UTC M5 bars."""
import numpy as np
import pandas as pd
from fx_stocks_all_strategies import prepare
from four_strategy_playbook import features,higher_bias
from research_folds import evaluate
from pathlib import Path


def detectors(d):
    bias=higher_bias(d)
    book={"London Judas":[],"NY Silver Bullet":[]}
    for day,indices in d.groupby(d.time.dt.date).indices.items():
        if day.weekday()>=5: continue
        frame=d.iloc[indices]
        minute=frame.time.dt.hour*60+frame.time.dt.minute
        for name in book:
            london=name=="London Judas"
            start,end=(420,600) if london else (840,900)
            prior=frame[(minute>= (0 if london else 780))&(minute<start)]
            window=frame[(minute>=start)&(minute<end)]
            if len(prior)!=(84 if london else 12) or len(window)!=(end-start)//5: continue
            for side in (1,-1):
                level=prior.low.min() if side==1 else prior.high.max()
                for i in window.index[:-1]:
                    wick=d.low.iat[i] if side==1 else d.high.iat[i]
                    if (wick-level)*side>=0 or (d.close.iat[i]-level)*side<=0: continue
                    pivot=d.swing_high.iat[i] if side==1 else d.swing_low.iat[i]
                    extreme=wick
                    for j in range(i+1,window.index[-1]):
                        if side*(d.close.iat[j]-level)<=0: break
                        extreme=min(extreme,d.low.iat[j]) if side==1 else max(extreme,d.high.iat[j])
                        if london:
                            if side*(d.close.iat[j]-pivot)<=0: continue
                            k=j
                        else:
                            if j<i+2: continue
                            lower,upper=(d.high.iat[j-2],d.low.iat[j]) if side==1 else (d.high.iat[j],d.low.iat[j-2])
                            if upper<=lower: continue
                            k=None
                            for retest in range(j+1,window.index[-1]):
                                if side*(d.close.iat[retest]-level)<=0: break
                                touch=d.low.iat[retest]<=upper if side==1 else d.high.iat[retest]>=lower
                                reclaim=d.close.iat[retest]>upper if side==1 else d.close.iat[retest]<lower
                                if touch and reclaim: k=retest;break
                            if k is None: continue
                        if bias[k]==side:
                            book[name].append((k,side,extreme-side*.2*d.atr14.iat[k],window.index[-1]+1))
                        break
    return {name:sorted(set(items)) for name,items in book.items()}


if __name__=="__main__":
    rows=[]
    for symbol in ("XAUUSD","EURUSD","GBPUSD","US_Tech_100"):
        d,spreads,valid=prepare(Path("data/m5")/(symbol+"_M5.csv"))
        for name,items in detectors(features(d,5)).items():
            signals=[v for v in items if valid[v[0]] and valid[v[0]+1]]
            rows.extend(dict(symbol=symbol,strategy=name,**row) for row in evaluate(d,spreads,signals))
    report=pd.DataFrame(rows);report.to_csv("session_walkforward.csv",index=False)
    print(report[report.fold=="pooled"].to_string(index=False))
