import numpy as np
import pandas as pd
from export_deriv_demo import OUT,SYMBOLS
from research_lab import add_indicators,_causal_spreads
from research_folds import evaluate


def signals(m,h):
    d=add_indicators(m)
    closed=pd.DatetimeIndex(d.time)+pd.Timedelta(minutes=15)
    series=h.set_index("time").close
    trend=np.sign(series-series.ewm(span=200,min_periods=200,adjust=False).mean())
    trend.index+=pd.Timedelta(hours=1)
    trend=trend.reindex(closed,method="ffill").to_numpy()
    for label in ("EMA VWAP","Volatility Breakout"):
        if label=="EMA VWAP":
            long=(d.low.shift()<=d.ema20.shift())&(d.close>d.high.shift())&(d.ema20>d.ema50)&(d.close>d.vwap)
            short=(d.high.shift()>=d.ema20.shift())&(d.close<d.low.shift())&(d.ema20<d.ema50)&(d.close<d.vwap)
        else:
            long=(d.close>d.high.shift().rolling(20).max())&(d.atr_ratio>1.05)
            short=(d.close<d.low.shift().rolling(20).min())&(d.atr_ratio>1.05)
        side=np.select([long&(trend>0),short&(trend<0)],[1,-1],0)
        yield label,[(closed[i],int(side[i]),d.close.iat[i]-side[i]*1.5*d.atr14.iat[i]) for i in np.flatnonzero(side)]


if __name__=="__main__":
    results=[]
    for symbol in SYMBOLS:
        def read(tf):
            return pd.read_csv(OUT/f"{symbol.replace(' ','_')}_{tf}.csv",parse_dates=["time"])
        m5,m15,h1=map(read,("M5","M15","H1"))
        clock=pd.Index(m5.time)
        for label,items in signals(m15,h1):
            entries=[(clock.get_loc(t)-1,s,stop,len(m5)) for t,s,stop in items if t in clock and t>m5.time.iat[0]]
            results.extend(dict(symbol=symbol,strategy=label,**r) for r in evaluate(m5,_causal_spreads(m5),entries,True))
    report=pd.DataFrame(results);report.to_csv("synthetic_walkforward.csv",index=False)
    print(report[report.fold=="pooled"].to_string(index=False))
