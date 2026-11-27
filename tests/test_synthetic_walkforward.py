import numpy as np
import pandas as pd
from synthetic_walkforward import signals


def test_prefix_signals_survive_future_price_reversal():
    times=pd.date_range("2024-01-01",periods=1200,freq="15min",tz="UTC")
    close=np.exp(np.linspace(4,5,len(times)))
    close[1000:]+=10*np.arange(200)
    d=pd.DataFrame(dict(time=times,open=close-.05,close=close,low=close-.1,high=close+.1,tick_volume=10))
    h=d.set_index("time").resample("1h").last().reset_index()
    start=times[1050]
    before=dict(signals(d.iloc[:1050],h[h.time<start]))
    d.loc[1050:,"close"]*=.7
    after=dict(signals(d,h))
    assert before["Volatility Breakout"]
    for name in before:
        assert before[name]==[s for s in after[name] if s[0]<=start]

