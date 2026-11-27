"""Export completed broker bars in pages, from the verified connected Deriv demo."""
import argparse
from pathlib import Path
import pandas as pd

SYMBOLS=["Volatility 75 Index","Volatility 100 Index","Volatility 25 Index","Boom 1000 Index","Crash 1000 Index"]
OUT=Path(__file__).resolve().parent/"data/deriv_demo"


def export(mt,path=None):
    if not (mt.initialize(path) if path else mt.initialize()): raise RuntimeError("MT5 unavailable")
    try:
        account=mt.account_info()
        if not account or account.trade_mode!=mt.ACCOUNT_TRADE_MODE_DEMO or not all(w in account.server.lower() for w in ("deriv","demo")) or account.currency!="USD":
            raise RuntimeError("Connected USD Deriv Demo terminal required")
        OUT.mkdir(parents=True,exist_ok=True)
        for symbol in SYMBOLS:
            if not mt.symbol_select(symbol,True): raise RuntimeError(f"Unavailable: {symbol}")
            info=mt.symbol_info(symbol)
            for name,tf in (("M5",mt.TIMEFRAME_M5),("M15",mt.TIMEFRAME_M15),("H1",mt.TIMEFRAME_H1)):
                # Anchor to a completed bar, so paging cannot shift when another bar opens.
                latest=mt.copy_rates_from_pos(symbol,tf,1,1)
                if latest is None or not len(latest): raise RuntimeError(f"No closed {symbol} {name} bars")
                end=pd.Timestamp(int(latest[0]["time"]),unit="s",tz="UTC")
                chunks=[]
                while True:
                    raw=mt.copy_rates_from(symbol,tf,end.to_pydatetime(),10000)
                    if raw is None or not len(raw): break
                    part=pd.DataFrame(raw)
                    chunks.append(part)
                    end=pd.Timestamp(int(part.time.min())-1,unit="s",tz="UTC")
                    if len(part)<10000: break
                if not chunks: raise RuntimeError(f"No history: {symbol} {name}")
                d=pd.concat(chunks).sort_values("time").drop_duplicates("time")
                d.time=pd.to_datetime(d.time,unit="s",utc=True)
                d=d.rename(columns={"spread":"spread_points"})
                for key,value in dict(symbol=symbol,point_size=info.point,tick_size=info.trade_tick_size,
                    tick_value=info.trade_tick_value_loss,volume_min=info.volume_min,volume_step=info.volume_step,volume_max=info.volume_max).items(): d[key]=value
                target=OUT/f"{symbol.replace(' ','_')}_{name}.csv"
                tmp=target.with_suffix(".tmp");d.to_csv(tmp,index=False);tmp.replace(target)
                print(symbol,name,len(d),d.time.min(),d.time.max(),flush=True)
    finally: mt.shutdown()

if __name__=="__main__":
    import MetaTrader5 as mt
    parser=argparse.ArgumentParser();parser.add_argument("--terminal");export(mt,parser.parse_args().terminal)
