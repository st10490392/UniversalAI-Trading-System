"""Export one year of M15 bars from the already-running MT5 demo terminal."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import pandas as pd

SYMBOLS=("EURUSD","XAUUSD","Volatility 75 Index","Boom 1000 Index","Crash 1000 Index")
ROOT=Path(__file__).resolve().parents[1]

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=ROOT/"data"/"historical_12m")
    a=p.parse_args()
    try:import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 Python package is not installed.",file=sys.stderr);return 2
    if not mt5.initialize(timeout=10000):
        print("Could not connect to the currently configured MT5 terminal; error code:",mt5.last_error()[0],file=sys.stderr);return 2
    try:
        account=mt5.account_info()
        if account is None or account.trade_mode!=mt5.ACCOUNT_TRADE_MODE_DEMO:
            print("Export aborted: the connected MT5 account is not verified as a demo account.",file=sys.stderr);return 3
        visible={s.name for s in (mt5.symbols_get() or ()) if s.visible}
        end=datetime.now(timezone.utc);start=end-timedelta(days=365)
        a.output_dir.mkdir(parents=True,exist_ok=True)
        exported=0
        for symbol in SYMBOLS:
            if symbol not in visible:
                print(f"SKIP {symbol}: not found in Market Watch");continue
            info=mt5.symbol_info(symbol)
            rates=mt5.copy_rates_range(symbol,mt5.TIMEFRAME_M15,start,end)
            if info is None or rates is None or len(rates)==0:
                print(f"SKIP {symbol}: no historical bars returned (MT5 error {mt5.last_error()[0]})");continue
            d=pd.DataFrame(rates)
            d["time"]=pd.to_datetime(d["time"],unit="s",utc=True)
            d=d.rename(columns={"tick_volume":"volume","spread":"spread_points"})
            d["point"]=float(info.point)
            cols=["time","open","high","low","close","volume","spread_points","point"]
            d=d[cols].drop_duplicates("time").sort_values("time")
            safe=symbol.replace(" ","_")
            dest=a.output_dir/f"{safe}_M15_12M.csv"
            d.to_csv(dest,index=False)
            print(f"EXPORTED {symbol}: {len(d)} bars, {d.time.iloc[0].isoformat()} to {d.time.iloc[-1].isoformat()}, file={dest.relative_to(ROOT)}")
            exported+=1
        print(f"EXPORT_COMPLETE files={exported} start_utc={start.isoformat()} end_utc={end.isoformat()}")
        return 0 if exported else 4
    finally:
        mt5.shutdown()

if __name__=="__main__":raise SystemExit(main())
