"""Export one year of M15 bars from the already-running MT5 demo terminal."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sys

MAJOR_FX=("EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY")
METAL_ALIASES=("xauusd","gold","xagusd","silver")
INDEX_ALIASES={
    "NASDAQ":("nasdaq","nas100","ustec","ustech100","ustech"),
    "GER30":("ger30","ger40","de30","de40","germany30","germany40","dax"),
    "US30":("us30","wallstreet30","dowjones","dj30"),
}
ROOT=Path(__file__).resolve().parents[1]

def classify_symbol(name:str)->list[str]:
    low=name.lower();compact=re.sub(r"[^a-z0-9]","",low);categories=[]
    if "volatility" in low or "ladder volatility" in low:categories.append("Volatility")
    if "boom" in low or "crash" in low:categories.append("Boom/Crash")
    if "step index" in low or "skew step" in low:categories.append("Step")
    if "jump" in low:categories.append("Jump")
    if "rangebreak" in compact:categories.append("Range Break")
    if any(compact.startswith(pair.lower()) or compact.endswith(pair.lower()) for pair in MAJOR_FX):categories.append("Major FX")
    if any(alias in compact for alias in METAL_ALIASES):categories.append("Metals")
    for family,aliases in INDEX_ALIASES.items():
        if any(alias in compact for alias in aliases):categories.append(family)
    return categories

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=ROOT/"data"/"historical_12m")
    p.add_argument("--list-only",action="store_true",help="show Market Watch matches without exporting bars")
    p.add_argument("--symbols",nargs="+",help="optional exact Market Watch symbols to export")
    a=p.parse_args()
    try:import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 Python package is not installed.",file=sys.stderr);return 2
    if not mt5.initialize(timeout=10000):
        print("Could not connect to the currently configured MT5 terminal; error code:",mt5.last_error()[0],file=sys.stderr);return 2
    try:
        import pandas as pd
        account=mt5.account_info()
        if account is None or account.trade_mode!=mt5.ACCOUNT_TRADE_MODE_DEMO:
            print("Export aborted: the connected MT5 account is not verified as a demo account.",file=sys.stderr);return 3
        visible=sorted({s.name for s in (mt5.symbols_get() or ()) if s.visible})
        selected={name:classify_symbol(name) for name in visible}
        selected={name:cats for name,cats in selected.items() if cats}
        if a.symbols:
            for requested in a.symbols:
                if requested not in visible:print(f"SKIP {requested}: not found in Market Watch")
            selected={name:cats for name,cats in selected.items() if name in set(a.symbols)}
        if not selected:
            print("No requested symbol families are visible in Market Watch.",file=sys.stderr);return 4
        end=datetime.now(timezone.utc);start=end-timedelta(days=365)
        a.output_dir.mkdir(parents=True,exist_ok=True)
        exported=0
        counts={}
        for cats in selected.values():
            for cat in cats:counts[cat]=counts.get(cat,0)+1
        print("MARKET_WATCH_FAMILIES "+", ".join(f"{k}={v}" for k,v in sorted(counts.items())))
        for name,cats in selected.items():print(f"MATCH {name}: {'/'.join(cats)}")
        for family in ("Volatility","Boom/Crash","Step","Jump","Range Break","Major FX","Metals","NASDAQ","GER30","US30"):
            if not counts.get(family):print(f"SKIP family {family}: no matching symbol in Market Watch")
        for pair in MAJOR_FX:
            if not any(pair.lower() in re.sub(r"[^a-z0-9]","",name.lower()) for name in visible):print(f"SKIP major pair {pair}: not found in Market Watch")
        for metal in ("XAUUSD","XAGUSD"):
            if not any(metal.lower() in re.sub(r"[^a-z0-9]","",name.lower()) or metal=="XAUUSD" and "gold" in name.lower() or metal=="XAGUSD" and "silver" in name.lower() for name in visible):print(f"SKIP metal {metal}: not found in Market Watch")
        if a.list_only:return 0
        for symbol,categories in selected.items():
            info=mt5.symbol_info(symbol)
            rates=mt5.copy_rates_range(symbol,mt5.TIMEFRAME_M15,start,end)
            if info is None or rates is None or len(rates)==0:
                print(f"SKIP {symbol}: no historical bars returned (MT5 error {mt5.last_error()[0]})");continue
            d=pd.DataFrame(rates)
            d["time"]=pd.to_datetime(d["time"],unit="s",utc=True)
            d=d.rename(columns={"tick_volume":"volume","spread":"spread_points"})
            d["point"]=float(info.point)
            d["symbol"]=symbol
            cols=["symbol","time","open","high","low","close","volume","spread_points","point"]
            d=d[cols].drop_duplicates("time").sort_values("time")
            safe=re.sub(r"[^A-Za-z0-9]+","_",symbol).strip("_")
            dest=a.output_dir/f"{safe}_M15_12M.csv"
            d.to_csv(dest,index=False)
            print(f"EXPORTED {symbol} [{'/'.join(categories)}]: {len(d)} bars, {d.time.iloc[0].isoformat()} to {d.time.iloc[-1].isoformat()}, file={dest.relative_to(ROOT)}")
            exported+=1
        print(f"EXPORT_COMPLETE files={exported} start_utc={start.isoformat()} end_utc={end.isoformat()}")
        return 0 if exported else 4
    finally:
        mt5.shutdown()

if __name__=="__main__":raise SystemExit(main())
