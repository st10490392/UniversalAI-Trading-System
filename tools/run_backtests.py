"""Run and compare backtests for each exported one-year MT5 CSV."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.export_mt5_history import classify_symbol

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,default=ROOT/"data"/"historical_12m")
    p.add_argument("--risk-pct",type=float,default=1.0)
    p.add_argument("--daily-dd",type=float,default=4.0)
    p.add_argument("--total-dd",type=float,default=10.0)
    p.add_argument("--balance",type=float,default=10000.0)
    p.add_argument("--spread-fallback-points",type=float,default=20.0)
    p.add_argument("--slippage-points",type=float,default=2.0)
    p.add_argument("--report",type=Path,default=ROOT/"data"/"historical_12m"/"backtest_summary.csv")
    a=p.parse_args();rows=[];failures=[]
    for file in sorted(a.input_dir.glob("*_M15_12M.csv")):
        symbol=file.name.removesuffix("_M15_12M.csv").replace("_"," ")
        try:
            with file.open(newline="",encoding="utf-8") as stream:
                first=next(csv.DictReader(stream),None)
            if first and first.get("symbol"):symbol=first["symbol"]
            command=[sys.executable,str(ROOT/"backtest.py"),str(file),"--symbol",symbol,"--risk-pct",str(a.risk_pct),"--daily-dd",str(a.daily_dd),"--total-dd",str(a.total_dd),"--balance",str(a.balance),"--spread-points",str(a.spread_fallback_points),"--slippage-points",str(a.slippage_points)]
            proc=subprocess.run(command,capture_output=True,text=True,cwd=ROOT,timeout=180)
            if proc.returncode:raise RuntimeError((proc.stderr or proc.stdout)[-1200:])
            result=json.loads(proc.stdout)
            rows.append({"symbol":symbol,"categories":"/".join(classify_symbol(symbol)),**result})
            # Persist progress so a constrained Windows VM can resume after memory pressure.
            a.report.parent.mkdir(parents=True,exist_ok=True)
            import pandas as pd
            pd.DataFrame(rows).to_csv(a.report,index=False)
            print(f"Completed {len(rows)}: {symbol}",flush=True)
        except Exception as exc:
            failures.append(symbol)
            print(f"SKIP {symbol}: backtest failed ({type(exc).__name__}: {exc})")
    print("\n12-month M15 backtest comparison (historical spread + slippage)")
    headers=("Symbol","Win Rate","Profit Factor","Max DD %","Return %","Trades")
    print(f"{headers[0]:<24} {headers[1]:>10} {headers[2]:>14} {headers[3]:>11} {headers[4]:>11} {headers[5]:>8}")
    print("-"*84)
    for r in rows:
        raw_pf=r["profit_factor"]
        if raw_pf is None:raw_pf=float("inf") if r["win_rate_pct"]>0 and r["trade_count"]>0 else 0.0
        pf="INF" if raw_pf==float("inf") else f"{raw_pf:.2f}"
        print(f"{r['symbol']:<24} {r['win_rate_pct']:>9.2f}% {pf:>14} {r['max_drawdown_pct']:>10.2f}% {r['total_return_pct']:>10.2f}% {r['trade_count']:>8d}")
    print(f"\nAssumptions: risk={a.risk_pct:g}%/trade, daily stop={a.daily_dd:g}%, total stop={a.total_dd:g}%, fallback spread={a.spread_fallback_points:g} points, slippage={a.slippage_points:g} points. Exported per-bar spread/point values are used when present.")
    if rows:
        print(f"Full summary written to {a.report}")
    if failures:print(f"Failed symbols ({len(failures)}): {', '.join(failures)}")
    return 0 if rows and not failures else (1 if rows else 2)

if __name__=="__main__":raise SystemExit(main())
