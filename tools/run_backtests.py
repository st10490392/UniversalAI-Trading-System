"""Run and compare backtests for each exported one-year MT5 CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backtest import load_csv, run_backtest

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,default=ROOT/"data"/"historical_12m")
    p.add_argument("--risk-pct",type=float,default=1.0)
    p.add_argument("--daily-dd",type=float,default=4.0)
    p.add_argument("--total-dd",type=float,default=10.0)
    p.add_argument("--balance",type=float,default=10000.0)
    p.add_argument("--spread-fallback-points",type=float,default=20.0)
    p.add_argument("--slippage-points",type=float,default=2.0)
    a=p.parse_args();rows=[]
    for file in sorted(a.input_dir.glob("*_M15_12M.csv")):
        symbol=file.name.removesuffix("_M15_12M.csv").replace("_"," ")
        try:
            result=run_backtest(load_csv(file),symbol,a.risk_pct,a.daily_dd,a.total_dd,a.balance,0.0,a.spread_fallback_points,a.slippage_points)
            rows.append((symbol,result))
        except Exception as exc:
            print(f"SKIP {symbol}: backtest failed ({type(exc).__name__}: {exc})")
    print("\n12-month M15 backtest comparison (historical spread + slippage)")
    headers=("Symbol","Win Rate","Profit Factor","Max DD %","Return %","Trades")
    print(f"{headers[0]:<24} {headers[1]:>10} {headers[2]:>14} {headers[3]:>11} {headers[4]:>11} {headers[5]:>8}")
    print("-"*84)
    for symbol,r in rows:
        pf="∞" if r["profit_factor"]==float("inf") else f"{r['profit_factor']:.2f}"
        print(f"{symbol:<24} {r['win_rate_pct']:>9.2f}% {pf:>14} {r['max_drawdown_pct']:>10.2f}% {r['total_return_pct']:>10.2f}% {r['trade_count']:>8d}")
    print(f"\nAssumptions: risk={a.risk_pct:g}%/trade, daily stop={a.daily_dd:g}%, total stop={a.total_dd:g}%, fallback spread={a.spread_fallback_points:g} points, slippage={a.slippage_points:g} points. Exported per-bar spread/point values are used when present.")
    return 0 if rows else 2

if __name__=="__main__":raise SystemExit(main())
