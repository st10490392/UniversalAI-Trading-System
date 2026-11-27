"""Replay broker M5 archives with complete resampled M15 and H1 indicators."""
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from four_strategy_playbook import ROOT, evaluate

OUT=ROOT/"data/research_2020_2025/four_strategy_playbook_m5"

def run(path):
    return evaluate(path.stem.removesuffix("_M5").replace("_"," "),path,5)

if __name__=="__main__":
    paths=sorted((ROOT/"data/m5").glob("*_M5.csv"))
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=pd.DataFrame(row for rows in pool.map(run,paths) for row in rows)
    OUT.mkdir(parents=True,exist_ok=True)
    results.to_csv(OUT/"walkforward_m5.csv",index=False)
    pooled=results[results.fold=="pooled"]
    lines=["# Broker M5 walk-forward replay","","Closed UTC M5 bars for ORB and sweep. EMA/VWAP and Asian signals use complete M15 bars, H1 regimes use complete hourly bars. Fills use the following M5 open. Asian band flatness compares 0.25 and 0.50 ATR. Router uses the original strict Asian rule. No historical calendar CSV found, so the red-folder filter is inactive. Broker tick values are export-time estimates. Validation selects profiles on the preceding fifth of trading dates for each of the final two fifths. Commission and slippage are charged on partial exits proportionally. Incomplete sessions use their last observed close.","","## Coverage (UTC)",""]
    for path in paths:
        t=pd.read_csv(path,usecols=["time"]).time
        lines.append(f"- {path.stem}: {len(t):,} bars, {t.iloc[0]} to {t.iloc[-1]}")
    cols=["symbol","strategy","trades","win_pct","baseline_pf","stressed_pf","max_dd_pct","avg_net_r","passed"]
    lines+=["","## Selected profiles, pooled OOS","","```text",pooled[cols].round(4).to_string(index=False),"```"]
    (OUT/"SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
