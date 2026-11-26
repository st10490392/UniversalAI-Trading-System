"""Historical M15 CSV backtester using brain.py's quantitative and SMC rules.

CSV needs time, open, high, low, close; volume is optional. Signals use completed
bars only. If SL and TP are both touched in one bar, SL is conservatively first.
"""
from __future__ import annotations
import argparse
from datetime import timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from brain import in_kill_zone, is_synthetic, market_bias_series, quant_metrics_series

def load_csv(path:str|Path)->pd.DataFrame:
    d=pd.read_csv(path)
    required={"time","open","high","low","close"}
    missing=required-set(d.columns)
    if missing:raise ValueError("CSV missing columns: "+", ".join(sorted(missing)))
    d["time"]=pd.to_datetime(d.time,utc=True,errors="raise")
    for col in ("open","high","low","close"):d[col]=pd.to_numeric(d[col],errors="raise")
    return d.sort_values("time").drop_duplicates("time").reset_index(drop=True)

def _resample_h1(d:pd.DataFrame)->pd.DataFrame:
    # MT5 timestamps mark bar opens: aggregate [hour, next hour) and label by close.
    x=d.set_index("time").resample("1h",label="right",closed="left").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna().reset_index()
    return x

def _walk_exit(future:pd.DataFrame,side:str,entry:float,sl:float,tp:float,trailing_atr:float|None,slippage_price:float)->tuple[float,str,int]:
    stop=sl
    for offset,row in enumerate(future.itertuples(index=False)):
        # On ambiguous OHLC bars prefer stop loss to avoid optimistic fills.
        spread=float(row.spread_price)
        if side=="BUY":
            if row.low<=stop:return stop-slippage_price,"loss",offset
            if row.high>=tp:return tp-slippage_price,"win",offset
            if trailing_atr and trailing_atr>0:stop=max(stop,float(row.close)-trailing_atr)
        else:
            if row.high+spread>=stop:return stop+slippage_price,"loss",offset
            if row.low+spread<=tp:return tp+slippage_price,"win",offset
            if trailing_atr and trailing_atr>0:stop=min(stop,float(row.close)+spread+trailing_atr)
    last=future.iloc[-1]
    close=float(last.close)-(slippage_price if side=="BUY" else -float(last.spread_price)-slippage_price)
    return close,"open_close",max(0,len(future)-1)

def _smc_side_at(high:np.ndarray,low:np.ndarray,close:np.ndarray,i:int,bias:str)->str|None:
    """Allocation-free equivalent of brain.detect_smc for the current bar."""
    if i<24 or bias not in ("BULLISH","BEARISH"):return None
    bullish_fvg=low[i]>high[i-2];bearish_fvg=high[i]<low[i-2]
    if bias=="BULLISH" and not bullish_fvg:return None
    if bias=="BEARISH" and not bearish_fvg:return None
    for j in range(i-2,max(19,i-6),-1):
        level=float(np.min(low[j-20:j]))
        if bias=="BULLISH" and low[j]<level and close[j]>level:return "BUY"
        if bias=="BEARISH" and high[j]>float(np.max(high[j-20:j])) and close[j]<float(np.max(high[j-20:j])):return "SELL"
    return None

def run_backtest(data:pd.DataFrame,symbol:str="EURUSD",risk_pct:float=1.0,max_daily_dd_pct:float=5.0,max_total_dd_pct:float=10.0,initial_balance:float=10000.0,trailing_atr_multiple:float=0.0,spread_points:float=20.0,slippage_points:float=2.0,point:float=0.00001)->dict[str,Any]:
    d=data.copy().reset_index(drop=True)
    if len(d)<400:raise ValueError("At least 400 M15 rows are required for warm-up and H1 structure")
    if risk_pct<=0 or max_daily_dd_pct<=0 or max_total_dd_pct<=0 or initial_balance<=0:raise ValueError("Risk and drawdown limits must be positive")
    h1=_resample_h1(d);h1_bias=market_bias_series(h1).to_numpy()
    quant=quant_metrics_series(d)
    highs=d.high.to_numpy(dtype=float);lows=d.low.to_numpy(dtype=float);closes=d.close.to_numpy(dtype=float)
    q_ready=quant.ready.to_numpy(dtype=bool);q_gate=quant.gate.to_numpy(dtype=bool);q_z=quant.zscore.to_numpy(dtype=float);q_atr=quant.atr14.to_numpy(dtype=float)
    signal_times=pd.to_datetime(d.time,utc=True)+pd.Timedelta(minutes=15)
    signal_ns=signal_times.astype("int64").to_numpy();signal_days=signal_times.dt.date.to_numpy()
    session_minutes=(signal_times.dt.hour*60+signal_times.dt.minute).to_numpy()
    session_ok=((session_minutes>=420)&(session_minutes<600))|((session_minutes>=720)&(session_minutes<960))
    points=d["point"].astype(float).to_numpy() if "point" in d.columns else np.full(len(d),point)
    raw_spread=d["spread_points"].astype(float).to_numpy() if "spread_points" in d.columns else np.full(len(d),spread_points)
    raw_spread=np.where(raw_spread>0,raw_spread,spread_points)
    spread_price=raw_spread*points;slippage_price=slippage_points*points
    d["spread_price"]=spread_price
    h1_ns=pd.to_datetime(h1.time,utc=True).astype("int64").to_numpy()
    balance=initial_balance; peak=initial_balance; max_dd=0.; trades=[]; day_start={}; blocked_days=set(); i=250
    while i<len(d)-1:
        # The signal is generated after this M15 candle has closed.
        ts=signal_times.iloc[i]; day=signal_days[i]
        if day not in day_start:day_start[day]=balance
        if day in blocked_days or (day_start[day]-balance)/day_start[day]*100>=max_daily_dd_pct or (peak-balance)/peak*100>=max_total_dd_pct:
            i+=1;continue
        if not is_synthetic(symbol) and not session_ok[i]:i+=1;continue
        # H1 data is truncated at the signal timestamp to avoid lookahead.
        hi=np.searchsorted(h1_ns,signal_ns[i],side="right")-1
        bias=str(h1_bias[hi]) if hi>=0 else "NEUTRAL"
        if not q_ready[i] or not q_gate[i]:i+=1;continue
        side=_smc_side_at(highs,lows,closes,i,bias)
        if side is None or (side=="BUY" and q_z[i]>2.) or (side=="SELL" and q_z[i]< -2.):i+=1;continue
        atr=float(q_atr[i]);slip=float(slippage_price[i]);spr=float(spread_price[i])
        bid_close=float(closes[i]);entry=bid_close+spr+slip if side=="BUY" else bid_close-slip
        # Sweep wick supplies the structural stop; reject nonpositive or pathological risk.
        sl=float(lows[i]) if side=="BUY" else float(highs[i])+spr
        risk=abs(entry-sl)
        if risk<=0 or risk>atr*3:i+=1;continue
        tp=entry+2.5*risk if side=="BUY" else entry-2.5*risk
        trail=atr*trailing_atr_multiple if trailing_atr_multiple>0 else None
        exit_price,outcome,bars_to_exit=_walk_exit(d.iloc[i+1:].reset_index(drop=True),side,entry,sl,tp,trail,slip)
        unit_r=(exit_price-entry)/risk if side=="BUY" else (entry-exit_price)/risk
        pnl=balance*(risk_pct/100)*unit_r; balance+=pnl; peak=max(peak,balance); max_dd=max(max_dd,(peak-balance)/peak*100)
        trades.append({"time":ts,"month":ts.strftime("%Y-%m"),"side":side,"entry":entry,"sl":sl,"tp":tp,"exit":exit_price,"outcome":outcome,"r":unit_r,"pnl":pnl,"balance":balance})
        if (day_start[day]-balance)/day_start[day]*100>=max_daily_dd_pct:blocked_days.add(day)
        # Single-position simulation: next setup only after this trade's exit horizon.
        i+=max(1,bars_to_exit+1)
    t=pd.DataFrame(trades); wins=t[t.pnl>0] if len(t) else t; losses=t[t.pnl<0] if len(t) else t
    profit_factor=float(wins.pnl.sum()/abs(losses.pnl.sum())) if len(losses) and losses.pnl.sum()!=0 else (float("inf") if len(wins) else 0.)
    monthly=t.groupby("month").size().to_dict() if len(t) else {}
    return {"trades":trades,"trade_count":len(t),"win_rate_pct":float((t.pnl>0).mean()*100) if len(t) else 0.,"profit_factor":profit_factor,"max_drawdown_pct":max_dd,"total_return_pct":(balance/initial_balance-1)*100,"monthly_trade_frequency":monthly,"ending_balance":balance}

def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("csv",help="MT5-exported M15 CSV");p.add_argument("--symbol",default="EURUSD");p.add_argument("--risk-pct",type=float,default=1.);p.add_argument("--daily-dd",type=float,default=5.);p.add_argument("--total-dd",type=float,default=10.);p.add_argument("--balance",type=float,default=10000.);p.add_argument("--trailing-atr",type=float,default=0.,help="0 disables optional ATR trailing");p.add_argument("--spread-points",type=float,default=20.,help="fallback spread when CSV spread is missing or zero");p.add_argument("--slippage-points",type=float,default=2.);p.add_argument("--point",type=float,default=0.00001,help="fallback symbol point when CSV has no point column");p.add_argument("--output",help="optional path for the detailed trade CSV");a=p.parse_args()
    result=run_backtest(load_csv(a.csv),a.symbol,a.risk_pct,a.daily_dd,a.total_dd,a.balance,a.trailing_atr,a.spread_points,a.slippage_points,a.point)
    summary={k:v for k,v in result.items() if k!="trades"};print(pd.Series(summary).to_json(indent=2,default_handler=str))
    if a.output:pd.DataFrame(result["trades"]).to_csv(a.output,index=False)

if __name__=="__main__":main()
