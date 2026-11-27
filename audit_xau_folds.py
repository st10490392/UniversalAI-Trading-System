"""Replay the frozen gold variant and test exclusion of incomplete sessions."""
import pandas as pd

from fx_stocks_all_strategies import prepare
from fx_stocks_session_orb import inventory
from research_lab import _stats
from strategy_audit_variants import Market, session_ranges, run_trades
from walkforward_standard import execution_columns
from xau_ny_breakout import breakout, entry_window

def audit(path):
    data, spreads, valid = prepare(path)
    sides, stops = breakout(data)
    days = sorted(set(data.time.dt.date))
    complete = session_ranges(data, Market("NY killzone UTC", "UTC", 780, 960, 1260))
    columns = execution_columns(data, spreads * 2)
    for start, end in ((days[len(days)*3//5], days[len(days)*4//5]),
                       (days[len(days)*4//5], days[-1] + pd.Timedelta(days=1))):
        primary, missing = [], []
        omitted = no_valuation = 0
        for day, indices in data.groupby(data.time.dt.date).indices.items():
            if not start <= day < end:
                continue
            entries = [i for i in indices if entry_window(data.time.iloc[i])]
            if not entries:
                continue
            cutoff = [i for i in indices if data.time.iloc[i].hour * 60 + data.time.iloc[i].minute < 1260]
            if day not in complete:
                omitted += 1
                if not any(data.time.iloc[i].strftime("%H:%M") == "15:45" for i in cutoff):
                    no_valuation += 1
            bounds = complete.get(day, (entries[0]-1, entries[-1]+1, cutoff[-1]+1))
            lo, hi, _ = bounds
            choices = [(i-1,int(sides[i-1]),float(stops[i-1])) for i in entries
                       if lo < i < hi and sides[i-1] and valid.iloc[i-1] and valid.iloc[i]
                       and (data.time.iloc[i]-data.time.iloc[i-1]).total_seconds()==900]
            trades = run_trades(columns,[choices],[(lo,hi,bounds[2])],3.,5.,1.5,1)
            (primary if day in complete else missing).extend(trades)
        def metrics(trades):
            return [{k:v for k,v in _stats(trades,r,apply_drawdown_stops=False).items()
                     if k in ("trades","profit_factor","win_rate_pct","max_drawdown_pct")} for r in (.25,.5,1.)]
        print(path.name,start,end,"excluded",omitted,"without 15:45",no_valuation,
              "base",metrics(primary),"sensitivity",metrics(sorted(primary+missing,key=lambda t:t["time"])))

if __name__=="__main__":
    for symbol,path in inventory():
        if symbol=="XAUUSD":
            audit(path)
