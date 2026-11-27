"""Run one read-only MT5 market-data bridge process per configured paper account."""
from mt5_live_adapter import Adapter,resolve_symbol
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import time
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from pathlib import Path

def main():
    import MetaTrader5 as mt5
    load_dotenv(Path(__file__).with_name(".bridge.env"))
    load_dotenv(Path(__file__).with_name(".env"))
    p=argparse.ArgumentParser(description=__doc__)
    for key,var in (("account","BRIDGE_ACCOUNT"),("terminal","MT5_TERMINAL_PATH"),("server","MT5_SERVER"),("login","MT5_LOGIN")):
        p.add_argument("--"+key,default=os.getenv(var),required=not os.getenv(var))
    p.add_argument("--feed-token-env",default="BRIDGE_SECRET")
    p.add_argument("--url",default=os.getenv("BACKEND_URL","http://127.0.0.1:8010"))
    p.add_argument("--symbols",nargs="+",default=None)
    args=p.parse_args()
    from urllib.parse import urlsplit
    endpoint=urlsplit(args.url)
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or not (
            endpoint.scheme=="https" or endpoint.scheme=="http" and endpoint.hostname in ("localhost","127.0.0.1","::1")):
        p.error("Remote BACKEND_URL requires HTTPS")
    secret=os.getenv(args.feed_token_env)
    if not secret: p.error("Missing account-specific feed credential")
    headers={"Authorization":"Bearer "+secret}
    base=args.url.rstrip("/")+"/accounts/"+quote(args.account,safe="")
    if not mt5.initialize(args.terminal): raise SystemExit("Terminal unavailable")
    def publish(path,data):
        response=requests.post(base+path,json=data,headers=headers,allow_redirects=False,timeout=(2,35))
        response.raise_for_status()
    response=requests.get(base+"/feed-config",headers=headers,allow_redirects=False,timeout=5)
    response.raise_for_status()
    config=response.json()
    symbols=args.symbols or config["symbols"]
    mapping={symbol:resolve_symbol(mt5,symbol) for symbol in symbols}
    for name in mapping.values():
        if not mt5.symbol_select(name,True): raise SystemExit("Broker symbol unavailable")
    last_plan=None
    adapter=None
    plan_due=0.
    sent={}; futures={}; last={}
    calendar_due=0.
    try:
        with ThreadPoolExecutor(max_workers=len(symbols)+1) as pool:
            while True:
                account=mt5.account_info()
                if not account or str(account.login)!=args.login or account.server!=args.server:
                    raise SystemExit("Terminal account mismatch")
                if config["currency"] and account.currency!=config["currency"]:
                    raise SystemExit("Terminal currency mismatch")
                if time.monotonic()>=calendar_due:
                    pool.submit(publish,"/calendar/refresh",{})
                    calendar_due=time.monotonic()+900
                for symbol in symbols:
                    name=mapping[symbol]
                    q=mt5.symbol_info_tick(name); info=mt5.symbol_info(name)
                    if not q or not info or not info.trade_tick_size or time.time()-q.time>30: continue
                    if sent.get(symbol)!=q.time_msc:
                        try:
                            publish("/quotes",dict(symbol=symbol,timestamp=q.time_msc/1000,bid=q.bid,ask=q.ask,
                                    cash_per_price_unit=info.trade_tick_value_loss/info.trade_tick_size,
                                    volume_min=info.volume_min,volume_max=info.volume_max,volume_step=info.volume_step))
                            sent[symbol]=q.time_msc
                        except requests.RequestException: continue
                    boundary=q.time//300
                    if last.get(symbol)==boundary: continue
                    if symbol in futures and not futures[symbol].done(): continue
                    bars={}
                    for timeframe,tf in (("M5",mt5.TIMEFRAME_M5),("H1",mt5.TIMEFRAME_H1),("H4",mt5.TIMEFRAME_H4),("D1",mt5.TIMEFRAME_D1),("M15",mt5.TIMEFRAME_M15)):
                        rows=mt5.copy_rates_from_pos(name,tf,1,250)
                        if rows is None: break
                        bars[timeframe]=[{k:r[k].item() for k in ("time","open","high","low","close","tick_volume")} for r in rows]
                    if len(bars)==5 and bars["M5"] and bars["M5"][-1]["time"]+300==boundary*300:
                        futures[symbol]=pool.submit(publish,"/bars/"+quote(symbol,safe=""),bars)
                        last[symbol]=boundary
                if time.monotonic()>=plan_due:
                    plan_due=time.monotonic()+1
                    try:
                        response=requests.get(base+"/execution-plan",headers=headers,allow_redirects=False,timeout=(2,3))
                        response.raise_for_status()
                        plan=response.json()
                        matches=(plan.get("login"),plan.get("server"))==(args.login,args.server)
                        if matches and (plan["enabled"] or os.path.exists(os.getenv("MT5_EXECUTION_DB","mt5_execution.db"))) and adapter is None:
                            adapter=Adapter(mt5,args.login,args.server,os.getenv("MT5_EXECUTION_DB","mt5_execution.db"))
                        if adapter and matches:
                            last_plan=plan
                            adapter.sync(plan)
                        elif adapter and last_plan:
                            adapter.sync({**last_plan,"enabled":False})
                    except requests.RequestException:
                        if adapter and last_plan:
                            try: adapter.sync({**last_plan,"enabled":False})
                            except (ValueError,RuntimeError): pass
                    except (ValueError,RuntimeError):
                        import logging
                        logging.getLogger(__name__).warning("Broker synchronization unavailable")
                time.sleep(.25)
    finally:
        mt5.shutdown()

if __name__=="__main__": main()
