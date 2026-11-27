"""Run one read-only MT5 market-data bridge process per configured paper account."""
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
    load_dotenv(Path(__file__).with_name(".env"))
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("account","terminal","server","login","feed-token-env"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--url",default="http://127.0.0.1:8010")
    p.add_argument("--symbols",nargs="+",default=None)
    args=p.parse_args()
    headers={"Authorization":"Bearer "+os.environ[args.feed_token_env]}
    base=args.url.rstrip("/")+"/accounts/"+quote(args.account,safe="")
    if not mt5.initialize(args.terminal): raise SystemExit("Terminal unavailable")
    def publish(path,data):
        response=requests.post(base+path,json=data,headers=headers,timeout=(2,35))
        response.raise_for_status()
    response=requests.get(base+"/feed-config",headers=headers,timeout=5)
    response.raise_for_status()
    config=response.json()
    symbols=args.symbols or config["symbols"]
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
                    q=mt5.symbol_info_tick(symbol); info=mt5.symbol_info(symbol)
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
                    for name,tf in (("M5",mt5.TIMEFRAME_M5),("H1",mt5.TIMEFRAME_H1),("H4",mt5.TIMEFRAME_H4),("D1",mt5.TIMEFRAME_D1),("M15",mt5.TIMEFRAME_M15)):
                        rows=mt5.copy_rates_from_pos(symbol,tf,1,250)
                        if rows is None: break
                        bars[name]=[{k:r[k].item() for k in ("time","open","high","low","close","tick_volume")} for r in rows]
                    if len(bars)==5 and bars["M5"] and bars["M5"][-1]["time"]+300==boundary*300:
                        futures[symbol]=pool.submit(publish,"/bars/"+quote(symbol,safe=""),bars)
                        last[symbol]=boundary
                time.sleep(.25)
    finally:
        mt5.shutdown()

if __name__=="__main__": main()
