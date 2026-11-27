"""Broker-side reconciliation for a single explicitly enrolled, connected MT5 terminal."""
import hashlib
import math
import os
import re
import sqlite3
import time
from datetime import datetime,timezone
from decimal import Decimal,ROUND_FLOOR,ROUND_CEILING


def real_allowed():
    return os.getenv("ALLOW_REAL_ACCOUNTS","").lower()=="true"


def resolve_symbol(mt,name):
    direct=mt.symbol_info(name)
    if direct: return direct.name
    aliases={"XAUUSD":["XAUUSD","Gold"],"Germany 40":["Germany 40","GER40","DE40"],
             "Wall Street 30":["US30","DJ30"],"US Tech 100":["US100","NAS100"]}
    keys=[name,*aliases.get(name,[])]
    names=[s.name for s in (mt.symbols_get() or ())]
    exact=[n for n in names if n.casefold() in {a.casefold() for a in keys}]
    options=exact or [n for n in names if any(re.fullmatch(re.escape(a)+r"(?:[._-][A-Za-z0-9]+|m|pro|micro|cash)",n,re.I) for a in keys)]
    if len(options)!=1: raise ValueError("Unavailable or ambiguous broker symbol: "+name)
    return options[0]


def floor_volume(value,step):
    return float((Decimal(str(value))/Decimal(str(step))).to_integral_value(rounding=ROUND_FLOOR)*Decimal(str(step)))


class Adapter:
    def __init__(self,mt,login,server,journal):
        self.mt,self.login,self.server=mt,str(login),server
        self.magic=int(hashlib.sha256((self.login+"@"+server).encode()).hexdigest()[:12],16)
        self.db=sqlite3.connect(journal)
        self.db.execute("CREATE TABLE IF NOT EXISTS attempts(magic INTEGER,id TEXT,tag TEXT,lots REAL,PRIMARY KEY(magic,id))")
        self.db.execute("CREATE TABLE IF NOT EXISTS initial_risk(magic INTEGER,id TEXT,stop REAL,PRIMARY KEY(magic,id))")
        self.db.execute("CREATE TABLE IF NOT EXISTS guards(magic INTEGER PRIMARY KEY,day TEXT,opening REAL,peak REAL,locked TEXT)")

    def account(self):
        a=self.mt.account_info()
        if not a or str(a.login)!=self.login or a.server!=self.server: raise ValueError("MT5 binding mismatch")
        if not (a.trade_mode==self.mt.ACCOUNT_TRADE_MODE_DEMO and "demo" in a.server.casefold()) and not real_allowed():
            raise ValueError("Broker execution requires verified demo mode and Demo server")
        return a

    def send(self,request):
        self.account()
        result=self.mt.order_send(request)
        if result is None or result.retcode not in (self.mt.TRADE_RETCODE_DONE,self.mt.TRADE_RETCODE_DONE_PARTIAL):
            raise RuntimeError("Broker request was not confirmed")

    def deal(self,symbol,side,volume,**fields):
        m=self.mt
        info=m.symbol_info(symbol); q=m.symbol_info_tick(symbol)
        if not info or not q or not 0<=time.time()-q.time<=30: raise ValueError("Stale broker quote")
        filling=m.ORDER_FILLING_IOC if info.filling_mode&2 else m.ORDER_FILLING_FOK
        return dict(action=m.TRADE_ACTION_DEAL,symbol=symbol,type=side,volume=volume,
                    price=q.ask if side==m.ORDER_TYPE_BUY else q.bid,deviation=10,
                    magic=self.magic,type_filling=filling,**fields)

    def sync(self,plan):
        m=self.mt
        a=self.account()
        positions=m.positions_get()
        if positions is None: raise ValueError("Missing broker positions")
        day=datetime.now(timezone.utc).date().isoformat()
        prev=self.db.execute("SELECT day,opening,peak,locked FROM guards WHERE magic=?",(self.magic,)).fetchone()
        opening=prev[1] if prev and prev[0]==day else a.equity
        peak=max(a.equity,prev[2] if prev else a.equity)
        locked=prev[3] if prev and (prev[0]==day or prev[3]=="TOTAL") else ""
        c=plan["risk"]
        if a.equity<=0 or peak<=0 or 100*(peak-a.equity)>=peak*c["max_dd_cap_pct"]: locked="TOTAL"
        elif opening-a.equity>=c.get("daily_loss_cap_usd",opening*c["daily_loss_cap_pct"]/100) and not locked: locked="DAILY"
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO guards VALUES(?,?,?,?,?)",(self.magic,day,opening,peak,locked))
        wanted={} if locked or not plan["enabled"] else {p["id"]:p for p in plan["positions"]}
        bytag={r[1]:(r[0],r[2]) for r in self.db.execute("SELECT id,tag,lots FROM attempts WHERE magic=?",(self.magic,))}
        for pos in positions:
            if pos.magic!=self.magic: continue
            key,initial=bytag.get(pos.comment,(None,0))
            p=wanted.get(key)
            opposite=m.ORDER_TYPE_SELL if pos.type==m.ORDER_TYPE_BUY else m.ORDER_TYPE_BUY
            info=m.symbol_info(pos.symbol)
            if not p:
                self.send(self.deal(pos.symbol,opposite,pos.volume,position=pos.ticket))
                continue
            buy=pos.type==m.ORDER_TYPE_BUY
            quote=m.symbol_info_tick(pos.symbol)
            if not quote or not 0<=time.time()-quote.time<=30: continue
            market=quote.bid if buy else quote.ask
            saved=self.db.execute("SELECT stop FROM initial_risk WHERE magic=? AND id=?",(self.magic,key)).fetchone()
            original=saved[0] if saved else p["stop"]
            distance=abs(pos.price_open-original)
            reached=(market-pos.price_open)*(1 if buy else -1)>=distance
            remainder=floor_volume(initial/2,info.volume_step)
            if p.get("exit_rule")=="partialtrail" and reached:
                amount=floor_volume(round(pos.volume-remainder,8),info.volume_step)
                if amount>=info.volume_min:
                    self.send(self.deal(pos.symbol,opposite,amount,position=pos.ticket))
            desired=p["stop"]
            if p.get("exit_rule")=="partialtrail" and pos.volume>remainder+1e-8:
                desired=original
            if p.get("exit_rule")=="be" and desired!=original:
                desired=pos.price_open
            stop=self.grid(desired,info.trade_tick_size,buy)
            if (stop>pos.sl if buy else stop<pos.sl) and abs(stop-pos.sl)>=info.trade_tick_size*.99:
                self.send(dict(action=m.TRADE_ACTION_SLTP,symbol=pos.symbol,position=pos.ticket,sl=stop,tp=pos.tp,magic=self.magic))

        for key,p in wanted.items():
            if not 0<=time.time()-p["opened_at"]<=20 or p.get("partial"): continue
            if self.db.execute("SELECT 1 FROM attempts WHERE magic=? AND id=?",(self.magic,key)).fetchone(): continue
            symbol=resolve_symbol(m,p["symbol"])
            if not m.symbol_select(symbol,True): continue
            current=m.positions_get(symbol=symbol)
            if current is None or current: continue
            info=m.symbol_info(symbol)
            buy=p["side"]=="BUY"
            request=self.deal(symbol,m.ORDER_TYPE_BUY if buy else m.ORDER_TYPE_SELL,0)
            price=request["price"]
            sl=self.grid(p["stop"],info.trade_tick_size,buy)
            target=self.grid(p["target"],info.trade_tick_size,not buy)
            if not (sl<price<target if buy else target<price<sl): continue
            loss=m.order_calc_profit(request["type"],symbol,1.,price,sl)
            if loss is None or not math.isfinite(loss) or loss>=0: continue
            budget=c.get("risk_amount_usd",min(a.balance,a.equity)*c["risk_pct"]/100)
            lots=floor_volume(min(info.volume_max,p["volume"],budget/-loss),info.volume_step)
            daily_cap=c.get("daily_loss_cap_usd",opening*c["daily_loss_cap_pct"]/100)
            headroom=min(daily_cap-(opening-a.equity),peak*c["max_dd_cap_pct"]/100-(peak-a.equity))
            if lots<info.volume_min or lots*(-loss)>=headroom: continue
            if p.get("exit_rule")=="partialtrail" and (floor_volume(lots/2,info.volume_step)<info.volume_min or abs(2*floor_volume(lots/2,info.volume_step)-lots)>1e-8): continue
            tag="uai"+hashlib.sha256(key.encode()).hexdigest()[:20]
            request.update(volume=lots,sl=sl,tp=0. if p.get("exit_rule") in ("partialtrail","be") else target,comment=tag)
            check=m.order_check(request)
            if check is None or check.retcode!=0: continue
            with self.db:
                if not self.db.execute("INSERT OR IGNORE INTO attempts VALUES(?,?,?,?)",(self.magic,key,tag,lots)).rowcount: continue
                self.db.execute("INSERT INTO initial_risk VALUES(?,?,?)",(self.magic,key,sl))
            self.send(request)

    @staticmethod
    def grid(price,tick,down):
        unit=Decimal(str(tick))
        return float((Decimal(str(price))/unit).to_integral_value(rounding=ROUND_FLOOR if down else ROUND_CEILING)*unit)
