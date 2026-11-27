"""Owner-issued account licenses and durable notification outbox."""
import hashlib
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

router=APIRouter()
from control_storage import transaction as store


def owner(authorization:str=Header(default="")):
    secret=os.getenv("OWNER_MASTER_KEY")
    if not secret or not secrets.compare_digest(authorization.encode(),("Bearer "+secret).encode()):
        raise HTTPException(401,"Owner credential required")


class Issue(BaseModel):
    account:str=Field(min_length=1,max_length=128)
    scope:Literal["SIGNALS_ONLY","EA_ONLY","FULL_ACCESS"]
    expires_at:datetime|None=None


class Binding(BaseModel):
    device_id:str=Field(min_length=1,max_length=200)
    mt5_account_id:str|None=None
    mt5_server:str|None=None


def public(row):
    return {k:row[k] for k in row.keys() if k!="digest"}


@router.post("/admin/licenses",dependencies=[Depends(owner)])
def issue(data:Issue):
    if data.account not in json.loads(os.getenv("PAPER_ACCOUNTS_JSON","{}")):
        raise HTTPException(422,"Unknown engine account")
    if data.expires_at and (not data.expires_at.tzinfo or data.expires_at.timestamp()<=time.time()):
        raise HTTPException(422,"Expiration must be future and timezone-aware")
    key="uai_"+secrets.token_urlsafe(32); uid=secrets.token_hex(12)
    with store() as db:
        db.execute("INSERT INTO mobile_license(id,digest,account,scope,expires) VALUES(?,?,?,?,?)",
                   (uid,hashlib.sha256(key.encode()).hexdigest(),data.account,data.scope,data.expires_at.timestamp() if data.expires_at else None))
    return {"id":uid,"license_key":key}


@router.get("/admin/licenses",dependencies=[Depends(owner)])
def listing():
    with store() as db: return [public(r) for r in db.execute("SELECT * FROM mobile_license")]


@router.get("/admin/licenses/{uid}",dependencies=[Depends(owner)])
def inspect(uid:str):
    with store() as db:
        row=db.execute("SELECT * FROM mobile_license WHERE id=?",(uid,)).fetchone()
        if row is None: raise HTTPException(404,"Unknown license")
        return public(row)


@router.post("/admin/licenses/{uid}/{action}",dependencies=[Depends(owner)])
def manage(uid:str,action:Literal["reset","revoke"]):
    sql="revoked=1" if action=="revoke" else "device=NULL,login=NULL,server=NULL"
    with store() as db:
        if not db.execute(f"UPDATE mobile_license SET {sql},generation=generation+1 WHERE id=?",(uid,)).rowcount:
            raise HTTPException(404,"Unknown license")
        db.execute("DELETE FROM push_device WHERE license=?",(uid,))
    return {"action":action}


def lookup(db,authorization):
    prefix="Bearer "
    if not authorization.startswith(prefix): raise HTTPException(401,"License required")
    digest=hashlib.sha256(authorization[len(prefix):].encode()).hexdigest()
    r=db.execute("SELECT * FROM mobile_license WHERE digest=?",(digest,)).fetchone()
    if not r or r["revoked"] or (r["expires"] and r["expires"]<=time.time()):
        raise HTTPException(401,"Invalid or inactive license")
    return r


@router.post("/licenses/activate")
def activate(data:Binding,authorization:str=Header(default="")):
    with store() as db:
        r=lookup(db,authorization)
        ea=r["scope"]!="SIGNALS_ONLY"
        login,server=(data.mt5_account_id,data.mt5_server) if ea else (None,None)
        if ea and (not login or not login.isdecimal() or not server or not server.strip()):
            raise HTTPException(422,"MT5 login and server required")
        if r["device"] and (r["device"],r["login"],r["server"])!=(data.device_id,login,server):
            raise HTTPException(403,"Owner reset required")
        db.execute("UPDATE mobile_license SET device=?,login=?,server=? WHERE id=?",(data.device_id,login,server,r["id"]))
        return {"account":r["account"],"scope":r["scope"]}


def licensed(authorization,device,scope,account=None):
    with store() as db:
        r=lookup(db,authorization)
        if not device or r["device"]!=device or (account is not None and r["account"]!=account):
            raise HTTPException(403,"Account or device binding mismatch")
        if r["scope"] not in (scope,"FULL_ACCESS"): raise HTTPException(403,"Insufficient scope")
        return dict(r)


class PushToken(BaseModel):
    token:str=Field(min_length=1,max_length=4096)


@router.post("/notifications/register-device")
def register_token(data:PushToken,authorization:str=Header(default=""),x_device_id:str=Header(default="")):
    r=licensed(authorization,x_device_id,"SIGNALS_ONLY")
    with store() as db:
        current=lookup(db,authorization)
        if current["device"]!=x_device_id: raise HTTPException(403,"Binding changed")
        if db.execute("SELECT 1 FROM push_device WHERE token=? AND license<>?",(data.token,r["id"])).fetchone():
            raise HTTPException(409,"Token already registered")
        db.execute("INSERT INTO push_device VALUES(?,?) ON CONFLICT(license) DO UPDATE SET token=excluded.token",(r["id"],data.token))
    return {"registered":True}


@router.get("/notifications/events")
def events(after:int=0,authorization:str=Header(default=""),x_device_id:str=Header(default="")):
    r=licensed(authorization,x_device_id,"SIGNALS_ONLY")
    with store() as db:
        return [dict(sequence=v[0],alert=json.loads(v[1])) for v in db.execute(
            "SELECT seq,payload FROM alert_event WHERE account=? AND seq>? ORDER BY seq LIMIT 100",(r["account"],after))]


def enqueue(db,account,alert):
    cur=db.execute("INSERT INTO alert_event(account,alert_id,payload) VALUES(?,?,?) ON CONFLICT(account,alert_id) DO NOTHING RETURNING seq",
                   (account,alert["id"],json.dumps(alert)))
    inserted=cur.fetchone()
    if inserted:
        db.execute("""INSERT INTO push_delivery(event,license,generation)
                    SELECT ?,id,generation FROM mobile_license JOIN push_device ON license=id
                    WHERE account=? AND revoked=0 AND (expires IS NULL OR expires>?) AND scope!='EA_ONLY'""",
                   (inserted[0],account,time.time()))


def deliver_one(project):
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    with store() as db:
        row=db.execute("""SELECT p.event,p.license,p.attempts,d.token,e.payload FROM push_delivery p
          JOIN push_device d ON d.license=p.license JOIN mobile_license l ON l.id=p.license
          JOIN alert_event e ON e.seq=p.event WHERE l.generation=p.generation AND l.revoked=0
          AND (l.expires IS NULL OR l.expires>?) AND due<=? ORDER BY due LIMIT 1""",(time.time(),time.time())).fetchone()
        if not row: return False
        db.execute("UPDATE push_delivery SET due=?,attempts=attempts+1 WHERE event=? AND license=?",
                   (time.time()+min(3600,60*2**min(6,row["attempts"])),row["event"],row["license"]))
    credentials,_=google.auth.default(scopes=["https://www.googleapis.com/auth/firebase.messaging"])
    with AuthorizedSession(credentials) as session:
        alert=json.loads(row["payload"])
        response=session.post(f"https://fcm.googleapis.com/v1/projects/{project}/messages:send",timeout=10,
                              json={"message":{"token":row["token"],"notification":{"title":"New setup","body":alert["symbol"]+" "+alert["side"]},
                                               "data":{"event_id":str(row["event"]),"alert_id":alert["id"]}}})
        response.raise_for_status()
    with store() as db: db.execute("DELETE FROM push_delivery WHERE event=? AND license=?",(row["event"],row["license"]))
    return True
