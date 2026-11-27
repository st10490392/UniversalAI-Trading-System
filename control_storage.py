"""Transactional paper/control storage. MT5 attempt journals stay on the terminal host."""
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS engines(account TEXT PRIMARY KEY, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS mobile_license(id TEXT PRIMARY KEY,digest TEXT UNIQUE NOT NULL,
 account TEXT NOT NULL,scope TEXT NOT NULL,expires DOUBLE PRECISION,revoked INTEGER DEFAULT 0,
 device TEXT,login TEXT,server TEXT,generation INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS push_device(license TEXT PRIMARY KEY,token TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS alert_event(seq INTEGER PRIMARY KEY,account TEXT,alert_id TEXT,payload TEXT,
 UNIQUE(account,alert_id));
CREATE TABLE IF NOT EXISTS push_delivery(event BIGINT,license TEXT,generation INTEGER,
 due DOUBLE PRECISION DEFAULT 0,attempts INTEGER DEFAULT 0,PRIMARY KEY(event,license));
"""
_initialized=set()
_init_lock=threading.Lock()


class Row(dict):
    def __getitem__(self,key):
        return tuple(self.values())[key] if isinstance(key,int) else super().__getitem__(key)


def pg_rows(cursor):
    names=[col.name for col in cursor.description or ()]
    return lambda values: Row(zip(names,values))


class Connection:
    def __init__(self,raw,postgres):
        self.raw=raw
        self.postgres=postgres

    def execute(self,sql,args=()):
        return self.raw.execute(sql.replace("?","%s") if self.postgres else sql,args)


@contextmanager
def transaction(lock="licenses"):
    url=os.getenv("DATABASE_URL","").strip()
    if url:
        import psycopg
        raw=psycopg.connect(url,connect_timeout=10,prepare_threshold=None,row_factory=pg_rows)
    else:
        path=os.getenv("PAPER_STATE_DB",Path(__file__).with_name("paper_state.db"))
        raw=sqlite3.connect(path,timeout=10)
        raw.row_factory=sqlite3.Row
    try:
        if url:
            with _init_lock:
                if url not in _initialized:
                    raw.execute("SELECT pg_advisory_xact_lock(1947070601)")
                    raw.execute("CREATE SCHEMA IF NOT EXISTS trading_control")
                    raw.execute("SET LOCAL search_path TO trading_control")
                    for stmt in SCHEMA.replace("seq INTEGER PRIMARY KEY","seq BIGSERIAL PRIMARY KEY").split(";"):
                        if stmt.strip(): raw.execute(stmt)
                    raw.commit()
                    _initialized.add(url)
            raw.execute("SET LOCAL search_path TO trading_control")
            raw.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,7129))",(lock,))
        else:
            raw.executescript(SCHEMA)
            raw.execute("BEGIN IMMEDIATE")
        yield Connection(raw,bool(url))
        raw.commit()
    finally:
        raw.close()
