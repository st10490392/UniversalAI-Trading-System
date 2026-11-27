"""Fetch the public Forex Factory weekly calendar; cache is persisted per account."""
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
import requests


def fetch_week(now):
    response=requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json",timeout=(3,8))
    response.raise_for_status()
    data=response.json()
    local=datetime.fromtimestamp(now,ZoneInfo("America/New_York"))
    start=(local-timedelta(days=(local.weekday()+1)%7)).replace(hour=0,minute=0,second=0,microsecond=0)
    end=start+timedelta(days=7)
    events=[]
    if not isinstance(data,list) or not data: raise ValueError("Missing weekly calendar")
    for row in data:
        stamp=datetime.fromisoformat(row["date"])
        if stamp.tzinfo is None or not start<=stamp<end: raise ValueError("Unexpected calendar week")
        events.append(dict(timestamp=stamp.timestamp(),currency=row["country"],impact=row["impact"]))
    return dict(start=start.timestamp(),end=end.timestamp(),events=events)
