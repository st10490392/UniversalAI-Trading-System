"""Download demo-account MT5 M15 history for research (2020-01-01..2025-12-31).

MT5 only returns bars currently available in the connected terminal's history.
Missing early bars are recorded as the earliest date the broker supplied.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data" / "history_2020_2025"
START = datetime(2020, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 1, 1, tzinfo=timezone.utc)  # exclusive; includes 2025-12-31
FX = ("EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "AUDUSD", "USDCAD", "XAUUSD")
INDEX_ALIASES = {
    "US100": ("us100", "nas100", "ustec", "nasdaq"),
    "US30": ("us30", "dj30", "wallstreet30", "dowjones"),
    "GER40": ("ger40", "de40", "ger30", "de30", "germany40", "germany30", "dax"),
}
SYNTHETIC_MARKERS = ("volatility", "boom", "crash", "stepindex", "skewstep", "jump", "rangebreak")


def classify_symbol(name: str) -> tuple[str, str] | None:
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    for pair in FX:
        if pair.lower() in compact:
            return ("forex", "forex")
    for family, aliases in INDEX_ALIASES.items():
        if any(alias in compact for alias in aliases):
            return ("index", family)
    if any(marker in compact for marker in SYNTHETIC_MARKERS):
        return ("synthetic", "synthetic")
    return None


def safe_name(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default=START.isoformat())
    parser.add_argument("--end", default=END.isoformat(), help="exclusive end timestamp")
    parser.add_argument("--symbols", nargs="*", help="optional exact Market Watch symbol names")
    args = parser.parse_args()
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package is unavailable; install requirements.txt", file=sys.stderr)
        return 2
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(timezone.utc)
    if start >= end:
        parser.error("--start must be before --end")
    if not mt5.initialize(timeout=15000):
        print(f"MT5 initialize failed (code {mt5.last_error()[0]})", file=sys.stderr)
        return 3
    try:
        account = mt5.account_info()
        if account is None or account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            print("Refusing to harvest: the connected MT5 account is not a demo account.", file=sys.stderr)
            return 4
        visible = sorted(s.name for s in (mt5.symbols_get() or ()) if s.visible)
        chosen = {s: classify_symbol(s) for s in visible if classify_symbol(s)}
        compact_visible={re.sub(r"[^a-z0-9]","",s.lower()) for s in visible}
        missing_requested=[]
        for pair in FX:
            if not any(pair.lower() in name for name in compact_visible):
                missing_requested.append({"family":"forex","symbol":pair,"reason":"not visible in Market Watch"})
        for family,aliases in INDEX_ALIASES.items():
            if not any(any(alias in name for alias in aliases) for name in compact_visible):
                missing_requested.append({"family":"index","symbol":family,"aliases":list(aliases),"reason":"not visible in Market Watch"})
        for family,aliases in {"Volatility":("volatility",),"Boom":("boom",),"Crash":("crash",),"Step Index":("stepindex","skewstep"),"Jump":("jump",),"Range Break":("rangebreak",)}.items():
            if not any(any(alias in name for alias in aliases) for name in compact_visible):
                missing_requested.append({"family":"synthetic","symbol":family,"reason":"not visible in Market Watch"})
        for missing in missing_requested:
            print(f"SKIP requested {missing['family']} {missing['symbol']}: not visible in Market Watch")
        if args.symbols is not None:
            requested = set(args.symbols)
            for name in sorted(requested - set(visible)):
                print(f"SKIP {name}: not in Market Watch")
            chosen = {s: meta for s, meta in chosen.items() if s in requested}
        if not chosen:
            print("No matching visible symbols found.", file=sys.stderr)
            return 5
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"period_start_utc": start.isoformat(), "period_end_exclusive_utc": end.isoformat(), "timeframe": "M15", "account_mode": "demo", "market_watch_matches":len(chosen), "missing_requested_market_watch":missing_requested, "symbols": [], "skipped_symbols": []}
        from datetime import timedelta
        for symbol, (asset_class, family) in chosen.items():
            info = mt5.symbol_info(symbol)
            if info is None:
                print(f"SKIP {symbol}: symbol_info unavailable")
                manifest["skipped_symbols"].append({"symbol":symbol,"reason":"symbol_info unavailable"})
                continue
            dest = args.output_dir / f"{safe_name(symbol)}_M15_2020_2025.csv"
            count, earliest, latest, last_ts = 0, None, None, -1
            try:
                with dest.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("symbol", "asset_class", "family", "time", "open", "high", "low", "close", "tick_volume", "real_volume", "spread_points", "point_size", "tick_size", "tick_value"))
                    cursor = start
                    while cursor < end:
                        chunk_end = min(cursor + timedelta(days=180), end)
                        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, cursor, chunk_end)
                        if rates is not None and len(rates):
                            # Write each half-year chunk immediately to bound RAM use.
                            ordered = sorted(rates, key=lambda r: int(r["time"]))
                            for r in ordered:
                                ts = int(r["time"])
                                if ts <= last_ts or ts < int(start.timestamp()) or ts >= int(end.timestamp()):
                                    continue
                                stamp = datetime.fromtimestamp(ts, timezone.utc).isoformat()
                                writer.writerow((symbol, asset_class, family, stamp, float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), int(r["tick_volume"]), int(r["real_volume"]), int(r["spread"]), float(info.point), float(info.trade_tick_size), float(info.trade_tick_value)))
                                count += 1
                                last_ts = ts
                                earliest = earliest or stamp
                                latest = stamp
                        cursor = chunk_end
                item = {"symbol": symbol, "asset_class": asset_class, "family": family, "bars": count, "earliest_utc": earliest, "latest_utc": latest, "file": dest.name}
                if count == 0:
                    dest.unlink(missing_ok=True)
                    print(f"SKIP {symbol}: broker returned no M15 bars for requested period")
                    manifest["skipped_symbols"].append({"symbol":symbol,"reason":"broker returned no bars in requested interval"})
                else:
                    manifest["symbols"].append(item)
                    print(f"EXPORTED {symbol} [{asset_class}/{family}]: {count:,} bars, {earliest} to {latest}", flush=True)
            except Exception as exc:
                dest.unlink(missing_ok=True)
                print(f"SKIP {symbol}: {type(exc).__name__}: {exc}", file=sys.stderr)
                manifest["skipped_symbols"].append({"symbol":symbol,"reason":f"{type(exc).__name__}: {exc}"})
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"HARVEST_COMPLETE symbols={len(manifest['symbols'])} bars={sum(i['bars'] for i in manifest['symbols']):,} output={args.output_dir}")
        return 0 if manifest["symbols"] else 6
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
