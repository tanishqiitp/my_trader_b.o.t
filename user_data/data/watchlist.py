"""Watchlist helpers for manual stocks and broker-imported holdings."""

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

ASSETS_FILE = os.path.join(os.path.dirname(__file__), "monitored_assets.json")


def load_watchlist() -> Dict[str, Dict[str, Any]]:
    with open(ASSETS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_watchlist(watchlist: Dict[str, Dict[str, Any]]) -> None:
    ordered = {key: watchlist[key] for key in sorted(watchlist.keys())}
    with open(ASSETS_FILE, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)


def normalize_key(value: str) -> str:
    cleaned = value.strip().upper()
    for suffix in (".NS", ".BO", ".NSE", ".BSE"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned.replace(" ", "_")


def normalize_exchange_symbol(value: str) -> str:
    cleaned = value.strip().upper()
    if cleaned.endswith(".NSE"):
        return cleaned.replace(".NSE", ".NS")
    if cleaned.endswith(".BSE"):
        return cleaned.replace(".BSE", ".BO")
    if cleaned.endswith(".NS") or cleaned.endswith(".BO"):
        return cleaned
    return f"{cleaned}.NS"


def _unique_values(values: Iterable[str]) -> List[str]:
    seen = set()
    unique = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            unique.append(cleaned)
    return unique


def parse_manual_entries(raw_text: str) -> List[Dict[str, Any]]:
    """Parse comma/newline stocks like `TCS`, `RELIANCE.NS`, or `Tata Motors:TATAMOTORS.NS`."""
    if not raw_text:
        return []

    entries = []
    chunks = []
    for line in raw_text.replace(",", "\n").splitlines():
        item = line.strip()
        if item:
            chunks.append(item)

    for item in chunks:
        if ":" in item:
            name, symbol = [part.strip() for part in item.split(":", 1)]
        else:
            name, symbol = item.strip(), item.strip()

        key = normalize_key(symbol)
        exchange_symbol = normalize_exchange_symbol(symbol)
        entries.append(
            {
                "key": key,
                "config": {
                    "type": "stock",
                    "symbol": exchange_symbol,
                    "name": name.upper() if name == symbol else name,
                    "aliases": _unique_values(
                        [key, name, exchange_symbol.replace(".NS", "").replace(".BO", "")]
                    ),
                    "source": "manual",
                },
            }
        )
    return entries


def upsert_manual_entries(raw_text: str, persist: bool = True) -> Dict[str, Dict[str, Any]]:
    watchlist = load_watchlist()
    for entry in parse_manual_entries(raw_text):
        key = entry["key"]
        existing = watchlist.get(key, {})
        merged = {**existing, **entry["config"]}
        merged["aliases"] = sorted(set(existing.get("aliases", []) + entry["config"].get("aliases", [])))
        watchlist[key] = merged
    if persist:
        save_watchlist(watchlist)
    return watchlist


def replace_with_manual_entries(raw_text: str, persist: bool = True) -> Dict[str, Dict[str, Any]]:
    watchlist: Dict[str, Dict[str, Any]] = {}
    for entry in parse_manual_entries(raw_text):
        watchlist[entry["key"]] = entry["config"]
    if persist:
        save_watchlist(watchlist)
    return watchlist


def _read_symbols_from_csv(path: str) -> List[Tuple[str, str]]:
    symbols: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (
                row.get("tradingsymbol")
                or row.get("Trading Symbol")
                or row.get("symbol")
                or row.get("Symbol")
                or row.get("ticker")
                or row.get("Ticker")
            )
            name = row.get("name") or row.get("Name") or row.get("instrument") or row.get("Instrument") or symbol
            if symbol:
                symbols.append((str(name), str(symbol)))
    return symbols


def _read_symbols_from_json(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("holdings", data) if isinstance(data, dict) else data
    symbols: List[Tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = row.get("tradingsymbol") or row.get("symbol") or row.get("ticker")
        name = row.get("name") or row.get("instrument") or symbol
        if symbol:
            symbols.append((str(name), str(symbol)))
    return symbols


def load_broker_export(provider: str) -> Dict[str, Any]:
    """Load broker holdings from an exported CSV/JSON file.

    Live broker APIs need account-specific credentials and approvals. This
    function gives the app a safe bridge today: export holdings/watchlist from
    Zerodha/Groww, point an env var to the file, and the dashboard can analyze it.
    """
    provider_key = provider.strip().upper()
    env_name = f"{provider_key}_HOLDINGS_FILE"
    path = os.environ.get(env_name)
    if not path:
        return {
            "status": "not_configured",
            "message": f"Set {env_name} to a Zerodha/Groww CSV or JSON export to import holdings.",
            "watchlist": load_watchlist(),
        }

    if not os.path.exists(path):
        return {
            "status": "error",
            "message": f"{env_name} points to a missing file: {path}",
            "watchlist": load_watchlist(),
        }

    if path.lower().endswith(".json"):
        symbols = _read_symbols_from_json(path)
    else:
        symbols = _read_symbols_from_csv(path)

    raw_text = "\n".join(f"{name}:{symbol}" for name, symbol in symbols)
    watchlist = replace_with_manual_entries(raw_text, persist=True)
    return {
        "status": "imported",
        "provider": provider.lower(),
        "count": len(watchlist),
        "watchlist": watchlist,
    }
