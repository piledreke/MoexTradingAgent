"""Normalisers that map raw MOEX rows to repository-friendly dicts.

ISS sometimes returns data with mixed casing or string-encoded numbers. These
helpers coerce field names to lowercase and parse datetimes once, so the
downstream storage layer can write directly.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional

from app.utils.time import merge_date_time, parse_moex_datetime


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _lowercase_keys(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in row.items()}


def _maybe_reference(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _build_ts(row: Mapping[str, Any]) -> Optional[datetime]:
    if row.get("ts"):
        return parse_moex_datetime(row["ts"])
    td = row.get("tradedate")
    tt = row.get("tradetime")
    if td:
        return merge_date_time(str(td), str(tt) if tt else None)
    return None


# ---------------------------------------------------------------------------
def normalize_tradestats(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in rows:
        r = _lowercase_keys(raw)
        secid = r.get("secid")
        if not secid:
            continue
        out.append({
            "secid": str(secid).upper(),
            "tradedate": r.get("tradedate"),
            "tradetime": r.get("tradetime"),
            "ts": _build_ts(r),
            "pr_open": _to_float(r.get("pr_open")),
            "pr_high": _to_float(r.get("pr_high")),
            "pr_low": _to_float(r.get("pr_low")),
            "pr_close": _to_float(r.get("pr_close")),
            "pr_std": _to_float(r.get("pr_std")),
            "vol": _to_float(r.get("vol")),
            "val": _to_float(r.get("val")),
            "trades": _to_int(r.get("trades")),
            "pr_vwap": _to_float(r.get("pr_vwap")),
            "pr_change": _to_float(r.get("pr_change")),
            "trades_b": _to_int(r.get("trades_b")),
            "trades_s": _to_int(r.get("trades_s")),
            "val_b": _to_float(r.get("val_b")),
            "val_s": _to_float(r.get("val_s")),
            "vol_b": _to_float(r.get("vol_b")),
            "vol_s": _to_float(r.get("vol_s")),
            "disb": _to_float(r.get("disb")),
            "pr_vwap_b": _to_float(r.get("pr_vwap_b")),
            "pr_vwap_s": _to_float(r.get("pr_vwap_s")),
            "systime": parse_moex_datetime(r.get("systime")),
        })
    return out


def normalize_orderstats(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    fields = (
        "put_orders_b", "put_orders_s", "put_val_b", "put_val_s", "put_vol_b",
        "put_vol_s", "put_vwap_b", "put_vwap_s", "put_vol", "put_val",
        "put_orders", "cancel_orders_b", "cancel_orders_s", "cancel_val_b",
        "cancel_val_s", "cancel_vol_b", "cancel_vol_s", "cancel_vwap_b",
        "cancel_vwap_s", "cancel_vol", "cancel_val", "cancel_orders",
    )
    for raw in rows:
        r = _lowercase_keys(raw)
        secid = r.get("secid")
        if not secid:
            continue
        entry: Dict[str, Any] = {
            "secid": str(secid).upper(),
            "tradedate": r.get("tradedate"),
            "tradetime": r.get("tradetime"),
            "ts": _build_ts(r),
            "systime": parse_moex_datetime(r.get("systime")),
        }
        for f in fields:
            entry[f] = _to_float(r.get(f))
        out.append(entry)
    return out


def normalize_obstats(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    fields = (
        "spread_bbo", "spread_lv10", "spread_1mio", "levels_b", "levels_s",
        "vol_b", "vol_s", "val_b", "val_s",
        "imbalance_vol_bbo", "imbalance_val_bbo", "imbalance_vol", "imbalance_val",
        "vwap_b", "vwap_s", "vwap_b_1mio", "vwap_s_1mio",
    )
    for raw in rows:
        r = _lowercase_keys(raw)
        secid = r.get("secid")
        if not secid:
            continue
        entry: Dict[str, Any] = {
            "secid": str(secid).upper(),
            "tradedate": r.get("tradedate"),
            "tradetime": r.get("tradetime"),
            "ts": _build_ts(r),
            "systime": parse_moex_datetime(r.get("systime")),
        }
        for f in fields:
            entry[f] = _to_float(r.get(f))
        out.append(entry)
    return out


def normalize_alerts(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in rows:
        r = _lowercase_keys(raw)
        secid = r.get("secid")
        alert_type = r.get("alert_type") or r.get("alert")
        if not secid or not alert_type:
            continue
        out.append({
            "secid": str(secid).upper(),
            "tradedate": r.get("tradedate"),
            "tradetime": r.get("tradetime"),
            "ts": _build_ts(r),
            "alert_type": str(alert_type),
            "threshold": _to_float(r.get("threshold")),
            "value": _to_float(r.get("value")),
            "reference": _maybe_reference(r.get("reference")),
            "systime": parse_moex_datetime(r.get("systime")),
        })
    return out


def normalize_hi2(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in rows:
        r = _lowercase_keys(raw)
        secid = r.get("secid")
        metric = r.get("metric")
        if not secid or not metric:
            continue
        out.append({
            "secid": str(secid).upper(),
            "tradedate": r.get("tradedate"),
            "tradetime": r.get("tradetime"),
            "metric": str(metric),
            "value": _to_float(r.get("value")),
            "reference": _maybe_reference(r.get("reference")),
            "systime": parse_moex_datetime(r.get("systime")),
        })
    return out


def normalize_candles(
    rows: Iterable[Mapping[str, Any]], secid: str, interval_min: int = 10
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    secid_u = secid.upper()
    for raw in rows:
        r = _lowercase_keys(raw)
        begin = parse_moex_datetime(r.get("begin"))
        if begin is None:
            continue
        out.append({
            "secid": secid_u,
            "begin": begin,
            "end": parse_moex_datetime(r.get("end")),
            "interval_min": int(interval_min),
            "open": _to_float(r.get("open")),
            "high": _to_float(r.get("high")),
            "low": _to_float(r.get("low")),
            "close": _to_float(r.get("close")),
            "volume": _to_float(r.get("volume")),
            "value": _to_float(r.get("value")),
        })
    return out


def normalize_marketdata(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in rows:
        r = _lowercase_keys(raw)
        secid = r.get("secid")
        if not secid:
            continue
        out.append({
            "secid": str(secid).upper(),
            "last": _to_float(r.get("last")),
            "bid": _to_float(r.get("bid")),
            "offer": _to_float(r.get("offer")),
            "spread": _to_float(r.get("spread")),
            "open": _to_float(r.get("open")),
            "high": _to_float(r.get("high")),
            "low": _to_float(r.get("low")),
            "last_change_pct": _to_float(r.get("lastchangeprcnt")),
            "voltoday": _to_float(r.get("voltoday")),
            "valtoday": _to_float(r.get("valtoday")),
            "waprice": _to_float(r.get("waprice")),
            "numtrades": _to_int(r.get("numtrades")),
            "updatetime": r.get("updatetime"),
            "systime": parse_moex_datetime(r.get("systime")),
            "lotsize": _to_int(r.get("lotsize")),
            "prevprice": _to_float(r.get("prevprice")),
            "decimals": _to_int(r.get("decimals")),
            "raw_json": json.dumps(dict(r), default=str),
        })
    return out
