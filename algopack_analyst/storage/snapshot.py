"""Atomic snapshot — все данные тикера на один момент времени.

Решает проблему race conditions: scoring должен оперировать
консистентным набором фичей, а не "одно от 19:42, другое от 19:39".
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytz

from config import STOCK_TO_FUTURE, settings
from storage.repository import get_repo
from utils.logger import logger
from utils.moex_client import get_moex_client

MSK = pytz.timezone("Europe/Moscow")


@dataclass
class DataSnapshot:
    """Immutable snapshot of all data for a ticker at a single moment."""

    ticker: str
    snapshot_id: str  # uuid-like, для трассировки
    captured_at: datetime  # МСК

    # Core data
    ohlcv: dict[int, pd.DataFrame] = field(default_factory=dict)  # tf -> df
    super_candles: pd.DataFrame = field(default_factory=pd.DataFrame)
    obstats: pd.DataFrame = field(default_factory=pd.DataFrame)
    orderstats: pd.DataFrame = field(default_factory=pd.DataFrame)
    orderbook: dict[str, Any] | None = None
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    hi2: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts: pd.DataFrame = field(default_factory=pd.DataFrame)
    futoi: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Cross-asset
    index_ohlcv: dict[str, pd.DataFrame] = field(default_factory=dict)  # IMOEX, sector

    # Guards
    market_open: bool = False
    suspended: bool = False
    tradable_tqbr: bool = True

    # Freshness (seconds ago)
    freshness: dict[str, int] = field(default_factory=dict)

    # Data quality flags
    quality: dict[str, str] = field(default_factory=dict)  # 'ok' | 'stale' | 'missing'

    @property
    def has_minimal_data(self) -> bool:
        """True if we have enough data to compute meaningful signals."""
        return (
            not self.ohlcv.get(1, pd.DataFrame()).empty
            and self.orderbook is not None
        )

    @property
    def last_price(self) -> float | None:
        """Best estimate of current price."""
        if self.orderbook and self.orderbook.get("bids") and self.orderbook.get("asks"):
            best_bid = self.orderbook["bids"][0][0]
            best_ask = self.orderbook["asks"][0][0]
            return (best_bid + best_ask) / 2
        df1 = self.ohlcv.get(1)
        if df1 is not None and not df1.empty:
            return float(df1.sort_values("ts")["c"].iloc[-1])
        return None

    def freshness_for(self, key: str) -> int | None:
        """Return age in seconds of data block, or None if absent."""
        return self.freshness.get(key)


class SnapshotBuilder:
    """Build atomic snapshots by reading repository in parallel."""

    REQUIRED_OHLCV_TFS = (1, 5, 15, 60)  # 1m, 5m, 15m, 1h
    DAILY_TF = 24 * 60

    def __init__(self) -> None:
        self.repo = get_repo()
        self.client = get_moex_client()

    async def build(self, ticker: str, *, include_cross_asset: bool = True) -> DataSnapshot:
        """Build a snapshot for ticker — all components in parallel."""
        ticker = ticker.upper()
        t0 = time.perf_counter()
        captured = datetime.now(MSK)
        snap_id = f"{ticker}_{int(captured.timestamp() * 1000)}"

        # Gather guards in parallel
        market_open_task = self.client.is_market_open()
        suspended_task = self.client.is_trading_suspended(ticker)
        tradable_task = self.client.is_tradable_tqbr(ticker)

        # Pull repo data (DuckDB is sync but fast; wrap in to_thread for true parallelism)
        loop = asyncio.get_event_loop()

        async def _r(fn, *a, **kw):
            return await loop.run_in_executor(None, lambda: fn(*a, **kw))

        ohlcv_tasks = {
            tf: _r(self.repo.latest_ohlcv, ticker, tf, 300)
            for tf in self.REQUIRED_OHLCV_TFS
        }
        ohlcv_tasks[self.DAILY_TF] = _r(self.repo.latest_ohlcv, ticker, self.DAILY_TF, 60)

        sc_task = _r(self.repo.latest_super_candles, ticker, 30)
        obs_task = _r(self.repo.latest_obstats, ticker, 10)
        ord_task = _r(self.repo.latest_orderstats, ticker, 10)
        ob_task = _r(self.repo.latest_orderbook, ticker)
        hi2_task = _r(self.repo.latest_hi2, ticker)
        alerts_task = _r(self.repo.alerts_window, ticker, 120)

        fut_ticker = STOCK_TO_FUTURE.get(ticker)
        futoi_task = (
            _r(self.repo.latest_futoi, fut_ticker, 240)
            if fut_ticker else None
        )

        index_tasks = {}
        if include_cross_asset:
            index_tasks = {
                "IMOEX": _r(self.repo.latest_ohlcv, "IMOEX", 60, 100),
            }

        # Await everything
        results = await asyncio.gather(
            market_open_task, suspended_task, tradable_task,
            *ohlcv_tasks.values(),
            sc_task, obs_task, ord_task, ob_task, hi2_task, alerts_task,
            *([futoi_task] if futoi_task else []),
            *index_tasks.values(),
            return_exceptions=True,
        )

        # Unpack
        idx = 0
        market_open = _ok(results[idx], False); idx += 1
        suspended = _ok(results[idx], False); idx += 1
        tradable = _ok(results[idx], True); idx += 1

        ohlcv: dict[int, pd.DataFrame] = {}
        for tf in ohlcv_tasks.keys():
            ohlcv[tf] = _ok(results[idx], pd.DataFrame()); idx += 1

        sc = _ok(results[idx], pd.DataFrame()); idx += 1
        obs = _ok(results[idx], pd.DataFrame()); idx += 1
        ords = _ok(results[idx], pd.DataFrame()); idx += 1
        ob = _ok(results[idx], None); idx += 1
        hi2 = _ok(results[idx], pd.DataFrame()); idx += 1
        alerts = _ok(results[idx], pd.DataFrame()); idx += 1

        futoi = pd.DataFrame()
        if futoi_task:
            futoi = _ok(results[idx], pd.DataFrame()); idx += 1

        index_ohlcv = {}
        for name in index_tasks.keys():
            index_ohlcv[name] = _ok(results[idx], pd.DataFrame()); idx += 1

        snap = DataSnapshot(
            ticker=ticker,
            snapshot_id=snap_id,
            captured_at=captured,
            ohlcv=ohlcv,
            super_candles=sc,
            obstats=obs,
            orderstats=ords,
            orderbook=ob,
            hi2=hi2,
            alerts=alerts,
            futoi=futoi,
            index_ohlcv=index_ohlcv,
            market_open=market_open,
            suspended=suspended,
            tradable_tqbr=tradable,
        )
        snap.freshness = _compute_freshness(snap, captured)
        snap.quality = _compute_quality(snap)

        dt = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"snapshot built ticker={ticker} id={snap_id} "
            f"dt={dt:.0f}ms quality={snap.quality}"
        )
        return snap


def _ok(val: Any, default: Any) -> Any:
    if isinstance(val, Exception):
        return default
    return val


def _compute_freshness(snap: DataSnapshot, now: datetime) -> dict[str, int]:
    """Compute age (seconds) of each data block."""
    now_naive = now.astimezone(MSK).replace(tzinfo=None)
    out: dict[str, int] = {}

    def _age(ts) -> int:
        try:
            t = pd.to_datetime(ts)
            if pd.isna(t):
                return -1
            tn = t.to_pydatetime()
            if tn.tzinfo:
                tn = tn.astimezone(MSK).replace(tzinfo=None)
            return int((now_naive - tn).total_seconds())
        except Exception:
            return -1

    for tf, df in snap.ohlcv.items():
        if not df.empty and "ts" in df.columns:
            out[f"ohlcv_{tf}m"] = _age(df["ts"].max())

    if not snap.super_candles.empty:
        out["super_candles"] = _age(snap.super_candles["ts"].max())
    if not snap.obstats.empty:
        out["obstats"] = _age(snap.obstats["ts"].max())
    if snap.orderbook:
        out["orderbook"] = _age(snap.orderbook.get("ts"))
    if not snap.alerts.empty:
        out["alerts"] = _age(snap.alerts["ts"].max())
    if not snap.futoi.empty:
        out["futoi"] = _age(snap.futoi["ts"].max())
    if not snap.hi2.empty and "date" in snap.hi2.columns:
        try:
            d = pd.to_datetime(snap.hi2["date"].iloc[0]).date()
            out["hi2"] = int((datetime.now(MSK).date() - d).total_seconds() * 86400)
        except Exception as e:
            logger.debug(f"_compute_freshness: failed to parse hi2 date: {e}")
    return out


def _compute_quality(snap: DataSnapshot) -> dict[str, str]:
    """Mark each block as ok/stale/missing based on freshness thresholds."""
    q: dict[str, str] = {}

    thresholds = {
        "ohlcv_1m": 90, "ohlcv_5m": 600, "ohlcv_15m": 1500,
        "ohlcv_60m": 3900, "super_candles": settings.FRESH_SUPER_CANDLES,
        "obstats": settings.FRESH_SUPER_CANDLES,
        "orderbook": settings.FRESH_ORDERBOOK,
        "alerts": settings.FRESH_ALERTS, "futoi": 900,
        "hi2": 36 * 3600,  # 1.5 days
    }
    for k, thr in thresholds.items():
        age = snap.freshness.get(k)
        if age is None or age < 0:
            q[k] = "missing"
        elif age > thr:
            q[k] = "stale"
        else:
            q[k] = "ok"
    return q


_builder: SnapshotBuilder | None = None


def get_snapshot_builder() -> SnapshotBuilder:
    global _builder
    if _builder is None:
        _builder = SnapshotBuilder()
    return _builder