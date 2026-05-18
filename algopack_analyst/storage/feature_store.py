"""Feature store — pre-computed snapshots of indicators for fast scoring.

Update strategy:
  - 1m features: recomputed every minute by scheduler
  - 5m/15m: every 5/15 minutes
  - 1h/1d: hourly
TTL-кэш в памяти + DuckDB persistence.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from storage.db import get_conn
from utils.logger import logger


@dataclass
class FeatureSnapshot:
    ticker: str
    timeframe: int
    ts: datetime
    features: dict[str, float]


class FeatureStore:
    """Persistent feature store backed by DuckDB."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], tuple[float, FeatureSnapshot]] = {}
        self._cache_ttl = 60  # seconds
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_snapshots (
                ticker VARCHAR NOT NULL,
                timeframe INTEGER NOT NULL,
                ts TIMESTAMP NOT NULL,
                features JSON,
                PRIMARY KEY (ticker, timeframe, ts)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fs_ticker_tf_ts "
            "ON feature_snapshots (ticker, timeframe, ts DESC)"
        )

    def save(self, snap: FeatureSnapshot) -> None:
        conn = get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO feature_snapshots (ticker, timeframe, ts, features)
               VALUES (?, ?, ?, ?)""",
            (snap.ticker, snap.timeframe, snap.ts,
             json.dumps(snap.features, default=str)),
        )
        self._cache[(snap.ticker, snap.timeframe)] = (time.time(), snap)

    def get_latest(self, ticker: str, timeframe: int) -> FeatureSnapshot | None:
        key = (ticker.upper(), timeframe)
        if key in self._cache:
            ts, snap = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return snap

        conn = get_conn()
        row = conn.execute(
            """SELECT ts, features FROM feature_snapshots
               WHERE ticker=? AND timeframe=?
               ORDER BY ts DESC LIMIT 1""",
            (ticker.upper(), timeframe),
        ).fetchone()
        if not row:
            return None
        snap = FeatureSnapshot(
            ticker=ticker.upper(), timeframe=timeframe,
            ts=row[0], features=json.loads(row[1]),
        )
        self._cache[key] = (time.time(), snap)
        return snap

    def get_history(
        self, ticker: str, timeframe: int, n: int = 50
    ) -> list[FeatureSnapshot]:
        conn = get_conn()
        rows = conn.execute(
            """SELECT ts, features FROM feature_snapshots
               WHERE ticker=? AND timeframe=?
               ORDER BY ts DESC LIMIT ?""",
            (ticker.upper(), timeframe, n),
        ).fetchall()
        return [
            FeatureSnapshot(ticker.upper(), timeframe, r[0], json.loads(r[1]))
            for r in rows
        ]


_store: FeatureStore | None = None


def get_feature_store() -> FeatureStore:
    global _store
    if _store is None:
        _store = FeatureStore()
    return _store


async def compute_and_store_features(
    ticker: str, ohlcv_by_tf: dict[int, pd.DataFrame]
) -> None:
    """Compute features for each TF and persist."""
    from analytics.technical import compute_all_features

    store = get_feature_store()
    for tf, df in ohlcv_by_tf.items():
        if df.empty or len(df) < 30:
            continue
        sorted_df = df.sort_values("ts") if "ts" in df.columns else df
        try:
            feats = compute_all_features(sorted_df)
            if not feats:
                continue
            store.save(FeatureSnapshot(
                ticker=ticker,
                timeframe=tf,
                ts=datetime.utcnow(),
                features=feats,
            ))
        except Exception as e:
            logger.warning(f"feature compute failed ticker={ticker} tf={tf}: {e}")