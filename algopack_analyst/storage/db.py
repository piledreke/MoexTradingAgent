"""DuckDB connection + schema initialization."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import duckdb

from config import settings
from utils.logger import logger

_lock = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None
_safe_conn: "SafeDuckDBConnection" | None = None


class SafeDuckDBResult:
    def __init__(self, result: Any, lock: threading.Lock) -> None:
        self._result = result
        self._lock = lock

    @property
    def description(self) -> Any:
        return self._result.description

    def fetch_df(self) -> Any:
        try:
            return self._result.fetch_df()
        finally:
            self._lock.release()

    def fetchone(self) -> Any:
        try:
            return self._result.fetchone()
        finally:
            self._lock.release()

    def fetchall(self) -> Any:
        try:
            return self._result.fetchall()
        finally:
            self._lock.release()


class SafeDuckDBConnection:
    def __init__(self, conn: duckdb.DuckDBPyConnection, lock: threading.Lock) -> None:
        self._conn = conn
        self._lock = lock

    def execute(self, *args: Any, **kwargs: Any) -> SafeDuckDBResult:
        self._lock.acquire()
        try:
            result = self._conn.execute(*args, **kwargs)
        except Exception:
            self._lock.release()
            raise
        return SafeDuckDBResult(result, self._lock)

    def executemany(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._conn.executemany(*args, **kwargs)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._conn, item)


SCHEMA_SQL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS super_candles_eq (
        ticker        VARCHAR NOT NULL,
        ts            TIMESTAMP NOT NULL,
        pr_open       DOUBLE, pr_high DOUBLE, pr_low DOUBLE, pr_close DOUBLE,
        pr_vwap       DOUBLE, pr_change DOUBLE,
        vol           DOUBLE, val DOUBLE, trades_count BIGINT,
        buy_vol       DOUBLE, sell_vol DOUBLE,
        buy_val       DOUBLE, sell_val DOUBLE,
        disb          DOUBLE, pr_vwap_b DOUBLE, pr_vwap_s DOUBLE,
        raw           JSON,
        PRIMARY KEY (ticker, ts)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS super_candles_fo (
        ticker VARCHAR NOT NULL, ts TIMESTAMP NOT NULL,
        pr_open DOUBLE, pr_high DOUBLE, pr_low DOUBLE, pr_close DOUBLE,
        pr_vwap DOUBLE, vol DOUBLE, val DOUBLE, trades_count BIGINT,
        buy_vol DOUBLE, sell_vol DOUBLE, disb DOUBLE,
        raw JSON,
        PRIMARY KEY (ticker, ts)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS obstats_eq (
        ticker VARCHAR NOT NULL, ts TIMESTAMP NOT NULL,
        spread_bbo DOUBLE, levels_b DOUBLE, levels_s DOUBLE,
        vol_b DOUBLE, vol_s DOUBLE, val_b DOUBLE, val_s DOUBLE,
        imbalance DOUBLE, micro_price DOUBLE,
        raw JSON,
        PRIMARY KEY (ticker, ts)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS orderstats_eq (
        ticker VARCHAR NOT NULL, ts TIMESTAMP NOT NULL,
        put_orders_b DOUBLE, put_orders_s DOUBLE,
        cancel_orders_b DOUBLE, cancel_orders_s DOUBLE,
        put_val_b DOUBLE, put_val_s DOUBLE,
        cancel_val_b DOUBLE, cancel_val_s DOUBLE,
        raw JSON,
        PRIMARY KEY (ticker, ts)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS futoi (
        ticker VARCHAR NOT NULL, ts TIMESTAMP NOT NULL,
        clgroup VARCHAR NOT NULL,
        pos_long DOUBLE, pos_short DOUBLE,
        pos_long_num BIGINT, pos_short_num BIGINT,
        raw JSON,
        PRIMARY KEY (ticker, ts, clgroup)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS hi2 (
        ticker VARCHAR NOT NULL, date DATE NOT NULL, market VARCHAR NOT NULL,
        hhi_volume DOUBLE, hhi_buy DOUBLE, hhi_sell DOUBLE,
        hhi_netflow_buy DOUBLE, hhi_netflow_sell DOUBLE,
        hhi_passive DOUBLE, hhi_active DOUBLE,
        hhi_passive_buy DOUBLE, hhi_active_buy DOUBLE,
        hhi_passive_sell DOUBLE, hhi_active_sell DOUBLE,
        raw JSON,
        PRIMARY KEY (ticker, date, market)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS mega_alerts (
        ticker VARCHAR NOT NULL, ts TIMESTAMP NOT NULL,
        alert_type VARCHAR, side VARCHAR,
        magnitude DOUBLE, severity VARCHAR,
        description VARCHAR, raw JSON,
        PRIMARY KEY (ticker, ts, alert_type)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS ohlcv (
        ticker VARCHAR NOT NULL, ts TIMESTAMP NOT NULL, timeframe INTEGER NOT NULL,
        o DOUBLE, h DOUBLE, l DOUBLE, c DOUBLE, v DOUBLE, value DOUBLE,
        PRIMARY KEY (ticker, ts, timeframe)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS orderbook_snapshots (
        ticker VARCHAR NOT NULL, ts TIMESTAMP NOT NULL,
        bids_json JSON, asks_json JSON,
        best_bid DOUBLE, best_ask DOUBLE,
        imbalance DOUBLE,
        PRIMARY KEY (ticker, ts)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS trades_log (
        ticker VARCHAR NOT NULL, ts TIMESTAMP NOT NULL,
        tradeno BIGINT, price DOUBLE, qty DOUBLE,
        direction VARCHAR,
        PRIMARY KEY (ticker, tradeno)
    );
    """,
    "CREATE SEQUENCE IF NOT EXISTS seq_agent_queries START 1;",
    """
    CREATE TABLE IF NOT EXISTS agent_queries (
        id BIGINT DEFAULT nextval('seq_agent_queries'),
        ts TIMESTAMP NOT NULL,
        query_text VARCHAR,
        response_json JSON,
        recommendation VARCHAR,
        latency_ms DOUBLE,
        PRIMARY KEY (id)
    );
    """,
    # Indexes
    "CREATE INDEX IF NOT EXISTS idx_sc_eq_ticker_ts ON super_candles_eq (ticker, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_obs_eq_ticker_ts ON obstats_eq (ticker, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ord_eq_ticker_ts ON orderstats_eq (ticker, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_ticker_ts ON mega_alerts (ticker, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_ts ON ohlcv (ticker, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ob_ticker_ts ON orderbook_snapshots (ticker, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_futoi_ticker_ts ON futoi (ticker, ts DESC);",
]


def get_conn() -> SafeDuckDBConnection:
    """Return shared DuckDB connection (lazy initialization)."""
    global _conn, _safe_conn
    with _lock:
        if _conn is None:
            Path(settings.DUCKDB_PATH).parent.mkdir(parents=True, exist_ok=True)
            _conn = duckdb.connect(settings.DUCKDB_PATH)
            _conn.execute("PRAGMA threads=4;")
            _conn.execute("PRAGMA memory_limit='4GB';")
            _init_schema(_conn)
            logger.info(f"DuckDB connected: {settings.DUCKDB_PATH}")
        if _safe_conn is None:
            _safe_conn = SafeDuckDBConnection(_conn, _lock)
    return _safe_conn


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    # sequence must be created before tables that use it
    for stmt in SCHEMA_SQL:
        try:
            conn.execute(stmt)
        except Exception as e:
            logger.error(f"DDL failed: {e} | sql={stmt[:80]}")


def close_db() -> None:
    global _conn, _safe_conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
            _safe_conn = None