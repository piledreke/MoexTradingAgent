"""CRUD + analytical queries over DuckDB."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Iterable

import pandas as pd

from storage.db import get_conn
from utils.logger import logger


class Repository:
    """Data-access layer for analyst storage."""

    # ─── Bulk insert helpers ────────────────────────────
    def _upsert_df(self, table: str, df: pd.DataFrame, pk: list[str]) -> None:
        if df.empty:
            return
        conn = get_conn()
        cols = list(df.columns)
        placeholders = ", ".join(["?"] * len(cols))
        col_list = ", ".join(cols)
        # DuckDB: INSERT OR REPLACE
        sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
        records = [tuple(_clean(v) for v in row) for row in df[cols].itertuples(index=False, name=None)]
        try:
            conn.executemany(sql, records)
        except Exception as e:
            logger.error(f"upsert {table} failed: {e}")

    # ─── Super candles ──────────────────────────────────
    def save_super_candles_eq(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        target_cols = [
            "ticker", "ts", "pr_open", "pr_high", "pr_low", "pr_close",
            "pr_vwap", "pr_change", "vol", "val", "trades_count",
            "buy_vol", "sell_vol", "buy_val", "sell_val",
            "disb", "pr_vwap_b", "pr_vwap_s", "raw",
        ]
        out = _prepare_df(df, target_cols, ts_cols=["ts"])
        self._upsert_df("super_candles_eq", out, pk=["ticker", "ts"])
        return len(out)

    def save_super_candles_fo(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        target_cols = [
            "ticker", "ts", "pr_open", "pr_high", "pr_low", "pr_close",
            "pr_vwap", "vol", "val", "trades_count",
            "buy_vol", "sell_vol", "disb", "raw",
        ]
        out = _prepare_df(df, target_cols, ts_cols=["ts"])
        self._upsert_df("super_candles_fo", out, pk=["ticker", "ts"])
        return len(out)

    def save_obstats(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        target_cols = [
            "ticker", "ts", "spread_bbo", "levels_b", "levels_s",
            "vol_b", "vol_s", "val_b", "val_s", "imbalance", "micro_price", "raw",
        ]
        out = _prepare_df(df, target_cols, ts_cols=["ts"])
        self._upsert_df("obstats_eq", out, pk=["ticker", "ts"])
        return len(out)

    def save_orderstats(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        target_cols = [
            "ticker", "ts", "put_orders_b", "put_orders_s",
            "cancel_orders_b", "cancel_orders_s",
            "put_val_b", "put_val_s", "cancel_val_b", "cancel_val_s", "raw",
        ]
        out = _prepare_df(df, target_cols, ts_cols=["ts"])
        self._upsert_df("orderstats_eq", out, pk=["ticker", "ts"])
        return len(out)

    # ─── FUTOI ──────────────────────────────────────────
    def save_futoi(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        target_cols = [
            "ticker", "ts", "clgroup",
            "pos_long", "pos_short", "pos_long_num", "pos_short_num", "raw",
        ]
        out = _prepare_df(df, target_cols, ts_cols=["ts"])
        self._upsert_df("futoi", out, pk=["ticker", "ts", "clgroup"])
        return len(out)

    # ─── HI2 ────────────────────────────────────────────
    def save_hi2(self, df: pd.DataFrame, market: str = "eq") -> int:
        if df.empty:
            return 0
        target_cols = [
            "ticker", "date", "market",
            "hhi_volume", "hhi_buy", "hhi_sell",
            "hhi_netflow_buy", "hhi_netflow_sell",
            "hhi_passive", "hhi_active",
            "hhi_passive_buy", "hhi_active_buy",
            "hhi_passive_sell", "hhi_active_sell", "raw",
        ]
        df = df.copy()
        df["market"] = market
        out = _prepare_df(df, target_cols, date_cols=["date"])
        self._upsert_df("hi2", out, pk=["ticker", "date", "market"])
        return len(out)

    # ─── Alerts ─────────────────────────────────────────
    def save_alerts(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        target_cols = [
            "ticker", "ts", "alert_type", "side",
            "magnitude", "severity", "description", "raw",
        ]
        out = _prepare_df(df, target_cols, ts_cols=["ts"])
        self._upsert_df("mega_alerts", out, pk=["ticker", "ts", "alert_type"])
        return len(out)

    # ─── OHLCV ──────────────────────────────────────────
    def save_ohlcv(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        target_cols = ["ticker", "ts", "timeframe", "o", "h", "l", "c", "v", "value"]
        out = _prepare_df(df, target_cols, ts_cols=["ts"])
        self._upsert_df("ohlcv", out, pk=["ticker", "ts", "timeframe"])
        return len(out)

    # ─── Orderbook ──────────────────────────────────────
    def save_orderbook_snapshot(
        self, ticker: str, ts: datetime, bids: list, asks: list,
        imbalance: float | None = None,
    ) -> None:
        conn = get_conn()
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        conn.execute(
            """INSERT OR REPLACE INTO orderbook_snapshots
               (ticker, ts, bids_json, asks_json, best_bid, best_ask, imbalance)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ticker, ts, json.dumps(bids), json.dumps(asks),
             best_bid, best_ask, imbalance),
        )

    # ─── Trades log ─────────────────────────────────────
    def save_trades(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = ["ticker", "ts", "tradeno", "price", "qty", "direction"]
        out = _prepare_df(df, cols, ts_cols=["ts"])
        self._upsert_df("trades_log", out, pk=["ticker", "tradeno"])
        return len(out)

    # ─── Agent queries audit ────────────────────────────
    def log_agent_query(
        self, query_text: str, response: dict, recommendation: str,
        latency_ms: float,
    ) -> None:
        conn = get_conn()
        conn.execute(
            """INSERT INTO agent_queries (ts, query_text, response_json,
                                          recommendation, latency_ms)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.utcnow(), query_text, json.dumps(response, default=str),
             recommendation, latency_ms),
        )

    # ─── Query helpers ──────────────────────────────────
    def latest_super_candles(self, ticker: str, n: int = 30) -> pd.DataFrame:
        conn = get_conn()
        return conn.execute(
            """SELECT * FROM super_candles_eq
               WHERE ticker = ? ORDER BY ts DESC LIMIT ?""",
            (ticker, n),
        ).fetch_df()

    def latest_obstats(self, ticker: str, n: int = 1) -> pd.DataFrame:
        conn = get_conn()
        return conn.execute(
            "SELECT * FROM obstats_eq WHERE ticker=? ORDER BY ts DESC LIMIT ?",
            (ticker, n),
        ).fetch_df()

    def latest_orderstats(self, ticker: str, n: int = 5) -> pd.DataFrame:
        conn = get_conn()
        return conn.execute(
            "SELECT * FROM orderstats_eq WHERE ticker=? ORDER BY ts DESC LIMIT ?",
            (ticker, n),
        ).fetch_df()

    def latest_hi2(self, ticker: str) -> pd.DataFrame:
        conn = get_conn()
        return conn.execute(
            "SELECT * FROM hi2 WHERE ticker=? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetch_df()

    def latest_futoi(self, ticker: str, lookback_min: int = 60) -> pd.DataFrame:
        conn = get_conn()
        return conn.execute(
            """SELECT * FROM futoi WHERE ticker=?
               AND ts > now() - INTERVAL '%d minutes'
               ORDER BY ts DESC""" % lookback_min,
            (ticker,),
        ).fetch_df()

    def alerts_window(self, ticker: str, minutes: int = 60) -> pd.DataFrame:
        conn = get_conn()
        return conn.execute(
            """SELECT * FROM mega_alerts WHERE ticker=?
               AND ts > now() - INTERVAL '%d minutes'
               ORDER BY ts DESC""" % minutes,
            (ticker,),
        ).fetch_df()

    def latest_orderbook(self, ticker: str) -> dict | None:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM orderbook_snapshots WHERE ticker=? ORDER BY ts DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        if not row:
            return None
        cols = [d[0] for d in conn.description]
        rec = dict(zip(cols, row))
        rec["bids"] = json.loads(rec.get("bids_json") or "[]")
        rec["asks"] = json.loads(rec.get("asks_json") or "[]")
        return rec

    def latest_ohlcv(self, ticker: str, timeframe: int, n: int = 100) -> pd.DataFrame:
        conn = get_conn()
        return conn.execute(
            """SELECT * FROM ohlcv WHERE ticker=? AND timeframe=?
               ORDER BY ts DESC LIMIT ?""",
            (ticker, timeframe, n),
        ).fetch_df()

    def top_tickers_by_volume(self, lookback_min: int = 60, limit: int = 30) -> list[str]:
        conn = get_conn()
        df = conn.execute(
            """SELECT ticker, SUM(vol) AS v FROM super_candles_eq
               WHERE ts > now() - INTERVAL '%d minutes'
               GROUP BY ticker ORDER BY v DESC LIMIT ?""" % lookback_min,
            (limit,),
        ).fetch_df()
        return df["ticker"].tolist() if not df.empty else []

    # ─── Retention / cold storage ───────────────────────
    def export_old_to_parquet(self, cold_dir: str, retention_days: int = 3) -> None:
        from pathlib import Path

        Path(cold_dir).mkdir(parents=True, exist_ok=True)
        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).strftime("%Y-%m-%d")
        for table in ("super_candles_eq", "obstats_eq", "orderstats_eq",
                      "orderbook_snapshots", "trades_log"):
            out = Path(cold_dir) / f"{table}_until_{cutoff}.parquet"
            conn = get_conn()
            try:
                conn.execute(
                    f"COPY (SELECT * FROM {table} WHERE ts < TIMESTAMP '{cutoff}') "
                    f"TO '{out}' (FORMAT PARQUET);"
                )
                conn.execute(f"DELETE FROM {table} WHERE ts < TIMESTAMP '{cutoff}'")
                logger.info(f"Exported & purged {table} → {out}")
            except Exception as e:
                logger.error(f"export_old_to_parquet failed for {table}: {e}")


# ─── Helpers ────────────────────────────────────────────
def _clean(v: Any) -> Any:
    if pd.isna(v):
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return v


def _prepare_df(
    df: pd.DataFrame,
    target_cols: list[str],
    ts_cols: Iterable[str] = (),
    date_cols: Iterable[str] = (),
) -> pd.DataFrame:
    df = df.copy()
    for c in target_cols:
        if c not in df.columns:
            df[c] = None
    for c in ts_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df[target_cols]


_repo: Repository | None = None


def get_repo() -> Repository:
    global _repo
    if _repo is None:
        _repo = Repository()
    return _repo