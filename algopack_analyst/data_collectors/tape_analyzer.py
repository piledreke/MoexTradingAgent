"""Tape analyzer — detect large prints (institutional activity) from trades log."""
from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd

from config import settings
from storage.db import get_conn
from storage.repository import get_repo
from utils.logger import logger
from utils.moex_client import get_moex_client


async def _analyze_ticker(ticker: str) -> dict[str, int]:
    client = get_moex_client()
    repo = get_repo()
    try:
        df = await client.get_trades(ticker, tradeno=0)
        if df.empty:
            return {"trades": 0, "large": 0}
        # Normalize column names
        df = df.rename(columns={
            "TRADENO": "tradeno", "TRADETIME": "tradetime",
            "TRADEDATE": "tradedate", "PRICE": "price",
            "QUANTITY": "qty", "BUYSELL": "direction",
        })
        df["ticker"] = ticker
        if "ts" not in df.columns:
            df["ts"] = pd.to_datetime(
                df["tradedate"].astype(str) + " " + df["tradetime"].astype(str),
                errors="coerce",
            )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
        df["value"] = df["price"] * df["qty"]
        df["direction"] = df["direction"].astype(str).str.upper().map(
            {"B": "buy", "S": "sell"}
        ).fillna("unknown")

        repo.save_trades(df[["ticker", "ts", "tradeno", "price", "qty", "direction"]])

        # Detect large prints (>= 95th percentile)
        if len(df) > 20:
            thr = df["value"].quantile(0.95)
            large = df[df["value"] >= thr]
            for _, row in large.iterrows():
                _log_large_print(ticker, row, thr)
            return {"trades": len(df), "large": len(large)}
        return {"trades": len(df), "large": 0}
    except Exception as e:
        logger.warning(f"tape {ticker} failed: {e}")
        return {"trades": 0, "large": 0}


def _log_large_print(ticker: str, row: pd.Series, threshold: float) -> None:
    """Log large print as a synthetic Mega Alert."""
    try:
        conn = get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO mega_alerts
            (ticker, ts, alert_type, side, magnitude, severity, description, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, row["ts"],
            "large_print",
            row.get("direction"),
            float(row["value"] / threshold),
            "info",
            f"Large print: {row['qty']:.0f} @ {row['price']:.2f} = {row['value']:.0f}₽",
            row.to_json(default_handler=str),
        ))
    except Exception as e:
        logger.warning(f"_log_large_print failed for {ticker}: {e}")


async def run_tape_analyzer(watchlist: list[str] | None = None) -> None:
    tickers = watchlist or settings.watchlist_list
    sem = asyncio.Semaphore(5)

    async def _one(t):
        async with sem:
            return await _analyze_ticker(t)

    results = await asyncio.gather(*[_one(t) for t in tickers], return_exceptions=True)
    total = {"trades": 0, "large": 0}
    for r in results:
        if isinstance(r, dict):
            for k, v in r.items():
                total[k] += v
    logger.info(f"tape analyzer: {total}")