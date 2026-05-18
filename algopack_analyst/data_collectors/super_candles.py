"""Collect Super Candles (tradestats + obstats + orderstats) every 5 min."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pandas as pd

from config import settings
from storage.repository import get_repo
from utils.logger import logger
from utils.moex_client import get_moex_client


async def _collect_one(ticker: str) -> dict[str, int]:
    """Collect tradestats, obstats, orderstats for one ticker (latest=1)."""
    client = get_moex_client()
    repo = get_repo()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    out = {"trade": 0, "obs": 0, "ord": 0}

    try:
        df_t = await client.get_tradestats(
            ticker, market="eq", date=today, latest=True
        )
        if not df_t.empty:
            df_t["ticker"] = ticker
            df_t["ts"] = _build_ts(df_t)
            df_t["raw"] = df_t.apply(lambda r: r.to_json(default_handler=str), axis=1)
            out["trade"] = repo.save_super_candles_eq(df_t)
    except Exception as e:
        logger.warning(f"tradestats {ticker} failed: {e}")

    try:
        df_o = await client.get_obstats(
            ticker, market="eq", date=today, latest=True
        )
        if not df_o.empty:
            df_o["ticker"] = ticker
            df_o["ts"] = _build_ts(df_o)
            df_o["raw"] = df_o.apply(lambda r: r.to_json(default_handler=str), axis=1)
            out["obs"] = repo.save_obstats(df_o)
    except Exception as e:
        logger.warning(f"obstats {ticker} failed: {e}")

    try:
        df_r = await client.get_orderstats(
            ticker, market="eq", date=today, latest=True
        )
        if not df_r.empty:
            df_r["ticker"] = ticker
            df_r["ts"] = _build_ts(df_r)
            df_r["raw"] = df_r.apply(lambda r: r.to_json(default_handler=str), axis=1)
            out["ord"] = repo.save_orderstats(df_r)
    except Exception as e:
        logger.warning(f"orderstats {ticker} failed: {e}")

    return out


def _build_ts(df: pd.DataFrame) -> pd.Series:
    """Build ts column from MOEX tradedate/tradetime fields."""
    if "ts" in df.columns:
        return pd.to_datetime(df["ts"], errors="coerce")
    if "tradedate" in df.columns and "tradetime" in df.columns:
        return pd.to_datetime(
            df["tradedate"].astype(str) + " " + df["tradetime"].astype(str),
            errors="coerce",
        )
    if "SYSTIME" in df.columns:
        return pd.to_datetime(df["SYSTIME"], errors="coerce")
    return pd.Series([datetime.utcnow()] * len(df))


async def run_super_candles_collector(watchlist: list[str] | None = None) -> None:
    """Collect Super Candles for all watchlist tickers concurrently."""
    tickers = watchlist or settings.watchlist_list
    logger.info(f"super_candles cycle for {len(tickers)} tickers")
    results = await asyncio.gather(
        *[_collect_one(t) for t in tickers], return_exceptions=True
    )
    total = {"trade": 0, "obs": 0, "ord": 0}
    for r in results:
        if isinstance(r, dict):
            for k, v in r.items():
                total[k] += v
    logger.info(f"super_candles done: {total}")