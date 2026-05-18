"""Collect FUTOI (open interest for futures) every 5 min."""
from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd

from config import STOCK_TO_FUTURE, settings
from storage.repository import get_repo
from utils.logger import logger
from utils.moex_client import get_moex_client


async def _collect_one(future_ticker: str) -> int:
    client = get_moex_client()
    repo = get_repo()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        df = await client.get_futoi(future_ticker, from_=today, till=today, latest=True)
        if df.empty:
            return 0
        df["ticker"] = future_ticker
        if "ts" not in df.columns:
            if "tradedate" in df.columns and "tradetime" in df.columns:
                df["ts"] = pd.to_datetime(
                    df["tradedate"].astype(str) + " " + df["tradetime"].astype(str)
                )
            else:
                df["ts"] = datetime.utcnow()
        df["raw"] = df.apply(lambda r: r.to_json(default_handler=str), axis=1)
        return repo.save_futoi(df)
    except Exception as e:
        logger.warning(f"futoi {future_ticker} failed: {e}")
        return 0


async def run_futoi_collector(watchlist: list[str] | None = None) -> None:
    tickers = watchlist or settings.watchlist_list
    futures = {STOCK_TO_FUTURE[t] for t in tickers if t in STOCK_TO_FUTURE}
    # Also always collect macro futures
    futures.update({"Si", "RI", "MX", "BR", "GD"})
    logger.info(f"futoi cycle for {len(futures)} contracts")
    results = await asyncio.gather(
        *[_collect_one(f) for f in futures], return_exceptions=True
    )
    total = sum(r for r in results if isinstance(r, int))
    logger.info(f"futoi saved rows: {total}")