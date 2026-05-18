"""Collect HI2 concentration index daily after 19:00 MSK."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from config import settings
from storage.repository import get_repo
from utils.logger import logger
from utils.moex_client import get_moex_client


async def _collect_one(ticker: str, market: str = "eq") -> int:
    client = get_moex_client()
    repo = get_repo()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    week_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        df = await client.get_hi2(ticker, market=market, from_=week_ago, till=today)
        if df.empty:
            return 0
        df["ticker"] = ticker
        if "date" not in df.columns and "tradedate" in df.columns:
            df["date"] = df["tradedate"]
        df["raw"] = df.apply(lambda r: r.to_json(default_handler=str), axis=1)
        return repo.save_hi2(df, market=market)
    except Exception as e:
        logger.warning(f"hi2 {ticker} failed: {e}")
        return 0


async def run_hi2_collector(watchlist: list[str] | None = None) -> None:
    tickers = watchlist or settings.watchlist_list
    logger.info(f"hi2 cycle for {len(tickers)} tickers")
    results = await asyncio.gather(
        *[_collect_one(t) for t in tickers], return_exceptions=True
    )
    total = sum(r for r in results if isinstance(r, int))
    logger.info(f"hi2 saved rows: {total}")