"""Poll Mega Alerts every minute."""
from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd

from config import settings
from storage.repository import get_repo
from utils.logger import logger
from utils.moex_client import get_moex_client


async def _collect_one(ticker: str) -> int:
    client = get_moex_client()
    repo = get_repo()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        df = await client.get_alerts(ticker, market="eq", from_=today, till=today)
        if df.empty:
            return 0
        df["ticker"] = ticker
        if "ts" not in df.columns:
            if "tradedate" in df.columns and "tradetime" in df.columns:
                df["ts"] = pd.to_datetime(
                    df["tradedate"].astype(str) + " " + df["tradetime"].astype(str)
                )
            else:
                df["ts"] = datetime.utcnow()
        if "severity" not in df.columns:
            df["severity"] = "info"
        df["raw"] = df.apply(lambda r: r.to_json(default_handler=str), axis=1)
        return repo.save_alerts(df)
    except Exception as e:
        logger.warning(f"alerts {ticker} failed: {e}")
        return 0


async def run_mega_alerts_collector(watchlist: list[str] | None = None) -> None:
    tickers = watchlist or settings.watchlist_list
    results = await asyncio.gather(
        *[_collect_one(t) for t in tickers], return_exceptions=True
    )
    total = sum(r for r in results if isinstance(r, int))
    logger.debug(f"mega_alerts saved rows: {total}")