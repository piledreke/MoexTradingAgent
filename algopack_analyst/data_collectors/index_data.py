"""Collect MOEX index data (IMOEX, MOEXOG, MOEXBC etc.) for cross-asset analysis."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pandas as pd

from storage.repository import get_repo
from utils.logger import logger
from utils.moex_client import get_moex_client


INDICES = [
    "IMOEX",     # main index
    "MOEXOG",    # oil & gas
    "MOEXBC",    # blue chips
    "MOEXBMI",   # broad market
    "MOEXFN",    # financials
    "MOEXMM",    # metals & mining
    "MOEXIT",    # IT
    "RTSI",      # RTS (USD-denominated)
]

# Map sector index → tickers (used in scoring for relative strength)
SECTOR_MAP: dict[str, list[str]] = {
    "MOEXOG": ["LKOH", "ROSN", "GAZP", "TATN", "SNGS", "NVTK"],
    "MOEXFN": ["SBER", "VTBR", "MOEX"],
    "MOEXMM": ["GMKN", "CHMF", "ALRS", "PLZL", "POLY"],
    "MOEXIT": ["YNDX", "OZON", "VKCO"],
}


def get_sector_for_ticker(ticker: str) -> str | None:
    """Get sector index for ticker."""
    for idx, tickers in SECTOR_MAP.items():
        if ticker.upper() in tickers:
            return idx
    return None


async def _collect_index(index: str) -> int:
    client = get_moex_client()
    repo = get_repo()
    till = datetime.utcnow()
    from_ = till - timedelta(days=10)
    n = 0
    try:
        # 1h candles
        df1h = await client.get_index_candles(
            index, interval=60,
            from_=from_.strftime("%Y-%m-%d %H:%M:%S"),
            till=till.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if not df1h.empty:
            df1h = df1h.rename(columns={
                "open": "o", "high": "h", "low": "l", "close": "c",
                "volume": "v", "value": "value"
            })
            df1h["ticker"] = index
            df1h["timeframe"] = 60
            n += repo.save_ohlcv(df1h)

        # 1d candles
        df1d = await client.get_index_candles(
            index, interval=24,
            from_=(till - timedelta(days=200)).strftime("%Y-%m-%d %H:%M:%S"),
            till=till.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if not df1d.empty:
            df1d = df1d.rename(columns={
                "open": "o", "high": "h", "low": "l", "close": "c",
                "volume": "v", "value": "value"
            })
            df1d["ticker"] = index
            df1d["timeframe"] = 24 * 60
            n += repo.save_ohlcv(df1d)
    except Exception as e:
        logger.warning(f"index {index} failed: {e}")
    return n


async def run_index_collector() -> None:
    logger.info(f"index cycle for {len(INDICES)} indices")
    results = await asyncio.gather(
        *[_collect_index(i) for i in INDICES], return_exceptions=True
    )
    total = sum(r for r in results if isinstance(r, int))
    logger.info(f"index data saved rows: {total}")