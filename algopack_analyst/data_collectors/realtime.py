"""Real-time OHLCV and orderbook collection."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pandas as pd

from config import settings
from storage.repository import get_repo
from utils.logger import logger
from utils.moex_client import get_moex_client


async def _collect_ohlcv(ticker: str, interval: int = 1) -> int:
    client = get_moex_client()
    repo = get_repo()
    till = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    from_ = (datetime.utcnow() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        df = await client.get_candles(ticker, interval=interval, from_=from_, till=till)
        if df.empty:
            return 0
        df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c",
                                "volume": "v", "value": "value"})
        df["ticker"] = ticker
        df["timeframe"] = interval
        if "ts" not in df.columns and "begin" in df.columns:
            df["ts"] = pd.to_datetime(df["begin"])
        return repo.save_ohlcv(df)
    except Exception as e:
        logger.warning(f"ohlcv {ticker} failed: {e}")
        return 0


async def _collect_orderbook(ticker: str) -> int:
    client = get_moex_client()
    repo = get_repo()
    try:
        ob = await client.get_orderbook(ticker)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            return 0
        bid_vol = sum(q for _, q in bids[:5])
        ask_vol = sum(q for _, q in asks[:5])
        imb = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0.0
        repo.save_orderbook_snapshot(
            ticker=ticker,
            ts=datetime.utcnow(),
            bids=bids,
            asks=asks,
            imbalance=imb,
        )
        return 1
    except Exception as e:
        logger.warning(f"orderbook {ticker} failed: {e}")
        return 0


async def run_ohlcv_collector(watchlist: list[str] | None = None) -> None:
    tickers = watchlist or settings.watchlist_list
    results = await asyncio.gather(
        *[_collect_ohlcv(t, 1) for t in tickers], return_exceptions=True
    )
    total = sum(r for r in results if isinstance(r, int))
    logger.debug(f"ohlcv saved rows: {total}")


async def run_orderbook_collector(watchlist: list[str] | None = None) -> None:
    tickers = watchlist or settings.watchlist_list
    results = await asyncio.gather(
        *[_collect_orderbook(t) for t in tickers], return_exceptions=True
    )
    total = sum(r for r in results if isinstance(r, int))
    logger.debug(f"orderbook snapshots: {total}")