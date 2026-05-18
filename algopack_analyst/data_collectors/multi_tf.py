"""Multi-timeframe OHLCV collector — fetches 1/5/15/60/1d candles."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pandas as pd

from config import settings
from storage.feature_store import compute_and_store_features
from storage.repository import get_repo
from utils.logger import logger
from utils.moex_client import get_moex_client

TIMEFRAMES = {
    1: timedelta(hours=4),       # 1m → 4h history
    5: timedelta(hours=12),      # 5m → 12h
    15: timedelta(days=2),       # 15m → 2 days
    60: timedelta(days=10),      # 1h  → 10 days
    24 * 60: timedelta(days=200) # 1d  → 200 days
}


async def _collect_tf(ticker: str, tf: int, lookback: timedelta) -> int:
    client = get_moex_client()
    repo = get_repo()
    till = datetime.utcnow()
    from_ = till - lookback
    try:
        df = await client.get_candles(
            ticker, interval=tf,
            from_=from_.strftime("%Y-%m-%d %H:%M:%S"),
            till=till.strftime("%Y-%m-%d %H:%M:%S"),
        )
        if df.empty:
            return 0
        df = df.rename(columns={
            "open": "o", "high": "h", "low": "l", "close": "c",
            "volume": "v", "value": "value"
        })
        df["ticker"] = ticker
        df["timeframe"] = tf
        if "ts" not in df.columns and "begin" in df.columns:
            df["ts"] = pd.to_datetime(df["begin"])
        return repo.save_ohlcv(df)
    except Exception as e:
        logger.warning(f"mtf ohlcv {ticker} tf={tf} failed: {e}")
        return 0


async def collect_ticker_mtf(ticker: str) -> dict[int, int]:
    """Collect all TFs for one ticker, then compute features."""
    results = await asyncio.gather(
        *[_collect_tf(ticker, tf, lb) for tf, lb in TIMEFRAMES.items()],
        return_exceptions=True,
    )
    summary = {}
    for (tf, _), r in zip(TIMEFRAMES.items(), results):
        summary[tf] = r if isinstance(r, int) else 0

    # After collection — pre-compute features
    repo = get_repo()
    ohlcv_by_tf: dict[int, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        df = repo.latest_ohlcv(ticker, tf, n=300)
        if not df.empty:
            ohlcv_by_tf[tf] = df
    try:
        await compute_and_store_features(ticker, ohlcv_by_tf)
    except Exception as e:
        logger.warning(f"feature store update failed for {ticker}: {e}")
    return summary


async def run_mtf_collector(watchlist: list[str] | None = None) -> None:
    tickers = watchlist or settings.watchlist_list
    logger.info(f"mtf cycle for {len(tickers)} tickers")
    sem = asyncio.Semaphore(5)

    async def _one(t):
        async with sem:
            return await collect_ticker_mtf(t)

    results = await asyncio.gather(*[_one(t) for t in tickers], return_exceptions=True)
    total = {tf: 0 for tf in TIMEFRAMES}
    for r in results:
        if isinstance(r, dict):
            for tf, n in r.items():
                total[tf] += n
    logger.info(f"mtf done: {total}")