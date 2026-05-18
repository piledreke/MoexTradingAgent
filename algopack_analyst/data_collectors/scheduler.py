"""APScheduler orchestration of all collectors."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config import settings
from data_collectors.futoi import run_futoi_collector
from data_collectors.hi2 import run_hi2_collector
from data_collectors.mega_alerts import run_mega_alerts_collector
from data_collectors.realtime import run_ohlcv_collector, run_orderbook_collector
from data_collectors.super_candles import run_super_candles_collector
from data_collectors.index_data import run_index_collector
from data_collectors.multi_tf import run_mtf_collector
from data_collectors.tape_analyzer import run_tape_analyzer
from storage.repository import get_repo
from utils.logger import logger
from utils.moex_client import get_moex_client


_scheduler: AsyncIOScheduler | None = None
_watchlist: list[str] = []


def get_watchlist() -> list[str]:
    return list(_watchlist)


def add_to_watchlist(tickers: list[str]) -> list[str]:
    global _watchlist
    for t in tickers:
        t = t.upper()
        if t not in _watchlist and len(_watchlist) < settings.MAX_WATCHLIST_SIZE:
            _watchlist.append(t)
    return _watchlist


async def _gated_run(coro_fn) -> None:
    """Skip collector cycle if market is closed."""
    try:
        client = get_moex_client()
        if not await client.is_market_open():
            logger.debug(f"market closed, skip {coro_fn.__name__}")
            return
        await coro_fn(_watchlist)
    except Exception as e:
        logger.exception(f"collector {coro_fn.__name__} crashed: {e}")


def start_scheduler() -> AsyncIOScheduler:
    global _scheduler, _watchlist
    if _scheduler is not None:
        return _scheduler

    _watchlist = list(settings.watchlist_list)
    sched = AsyncIOScheduler(timezone="Europe/Moscow")

    # Super Candles every 5 min
    sched.add_job(
        _gated_run, IntervalTrigger(seconds=settings.SUPER_CANDLES_INTERVAL),
        args=[run_super_candles_collector], id="super_candles",
        max_instances=1, coalesce=True,
    )
    # FUTOI every 5 min
    sched.add_job(
        _gated_run, IntervalTrigger(seconds=settings.FUTOI_INTERVAL),
        args=[run_futoi_collector], id="futoi",
        max_instances=1, coalesce=True,
    )
    # Mega Alerts every minute
    sched.add_job(
        _gated_run, IntervalTrigger(seconds=settings.MEGA_ALERTS_INTERVAL),
        args=[run_mega_alerts_collector], id="mega_alerts",
        max_instances=1, coalesce=True,
    )
    # OHLCV every minute
    sched.add_job(
        _gated_run, IntervalTrigger(seconds=settings.OHLCV_INTERVAL),
        args=[run_ohlcv_collector], id="ohlcv",
        max_instances=1, coalesce=True,
    )
    # Orderbook every 10 sec
    sched.add_job(
        _gated_run, IntervalTrigger(seconds=settings.ORDERBOOK_INTERVAL),
        args=[run_orderbook_collector], id="orderbook",
        max_instances=1, coalesce=True,
    )
    # HI2 daily at 19:05 MSK
    sched.add_job(
        run_hi2_collector,
        CronTrigger(hour=settings.HI2_CRON_HOUR,
                    minute=settings.HI2_CRON_MINUTE, timezone="Europe/Moscow"),
        args=[_watchlist], id="hi2", max_instances=1, coalesce=True,
    )
    # Multi-timeframe OHLCV collector
    sched.add_job(
        _gated_run, IntervalTrigger(seconds=getattr(settings, "MTF_INTERVAL", 60)),
        args=[run_mtf_collector], id="mtf_ohlcv",
        max_instances=1, coalesce=True,
    )
    # Index data collector
    sched.add_job(
        run_index_collector,
        IntervalTrigger(seconds=getattr(settings, "INDEX_INTERVAL", 300)),
        id="index_data", max_instances=1, coalesce=True,
    )
    # Tape analyzer for large prints
    sched.add_job(
        _gated_run, IntervalTrigger(seconds=getattr(settings, "TAPE_INTERVAL", 30)),
        args=[run_tape_analyzer], id="tape",
        max_instances=1, coalesce=True,
    )
    # Retention/cold-export daily at 22:00 MSK
    sched.add_job(
        lambda: get_repo().export_old_to_parquet(
            settings.PARQUET_COLD_DIR, settings.HOT_DATA_RETENTION_DAYS
        ),
        CronTrigger(hour=22, minute=0, timezone="Europe/Moscow"),
        id="retention", max_instances=1, coalesce=True,
    )

    sched.start()
    _scheduler = sched
    logger.info("Scheduler started with all collectors")
    return sched


async def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler shut down")


@asynccontextmanager
async def scheduler_lifespan():
    if settings.ENABLE_COLLECTORS:
        start_scheduler()
    try:
        yield
    finally:
        await shutdown_scheduler()