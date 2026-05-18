"""Polling orchestrator for the technical advisor.

Cadences (per spec):

* ``marketdata`` / ``alerts`` -> every ``POLL_INTERVAL_SECONDS`` (~60s)
* ALGOPACK ``tradestats``/``orderstats``/``obstats`` -> every
  ``SUPER_CANDLE_INTERVAL_SECONDS`` (~300s) with a 10-20s lag
* ``hi2`` -> once a day after ``HI2_HOUR_MSK:HI2_MINUTE_MSK`` MSK
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional

from app.clients.moex_client import MoexClient
from app.config import Settings, get_settings
from app.ingestion.backfill import Backfiller
from app.ingestion.normalizers import (
    normalize_alerts,
    normalize_candles,
    normalize_hi2,
    normalize_marketdata,
    normalize_obstats,
    normalize_orderstats,
    normalize_tradestats,
)
from app.logging_config import get_logger
from app.storage.repository import Repository
from app.utils.time import is_msk_trading_window, msk_today, now_msk

_LOG = get_logger(__name__)


@dataclass
class CycleResult:
    fetched: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class IngestionService:
    """High-level ingestion façade used by CLI ``once`` and ``run``."""

    def __init__(
        self,
        repo: Repository,
        client: MoexClient,
        settings: Optional[Settings] = None,
    ) -> None:
        self.repo = repo
        self.client = client
        self.settings = settings or get_settings()
        self._last_super = 0.0
        self._last_market = 0.0
        self._last_hi2_date: Optional[date] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Sub-cycles
    # ------------------------------------------------------------------
    def fetch_super_candles(self) -> Dict[str, int]:
        """Fetch latest tradestats / orderstats / obstats for today."""
        out: Dict[str, int] = {"tradestats": 0, "orderstats": 0, "obstats": 0}
        if not self.client.has_token:
            return out
        today = msk_today()
        try:
            rows = self.client.fetch_tradestats(date_from=today, latest=True)
            out["tradestats"] = self.repo.save_tradestats(normalize_tradestats(rows))
        except Exception as exc:
            _LOG.warning("fetch_tradestats_failed", extra={"error": str(exc)})
        try:
            rows = self.client.fetch_orderstats(date_from=today, latest=True)
            out["orderstats"] = self.repo.save_orderstats(normalize_orderstats(rows))
        except Exception as exc:
            _LOG.warning("fetch_orderstats_failed", extra={"error": str(exc)})
        try:
            rows = self.client.fetch_obstats(date_from=today, latest=True)
            out["obstats"] = self.repo.save_obstats(normalize_obstats(rows))
        except Exception as exc:
            _LOG.warning("fetch_obstats_failed", extra={"error": str(exc)})
        return out

    def fetch_alerts(self) -> int:
        if not self.client.has_token:
            return 0
        try:
            rows = self.client.fetch_alerts(date_from=msk_today())
            return self.repo.save_alerts(normalize_alerts(rows))
        except Exception as exc:
            _LOG.warning("fetch_alerts_failed", extra={"error": str(exc)})
            return 0

    def fetch_marketdata(self) -> int:
        universe = self.settings.universe
        try:
            md = self.client.fetch_marketdata(secids=universe)
        except Exception as exc:
            _LOG.warning("fetch_marketdata_failed", extra={"error": str(exc)})
            return 0
        rows = list(md.values())
        return self.repo.save_marketdata_snapshot(normalize_marketdata(rows))

    def fetch_intraday_candles(self, days: int = 2, interval_min: int = 10) -> Dict[str, int]:
        out: Dict[str, int] = {}
        start = msk_today() - timedelta(days=days)
        till = msk_today()
        for secid in self.settings.universe:
            try:
                rows = self.client.fetch_candles(
                    secid=secid, date_from=start, date_till=till, interval=interval_min
                )
                out[secid] = self.repo.save_intraday_candles(
                    normalize_candles(rows, secid, interval_min=interval_min)
                )
            except Exception as exc:
                _LOG.warning("fetch_candles_failed", extra={"secid": secid, "error": str(exc)})
                out[secid] = 0
        return out

    def fetch_hi2(self) -> int:
        if not self.client.has_token:
            return 0
        try:
            rows = self.client.fetch_hi2(date_from=msk_today(), date_till=msk_today())
            return self.repo.save_hi2(normalize_hi2(rows))
        except Exception as exc:
            _LOG.warning("fetch_hi2_failed", extra={"error": str(exc)})
            return 0

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------
    def run_once(self, refresh_candles: bool = True) -> CycleResult:
        start = time.time()
        result = CycleResult()

        try:
            result.fetched["marketdata"] = self.fetch_marketdata()
        except Exception as exc:
            result.errors.append(f"marketdata: {exc}")

        try:
            result.fetched["alerts"] = self.fetch_alerts()
        except Exception as exc:
            result.errors.append(f"alerts: {exc}")

        try:
            sc = self.fetch_super_candles()
            for k, v in sc.items():
                result.fetched[k] = v
        except Exception as exc:
            result.errors.append(f"super_candles: {exc}")

        if refresh_candles:
            try:
                candles = self.fetch_intraday_candles(days=2)
                result.fetched["candles"] = sum(candles.values())
            except Exception as exc:
                result.errors.append(f"candles: {exc}")

        # HI2 once a day after the configured threshold.
        today = msk_today()
        if self._last_hi2_date != today:
            now = now_msk()
            if (now.hour, now.minute) >= (self.settings.hi2_hour_msk, self.settings.hi2_minute_msk):
                try:
                    result.fetched["hi2"] = self.fetch_hi2()
                    self._last_hi2_date = today
                except Exception as exc:
                    result.errors.append(f"hi2: {exc}")

        elapsed = time.time() - start
        self.repo.log_event(
            level="INFO" if not result.errors else "WARNING",
            event_type="ingestion_cycle",
            message=f"cycle done in {elapsed:.2f}s",
            payload={"fetched": result.fetched, "errors": result.errors, "elapsed_s": round(elapsed, 3)},
        )
        return result

    # ------------------------------------------------------------------
    def lightweight_backfill_if_empty(self, days: int = 5) -> None:
        """Run a backfill only if the DB looks empty for the universe."""
        ages = self.repo.last_ingestion_age()
        if all((v is None or v > 7 * 24 * 3600) for v in ages.values()):
            _LOG.info("lightweight_backfill_triggered", extra={"days": days})
            backfiller = Backfiller(self.repo, self.client, self.settings)
            backfiller.backfill_universe(self.settings.universe, days=days)

    # ------------------------------------------------------------------
    # Long-running loop
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    def run_forever(self, on_cycle: Optional[Callable[[CycleResult], None]] = None) -> None:
        poll = max(5, self.settings.poll_interval_seconds)
        super_poll = max(poll, self.settings.super_candle_interval_seconds)
        next_super = 0.0
        next_candles = 0.0
        next_retention = 0.0
        retention_period = 6 * 3600  # 6h
        _LOG.info(
            "scheduler_started",
            extra={
                "poll_interval": poll,
                "super_interval": super_poll,
                "universe_size": len(self.settings.universe),
            },
        )
        while not self._stop.is_set():
            cycle_start = time.time()
            result = CycleResult()
            try:
                result.fetched["marketdata"] = self.fetch_marketdata()
            except Exception as exc:
                result.errors.append(f"marketdata: {exc}")
            try:
                result.fetched["alerts"] = self.fetch_alerts()
            except Exception as exc:
                result.errors.append(f"alerts: {exc}")

            now = time.time()
            if now >= next_super:
                # Spec says: 5min + small lag 10-20s.
                if is_msk_trading_window():
                    time.sleep(15)
                try:
                    sc = self.fetch_super_candles()
                    for k, v in sc.items():
                        result.fetched[k] = v
                except Exception as exc:
                    result.errors.append(f"super_candles: {exc}")
                next_super = now + super_poll

            if now >= next_candles:
                try:
                    candles = self.fetch_intraday_candles(days=1)
                    result.fetched["candles"] = sum(candles.values())
                except Exception as exc:
                    result.errors.append(f"candles: {exc}")
                next_candles = now + super_poll

            today = msk_today()
            if self._last_hi2_date != today:
                msk = now_msk()
                if (msk.hour, msk.minute) >= (self.settings.hi2_hour_msk, self.settings.hi2_minute_msk):
                    try:
                        result.fetched["hi2"] = self.fetch_hi2()
                        self._last_hi2_date = today
                    except Exception as exc:
                        result.errors.append(f"hi2: {exc}")

            if now >= next_retention:
                try:
                    removed = self.repo.apply_retention(self.settings.raw_cache_retention_days)
                    self.repo.log_event(
                        level="INFO",
                        event_type="retention_applied",
                        message="raw cache retention applied",
                        payload={"removed": removed, "days": self.settings.raw_cache_retention_days},
                    )
                except Exception as exc:
                    result.errors.append(f"retention: {exc}")
                next_retention = now + retention_period

            elapsed = time.time() - cycle_start
            self.repo.log_event(
                level="INFO" if not result.errors else "WARNING",
                event_type="ingestion_cycle",
                message=f"cycle done in {elapsed:.2f}s",
                payload={"fetched": result.fetched, "errors": result.errors, "elapsed_s": round(elapsed, 3)},
            )
            if on_cycle is not None:
                try:
                    on_cycle(result)
                except Exception as exc:
                    _LOG.error("on_cycle_callback_failed", extra={"error": str(exc)})

            # Sleep until next poll, but exit promptly on stop.
            sleep_for = max(1.0, poll - (time.time() - cycle_start))
            self._stop.wait(timeout=sleep_for)
