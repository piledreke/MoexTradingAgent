"""Historical backfill of ALGOPACK datasets + 5m candles."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional

from app.clients.moex_client import MoexClient
from app.config import Settings, get_settings
from app.ingestion.normalizers import (
    normalize_alerts,
    normalize_candles,
    normalize_hi2,
    normalize_obstats,
    normalize_orderstats,
    normalize_tradestats,
)
from app.logging_config import get_logger
from app.storage.repository import Repository
from app.utils.time import msk_today

_LOG = get_logger(__name__)


class Backfiller:
    """Pull last N business days of historic data into local SQLite."""

    def __init__(
        self,
        repo: Repository,
        client: MoexClient,
        settings: Optional[Settings] = None,
    ) -> None:
        self.repo = repo
        self.client = client
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    def backfill_ticker(self, secid: str, days: int) -> Dict[str, int]:
        """Backfill the canonical ALGOPACK datasets + 5m candles for one ticker."""
        days = max(1, int(days))
        till = msk_today()
        start = till - timedelta(days=days)
        counts: Dict[str, int] = {}

        if self.client.has_token:
            try:
                rows = self.client.fetch_tradestats(secid=secid, date_from=start, date_till=till)
                counts["tradestats"] = self.repo.save_tradestats(normalize_tradestats(rows))
            except Exception as exc:
                _LOG.warning("backfill_tradestats_failed", extra={"secid": secid, "error": str(exc)})
                counts["tradestats"] = 0
            try:
                rows = self.client.fetch_orderstats(secid=secid, date_from=start, date_till=till)
                counts["orderstats"] = self.repo.save_orderstats(normalize_orderstats(rows))
            except Exception as exc:
                _LOG.warning("backfill_orderstats_failed", extra={"secid": secid, "error": str(exc)})
                counts["orderstats"] = 0
            try:
                rows = self.client.fetch_obstats(secid=secid, date_from=start, date_till=till)
                counts["obstats"] = self.repo.save_obstats(normalize_obstats(rows))
            except Exception as exc:
                _LOG.warning("backfill_obstats_failed", extra={"secid": secid, "error": str(exc)})
                counts["obstats"] = 0
            try:
                rows = self.client.fetch_alerts(secid=secid, date_from=start, date_till=till)
                counts["alerts"] = self.repo.save_alerts(normalize_alerts(rows))
            except Exception as exc:
                _LOG.warning("backfill_alerts_failed", extra={"secid": secid, "error": str(exc)})
                counts["alerts"] = 0
            try:
                rows = self.client.fetch_hi2(secid=secid, date_from=start, date_till=till)
                counts["hi2"] = self.repo.save_hi2(normalize_hi2(rows))
            except Exception as exc:
                _LOG.warning("backfill_hi2_failed", extra={"secid": secid, "error": str(exc)})
                counts["hi2"] = 0
        else:
            counts.update({"tradestats": 0, "orderstats": 0, "obstats": 0, "alerts": 0, "hi2": 0})

        try:
            candle_rows = self.client.fetch_candles(secid=secid, date_from=start, date_till=till, interval=10)
            counts["candles"] = self.repo.save_intraday_candles(
                normalize_candles(candle_rows, secid, interval_min=10)
            )
        except Exception as exc:
            _LOG.warning("backfill_candles_failed", extra={"secid": secid, "error": str(exc)})
            counts["candles"] = 0

        _LOG.info("backfill_ticker_done", extra={"secid": secid, "days": days, **counts})
        return counts

    # ------------------------------------------------------------------
    def backfill_universe(self, universe: Iterable[str], days: int) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for secid in universe:
            out[secid] = self.backfill_ticker(secid, days)
        self.repo.log_event(
            level="INFO",
            event_type="backfill_universe_done",
            message=f"backfill universe={len(out)} days={days}",
            payload={"days": days, "tickers": list(out.keys())},
        )
        return out
