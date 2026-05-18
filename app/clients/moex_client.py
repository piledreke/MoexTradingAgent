"""MOEX ISS / ALGOPACK HTTP client.

Two endpoints are involved:

* ``https://apim.moex.com/iss`` – ALGOPACK datasets (tradestats, orderstats,
  obstats, alerts, hi2). Requires ``Authorization: Bearer <APIKEY>`` header.
* ``https://iss.moex.com/iss`` – free public ISS data (candles, marketdata,
  orderbook). Works without a token.

Every response is normalised into a list of dicts via :func:`parse_iss_block`,
which understands the ``{"data": {"columns": [...], "data": [[...], ...]}}``
shape used by ISS and ALGOPACK uniformly.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import httpx

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.utils.retry import retry_call

_LOG = get_logger(__name__)

# Algopack equity dataset path prefix.
ALGOPACK_EQ_PATH = "datashop/algopack/eq"

# Lowercased ISS column renames we want to apply *uniformly*.
_COL_ALIASES = {
    "ticker": "secid",
    "board": "boardid",
}


def parse_iss_block(payload: Any, block: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse one ISS-style block into a list of dicts.

    Accepts either the full ISS JSON envelope (a dict possibly with several
    blocks like ``{"securities": {...}, "marketdata": {...}}``) or a single
    block dict ``{"columns": [...], "data": [...]}``. When ``block`` is given
    and the envelope contains it, only that block is decoded.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        # Already a list of dicts.
        return [dict(r) for r in payload if isinstance(r, Mapping)]
    if not isinstance(payload, Mapping):
        return []

    # Envelope with multiple blocks.
    if block and block in payload and isinstance(payload[block], Mapping):
        return parse_iss_block(payload[block])

    if "columns" in payload and "data" in payload:
        columns_raw = payload.get("columns") or []
        data_rows = payload.get("data") or []
        columns = [_COL_ALIASES.get(str(c).lower(), str(c).lower()) for c in columns_raw]
        out: List[Dict[str, Any]] = []
        for row in data_rows:
            if not isinstance(row, (list, tuple)):
                continue
            out.append({columns[i]: row[i] for i in range(min(len(columns), len(row)))})
        return out

    # Envelope of envelopes.
    for key, val in payload.items():
        if isinstance(val, Mapping) and "columns" in val and "data" in val:
            return parse_iss_block(val)
    return []


def parse_iss_envelope(payload: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Parse every block in an ISS envelope: ``{block_name: [row, ...]}``."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not isinstance(payload, Mapping):
        return out
    for key, val in payload.items():
        if isinstance(val, Mapping) and "columns" in val and "data" in val:
            out[key] = parse_iss_block(val)
    return out


# ---------------------------------------------------------------------------
class MoexClient:
    """HTTP client for ISS and ALGOPACK endpoints."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._token = self.settings.moex_token
        self._timeout = self.settings.moex_request_timeout
        self._max_retries = max(1, self.settings.moex_max_retries)
        headers = {"User-Agent": "moex-tech-agent/0.1"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._algopack_client = httpx.Client(
            base_url=self.settings.moex_base_url,
            headers=headers,
            timeout=self._timeout,
            follow_redirects=True,
        )
        # The public ISS endpoint *also* accepts the bearer token, but does not
        # require it. We reuse the same headers for consistency.
        self._public_client = httpx.Client(
            base_url=self.settings.moex_public_base_url,
            headers=headers,
            timeout=self._timeout,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------
    @property
    def has_token(self) -> bool:
        return bool(self._token)

    def close(self) -> None:
        try:
            self._algopack_client.close()
        finally:
            self._public_client.close()

    def __enter__(self) -> "MoexClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _is_retryable_http(exc: BaseException) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            # Don't retry 4xx (except 408, 425, 429).
            if 400 <= status < 500 and status not in (408, 425, 429):
                return False
            return True
        if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
            return True
        return True

    def _get_json(self, client: httpx.Client, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        # ``path`` must end with .json for ISS endpoints.
        if "?" not in path and not path.endswith(".json"):
            path = path + ".json"

        def _call() -> Any:
            response = client.get(path, params=params)
            response.raise_for_status()
            return response.json()

        return retry_call(
            _call,
            attempts=self._max_retries,
            retry_on=(httpx.HTTPError,),
            is_retryable=self._is_retryable_http,
            on_retry=lambda n, e, d: _LOG.warning(
                "moex_retry",
                extra={"path": path, "attempt": n, "delay": d, "error": str(e)},
            ),
        )

    # ------------------------------------------------------------------
    # ALGOPACK datasets
    # ------------------------------------------------------------------
    def _algopack_paged(
        self,
        endpoint: str,
        secid: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        page_size: int = 1000,
        max_pages: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch one ALGOPACK endpoint with ``start`` pagination.

        IMPORTANT — observed server behavior (2026-05): regardless of the
        requested ``limit``, the ALGOPACK API caps each response at **1000
        rows**. The previous "break when ``len(rows) < page_size``" heuristic
        therefore returned only the first page and truncated multi-day
        backfills (we saw exactly 1000 rows per ticker for 30 days, where
        ~5000 was expected).

        New rule: keep paging by ``start`` offset until an empty page comes
        back or ``max_pages`` is hit.
        """
        if not self._token:
            raise RuntimeError(
                "ALGOPACK endpoint requested but MOEX_API_KEY / MOEX_ALGOPACK_TOKEN is not set"
            )
        base = f"{ALGOPACK_EQ_PATH}/{endpoint}"
        if secid:
            base = f"{base}/{secid}"
        out: List[Dict[str, Any]] = []
        start = 0
        for page in range(max_pages):
            q = dict(params or {})
            q.setdefault("limit", page_size)
            q["start"] = start
            payload = self._get_json(self._algopack_client, base, params=q)
            rows = parse_iss_block(payload, block="data")
            if not rows:
                break
            out.extend(rows)
            start += len(rows)
            # Safety: if the server is returning the same rows over and over
            # (unlikely but possible), break to avoid an infinite loop.
            if page + 1 == max_pages:
                _LOG.warning(
                    "algopack_pagination_max_pages_reached",
                    extra={
                        "endpoint": endpoint,
                        "secid": secid,
                        "max_pages": max_pages,
                        "rows_so_far": len(out),
                    },
                )
        return out

    def fetch_tradestats(
        self,
        secid: Optional[str] = None,
        date_from: Optional[date | str] = None,
        date_till: Optional[date | str] = None,
        latest: bool = False,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if date_from is not None and secid is None:
            params["date"] = _date_str(date_from)
        else:
            if date_from is not None:
                params["from"] = _date_str(date_from)
            if date_till is not None:
                params["till"] = _date_str(date_till)
        if latest:
            params["latest"] = 1
        return self._algopack_paged("tradestats", secid, params)

    def fetch_orderstats(
        self,
        secid: Optional[str] = None,
        date_from: Optional[date | str] = None,
        date_till: Optional[date | str] = None,
        latest: bool = False,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if date_from is not None and secid is None:
            params["date"] = _date_str(date_from)
        else:
            if date_from is not None:
                params["from"] = _date_str(date_from)
            if date_till is not None:
                params["till"] = _date_str(date_till)
        if latest:
            params["latest"] = 1
        return self._algopack_paged("orderstats", secid, params)

    def fetch_obstats(
        self,
        secid: Optional[str] = None,
        date_from: Optional[date | str] = None,
        date_till: Optional[date | str] = None,
        latest: bool = False,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if date_from is not None and secid is None:
            params["date"] = _date_str(date_from)
        else:
            if date_from is not None:
                params["from"] = _date_str(date_from)
            if date_till is not None:
                params["till"] = _date_str(date_till)
        if latest:
            params["latest"] = 1
        return self._algopack_paged("obstats", secid, params)

    def fetch_alerts(
        self,
        secid: Optional[str] = None,
        date_from: Optional[date | str] = None,
        date_till: Optional[date | str] = None,
        latest: bool = False,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if date_from is not None and secid is None:
            params["date"] = _date_str(date_from)
        else:
            if date_from is not None:
                params["from"] = _date_str(date_from)
            if date_till is not None:
                params["till"] = _date_str(date_till)
        if latest:
            params["latest"] = 1
        return self._algopack_paged("alerts", secid, params)

    def fetch_hi2(
        self,
        secid: Optional[str] = None,
        date_from: Optional[date | str] = None,
        date_till: Optional[date | str] = None,
        latest: bool = False,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if date_from is not None and secid is None:
            params["date"] = _date_str(date_from)
        else:
            if date_from is not None:
                params["from"] = _date_str(date_from)
            if date_till is not None:
                params["till"] = _date_str(date_till)
        if latest:
            params["latest"] = 1
        return self._algopack_paged("hi2", secid, params)

    # ------------------------------------------------------------------
    # Real-time public ISS endpoints (TQBR board for equities).
    # ------------------------------------------------------------------
    def fetch_candles(
        self,
        secid: str,
        date_from: Optional[date | str] = None,
        date_till: Optional[date | str] = None,
        interval: int = 10,
        board: str = "TQBR",
        market: str = "shares",
        engine: str = "stock",
        page_size: int = 500,
        max_pages: int = 20,
    ) -> List[Dict[str, Any]]:
        # MOEX ISS only accepts intervals 1, 10, 60, 24, 7, 31. Coerce.
        valid_intervals = (1, 10, 60, 24, 7, 31)
        if interval not in valid_intervals:
            interval = 10
        path = (
            f"engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/candles"
        )
        params: Dict[str, Any] = {"interval": interval, "limit": page_size}
        if date_from is not None:
            params["from"] = _date_str(date_from)
        if date_till is not None:
            params["till"] = _date_str(date_till)
        out: List[Dict[str, Any]] = []
        start = 0
        for _ in range(max_pages):
            params["start"] = start
            payload = self._get_json(self._public_client, path, params=params)
            block = payload.get("candles") if isinstance(payload, Mapping) else None
            rows = parse_iss_block(block or payload, block="candles")
            if not rows:
                break
            out.extend(rows)
            start += len(rows)
        return out

    def fetch_marketdata(
        self,
        secids: Optional[Sequence[str]] = None,
        board: str = "TQBR",
        market: str = "shares",
        engine: str = "stock",
    ) -> Dict[str, Dict[str, Any]]:
        """Return ``{secid: row}`` with securities+marketdata merged."""
        path = f"engines/{engine}/markets/{market}/boards/{board}/securities"
        params: Dict[str, Any] = {"iss.meta": "off"}
        if secids:
            params["securities"] = ",".join(s.upper() for s in secids)
        payload = self._get_json(self._public_client, path, params=params)
        blocks = parse_iss_envelope(payload)
        sec_rows = {r.get("secid"): r for r in blocks.get("securities", []) if r.get("secid")}
        md_rows = {r.get("secid"): r for r in blocks.get("marketdata", []) if r.get("secid")}
        merged: Dict[str, Dict[str, Any]] = {}
        for secid in set(sec_rows) | set(md_rows):
            entry = {}
            entry.update(sec_rows.get(secid, {}))
            entry.update(md_rows.get(secid, {}))
            merged[secid] = entry
        return merged


# ---------------------------------------------------------------------------
def _date_str(value: date | str | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
