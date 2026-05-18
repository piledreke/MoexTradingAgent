"""Arenago read-only client.

Hard guarantees:

* ``submit_order`` and any other write endpoints are **forbidden**. We do not
  even expose a method for them. :meth:`submit_order` exists only to raise a
  loud :class:`RuntimeError` if some caller tries to "monkey-patch" the agent
  into trading – it is documented as forbidden.
* Reads are: ``GET /api/positions/<portfolio>``, ``GET /api/trades/<portfolio>``,
  ``GET /api/bots``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.utils.retry import retry_call

_LOG = get_logger(__name__)


class ArenagoForbiddenError(RuntimeError):
    """Raised whenever a forbidden Arenago method is invoked."""


class ArenagoClient:
    """Read-only client used to enrich recommendations with portfolio state."""

    FORBIDDEN_PATHS = (
        "/api/submit_order",
        "/api/cancel_order",
        "/api/orders/submit",
        "/api/orders/cancel",
    )

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._token = self.settings.arenago_token
        self._base_url = self.settings.arenago_base_url.rstrip("/")
        self._timeout = self.settings.arenago_timeout
        headers = {"User-Agent": "moex-tech-agent/0.1"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=self._timeout,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self._token)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ArenagoClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status in (401, 403, 404):
                return False
            if 400 <= status < 500:
                return False
            return True
        return True

    def _get(self, path: str) -> Optional[Any]:
        if not self.enabled:
            return None
        # Pre-flight forbidden-path guard.
        for forbidden in self.FORBIDDEN_PATHS:
            if path.startswith(forbidden):
                raise ArenagoForbiddenError(
                    f"Arenago call to {path} is forbidden; technical agent does not trade"
                )

        def _call() -> Any:
            r = self._client.get(path)
            r.raise_for_status()
            return r.json()

        try:
            return retry_call(_call, attempts=3, retry_on=(httpx.HTTPError,), is_retryable=self._is_retryable)
        except Exception as exc:
            _LOG.warning("arena_request_failed", extra={"path": path, "error": str(exc)})
            return None

    # ------------------------------------------------------------------
    def get_positions(self, portfolio: str) -> Optional[List[Dict[str, Any]]]:
        data = self._get(f"/api/positions/{portfolio}")
        return _ensure_list(data)

    def get_trades(self, portfolio: str) -> Optional[List[Dict[str, Any]]]:
        data = self._get(f"/api/trades/{portfolio}")
        return _ensure_list(data)

    def get_bots(self) -> Optional[List[Dict[str, Any]]]:
        data = self._get("/api/bots")
        return _ensure_list(data)

    # ------------------------------------------------------------------
    # NEVER call this. It exists only as a tripwire.
    # ------------------------------------------------------------------
    def submit_order(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise ArenagoForbiddenError(
            "submit_order is forbidden: the technical advisor never places trades. "
            "It only returns recommendations to the main trading agent."
        )


def _ensure_list(data: Any) -> Optional[List[Dict[str, Any]]]:
    if data is None:
        return None
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        # Some APIs return ``{"items": [...]}`` envelopes.
        for k in ("items", "data", "result", "positions", "trades", "bots"):
            if isinstance(data.get(k), list):
                return [r for r in data[k] if isinstance(r, dict)]
        return [data]
    return None
