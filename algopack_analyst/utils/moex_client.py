"""Unified async client for MOEX ISS + ALGOPACK (skills S1—S18)."""
from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import aiohttp
import pandas as pd
import pytz
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from utils.logger import logger

MSK = pytz.timezone("Europe/Moscow")


class MoexAPIError(Exception):
    """Raised when MOEX ISS / ALGOPACK returns non-recoverable error."""


class MoexClient:
    """Unified async client for MOEX ISS + ALGOPACK endpoints (S1—S18).

    Automatically falls back to public ISS when ALGOPACK token is absent.
    """

    def __init__(
        self,
        algopack_token: str | None = None,
        max_concurrent: int = 10,
        timeout: int = 30,
        cache_ttl: int = 5,
    ) -> None:
        self._token = algopack_token or settings.MOEX_ALGOPACK_TOKEN
        self._algopack_base = settings.MOEX_ALGOPACK_BASE
        self._public_base = settings.MOEX_PUBLIC_BASE
        self._sem = asyncio.Semaphore(max_concurrent)
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, Any]] = {}
        self._warn_cache: dict[str, float] = {}
        self._session: aiohttp.ClientSession | None = None

        if not self._token:
            logger.warning(
                "ALGOPACK token not set — falling back to public ISS (15-min delay)"
            )

    async def __aenter__(self) -> "MoexClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=settings.MOEX_MAX_CONCURRENT)
            self._session = aiohttp.ClientSession(
                timeout=self._timeout, connector=connector
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ───────────────────────────── Low level ────────────────────────────
    async def request(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        use_algopack: bool | None = None,
    ) -> dict[str, Any]:
        """Perform GET request to MOEX ISS or ALGOPACK.

        Args:
            path: Path component starting with `/iss/...`.
            params: Query parameters.
            use_algopack: Force ALGOPACK base (with auth). Falls back to public
                if token missing.

        Returns:
            Parsed JSON response.
        """
        params = params or {}
        # Do not force ISS query params here; callers may pass params when needed.

        if use_algopack is None:
            use_algopack = bool(self._token)

        base = self._algopack_base if (use_algopack and self._token) else self._public_base
        url = f"{base}{path}"
        cache_key = f"{url}?{urlencode(sorted(params.items()))}"

        # Cache lookup
        if cache_key in self._cache:
            ts, val = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return val

        headers: dict[str, str] = {"Accept": "application/json"}
        if use_algopack and self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        session = await self._ensure_session()

        async def _do_request() -> dict[str, Any]:
            t0 = time.perf_counter()
            async with self._sem:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 429:
                        ra = int(resp.headers.get("Retry-After", "5"))
                        logger.warning(f"429 from MOEX, sleeping {ra}s")
                        await asyncio.sleep(ra)
                        raise aiohttp.ClientError("rate_limited")
                    if resp.status >= 500:
                        raise aiohttp.ClientError(f"server_error_{resp.status}")
                    if resp.status >= 400:
                        body = await resp.text()
                        raise MoexAPIError(f"{resp.status}: {body[:200]}")
                    try:
                        data = await resp.json(content_type=None)
                    except Exception as parse_err:
                        body = await resp.text()
                        self._warn_once(
                            f"non_json:{url}:{resp.status}",
                            f"MOEX returned non-JSON response (status={resp.status}): {body[:400]!r}",
                            ttl=300,
                        )
                        raise aiohttp.ClientError(f"invalid_json_response: {parse_err}")
                    dt = (time.perf_counter() - t0) * 1000
                    logger.debug(
                        f"MOEX GET {path} status={resp.status} "
                        f"dt={dt:.0f}ms params={params}"
                    )
                    return data

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, max=10),
                retry=retry_if_exception_type(
                    (aiohttp.ClientError, asyncio.TimeoutError)
                ),
                reraise=True,
            ):
                with attempt:
                    data = await _do_request()
        except Exception as e:
            if use_algopack and self._token:
                self._warn_once(
                    f"algopack_failed:{path}",
                    f"ALGOPACK failed ({e}); public fallback disabled",
                    ttl=300,
                )
            raise

        self._cache[cache_key] = (time.time(), data)
        return data

    # ───────────────────────────── Parser ───────────────────────────────
    @staticmethod
    def to_dataframe(payload: dict[str, Any], block: str = "securities") -> pd.DataFrame:
        """Parse MOEX ISS response block into typed DataFrame.

        ISS `iss.json=extended` returns list of dicts per block. The classic
        format also supported (columns + data).
        """
        # Extended: payload is list of blocks dicts
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and block in item:
                    rows = item[block]
                    if isinstance(rows, list):
                        df = pd.DataFrame(rows)
                        return _coerce_numerics(df)
            return pd.DataFrame()

        # Classic format
        if isinstance(payload, dict) and block in payload:
            section = payload[block]
            if isinstance(section, dict) and "columns" in section and "data" in section:
                df = pd.DataFrame(section["data"], columns=section["columns"])
                return _coerce_numerics(df)
            if isinstance(section, list):
                return _coerce_numerics(pd.DataFrame(section))
        return pd.DataFrame()

    # ═════════════════════════ S1—S18 methods ═══════════════════════════

    # [S1] All shares TQBR
    async def get_all_shares(self) -> pd.DataFrame:
        data = await self.request(
            "/engines/stock/markets/shares/boards/tqbr/securities.json"
        )
        return self.to_dataframe(data, "securities")

    # [S2] Single share snapshot
    async def get_share(self, ticker: str) -> dict[str, Any]:
        data = await self.request(
            f"/engines/stock/markets/shares/boards/tqbr/securities/{ticker}.json"
        )
        df_sec = self.to_dataframe(data, "securities")
        df_md = self.to_dataframe(data, "marketdata")
        return {
            "securities": df_sec.iloc[0].to_dict() if not df_sec.empty else {},
            "marketdata": df_md.iloc[0].to_dict() if not df_md.empty else {},
        }

    # [S3] OHLCV candles
    async def get_candles(
        self,
        ticker: str,
        *,
        interval: int = 60,
        from_: str,
        till: str,
    ) -> pd.DataFrame:
        params = {"from": from_, "till": till, "interval": interval}
        data = await self.request(
            f"/engines/stock/markets/shares/boards/tqbr/securities/{ticker}/candles.json",
            params=params,
        )
        df = self.to_dataframe(data, "candles")
        if not df.empty and "begin" in df.columns:
            df["ts"] = pd.to_datetime(df["begin"]).dt.tz_localize(MSK, nonexistent="shift_forward")
        return df

    # [S4] Orderbook (20 levels)
    async def get_orderbook(self, ticker: str) -> dict[str, Any]:
        # Public ISS returns an HTML denial page for this endpoint in our environment.
        # Prefer ALGOPACK when a token is available; otherwise fall back to public.
        data = await self.request(
            f"/engines/stock/markets/shares/boards/tqbr/securities/{ticker}/orderbook.json",
            use_algopack=bool(self._token),
        )
        df = self.to_dataframe(data, "orderbook")
        bids: list[tuple[float, float]] = []
        asks: list[tuple[float, float]] = []
        if not df.empty:
            for _, row in df.iterrows():
                price = float(row.get("PRICE", 0) or 0)
                qty = float(row.get("QUANTITY", 0) or 0)
                side = row.get("BUYSELL")
                if side == "B":
                    bids.append((price, qty))
                elif side == "S":
                    asks.append((price, qty))
        bids.sort(key=lambda x: -x[0])
        asks.sort(key=lambda x: x[0])
        return {
            "ticker": ticker,
            "ts": datetime.now(MSK).isoformat(),
            "bids": bids,
            "asks": asks,
        }

    # [S5] Trades tape
    async def get_trades(self, ticker: str, tradeno: int = 0) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if tradeno:
            params["tradeno"] = tradeno
        data = await self.request(
            f"/engines/stock/markets/shares/boards/tqbr/securities/{ticker}/trades.json",
            params=params,
        )
        return self.to_dataframe(data, "trades")

    # [S6] Tradestats (Super Candles)
    async def get_tradestats(
        self,
        ticker: str | None = None,
        *,
        market: str = "eq",
        date: str | None = None,
        from_: str | None = None,
        till: str | None = None,
        latest: bool = False,
        limit: int | None = None,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if date:
            params["date"] = date
        if from_:
            params["from"] = from_
        if till:
            params["till"] = till
        if latest:
            params["latest"] = 1
        if limit:
            params["limit"] = limit

        suffix = f"/{ticker}" if ticker else ""
        path = f"/datashop/algopack/{market}/tradestats{suffix}.json"
        data = await self.request(path, params=params, use_algopack=True)
        return self.to_dataframe(data, "data")

    # [S7] Obstats
    async def get_obstats(
        self,
        ticker: str,
        *,
        market: str = "eq",
        date: str | None = None,
        from_: str | None = None,
        till: str | None = None,
        latest: bool = False,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if date:
            params["date"] = date
        if from_:
            params["from"] = from_
        if till:
            params["till"] = till
        if latest:
            params["latest"] = 1
        path = f"/datashop/algopack/{market}/obstats/{ticker}.json"
        data = await self.request(path, params=params, use_algopack=True)
        return self.to_dataframe(data, "data")

    # [S8] Orderstats
    async def get_orderstats(
        self,
        ticker: str,
        *,
        market: str = "eq",
        date: str | None = None,
        from_: str | None = None,
        till: str | None = None,
        latest: bool = False,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if date:
            params["date"] = date
        if from_:
            params["from"] = from_
        if till:
            params["till"] = till
        if latest:
            params["latest"] = 1
        path = f"/datashop/algopack/{market}/orderstats/{ticker}.json"
        data = await self.request(path, params=params, use_algopack=True)
        return self.to_dataframe(data, "data")

    # [S9] HI2 — concentration
    async def get_hi2(
        self,
        ticker: str,
        *,
        market: str = "eq",
        from_: str | None = None,
        till: str | None = None,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if from_:
            params["from"] = from_
        if till:
            params["till"] = till
        path = f"/datashop/algopack/{market}/hi2/{ticker}.json"
        data = await self.request(path, params=params, use_algopack=True)
        return self.to_dataframe(data, "data")

    # [S10] Mega Alerts
    async def get_alerts(
        self,
        ticker: str,
        *,
        market: str = "eq",
        from_: str | None = None,
        till: str | None = None,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if from_:
            params["from"] = from_
        if till:
            params["till"] = till
        path = f"/datashop/algopack/{market}/alerts/{ticker}.json"
        data = await self.request(path, params=params, use_algopack=True)
        return self.to_dataframe(data, "data")

    # [S11] FUTOI
    async def get_futoi(
        self,
        ticker: str,
        from_: str | None = None,
        till: str | None = None,
        latest: bool = False,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if from_:
            params["from"] = from_
        if till:
            params["till"] = till
        if latest:
            params["latest"] = 1
        path = f"/analyticalproducts/futoi/securities/{ticker}.json"
        data = await self.request(path, params=params, use_algopack=True)
        return self.to_dataframe(data, "futoi")

    # [S12] Calendar of non-trading days
    async def get_calendar(
        self,
        from_: str | None = None,
        till: str | None = None,
        show_all_days: bool = False,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if from_:
            params["from"] = from_
        if till:
            params["till"] = till
        if show_all_days:
            params["show_all_days"] = 1
        data = await self.request("/calendars/stock.json", params=params)
        return self.to_dataframe(data, "calendar")

    # [S13] Today session schedule
    async def get_session_schedule(self) -> pd.DataFrame:
        data = await self.request("/calendars/stock/session.json")
        return self.to_dataframe(data, "session")

    # [S14] Suspensions
    async def get_suspensions(
        self, from_: str | None = None, till: str | None = None
    ) -> pd.DataFrame:
        params: dict[str, Any] = {}
        if from_:
            params["from"] = from_
        if till:
            params["till"] = till
        data = await self.request(
            "/calendars/stock/securities/suspended/details.json", params=params
        )
        return self.to_dataframe(data, "suspended")

    # [S15] Boards
    async def get_boards(self, ticker: str) -> pd.DataFrame:
        data = await self.request(f"/securities/{ticker}.json")
        return self.to_dataframe(data, "boards")

    # [S16] Index candles
    async def get_index_candles(
        self,
        index: str,
        *,
        interval: int = 60,
        from_: str | None = None,
        till: str | None = None,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {"interval": interval}
        if from_:
            params["from"] = from_
        if till:
            params["till"] = till
        data = await self.request(
            f"/engines/stock/markets/index/securities/{index}/candles.json",
            params=params,
        )
        df = self.to_dataframe(data, "candles")
        if not df.empty and "begin" in df.columns:
            df["ts"] = pd.to_datetime(df["begin"])
        return df

    # [S17] Security card
    async def get_security_info(self, ticker: str) -> dict[str, Any]:
        data = await self.request(f"/securities/{ticker}.json")
        df = self.to_dataframe(data, "description")
        if df.empty:
            return {}
        return dict(zip(df["name"], df["value"])) if "name" in df.columns else {}

    # [S18] Dividends
    async def get_dividends(self, ticker: str) -> pd.DataFrame:
        data = await self.request(f"/securities/{ticker}/dividends.json")
        return self.to_dataframe(data, "dividends")

    # ───────────────────────────── Guards ───────────────────────────────
    async def is_market_open(self) -> bool:
        """Check if MOEX stock market is currently in trading session."""
        try:
            df = await self.get_session_schedule()
            if df.empty:
                # Fallback: working hours heuristic
                now = datetime.now(MSK)
                return (
                    now.weekday() < 5
                    and (now.hour > 9 or (now.hour == 9 and now.minute >= 50))
                    and now.hour < 19
                )
            # If any row indicates trading in progress
            now = datetime.now(MSK)
            for _, row in df.iterrows():
                start = row.get("start")
                end = row.get("end")
                if start and end:
                    try:
                        s = pd.to_datetime(start).tz_localize(MSK) if pd.to_datetime(start).tzinfo is None else pd.to_datetime(start)
                        e = pd.to_datetime(end).tz_localize(MSK) if pd.to_datetime(end).tzinfo is None else pd.to_datetime(end)
                        if s <= now <= e:
                            return True
                    except Exception:
                        continue
            return False
        except Exception as e:
            self._warn_once(
                "is_market_open_failed",
                f"is_market_open failed: {e}, defaulting to heuristic",
                ttl=300,
            )
            now = datetime.now(MSK)
            return now.weekday() < 5 and 10 <= now.hour < 19

    async def is_trading_suspended(self, ticker: str) -> bool:
        try:
            today = datetime.now(MSK).strftime("%Y-%m-%d")
            df = await self.get_suspensions(from_=today, till=today)
            if df.empty:
                return False
            cols = [c for c in df.columns if "sec" in c.lower() or "ticker" in c.lower()]
            for c in cols:
                if (df[c].astype(str).str.upper() == ticker.upper()).any():
                    return True
            return False
        except Exception as e:
            logger.warning(f"is_trading_suspended failed: {e}")
            return False

    async def is_tradable_tqbr(self, ticker: str) -> bool:
        try:
            df = await self.get_boards(ticker)
            if df.empty:
                return False
            if "boardid" in df.columns:
                return (df["boardid"].astype(str).str.upper() == "TQBR").any()
            return False
        except Exception:
            return False

    def _warn_once(self, key: str, message: str, ttl: int = 300) -> None:
        now = time.time()
        last = self._warn_cache.get(key)
        if last is None or (now - last) >= ttl:
            self._warn_cache[key] = now
            logger.warning(message)


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric-looking columns to numeric dtype.

    Strategy:
    - For object dtype columns, attempt `pd.to_numeric(..., errors='coerce')`.
    - If at least 50% of values convert to numeric, keep the converted column
      (NaNs are allowed). Otherwise leave the original column intact.
    """
    if df.empty:
        return df
    for col in df.columns:
        if df[col].dtype != object:
            continue
        with contextlib.suppress(ValueError, TypeError):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() / max(1, len(converted)) >= 0.5:
                df[col] = converted
    return df


# Singleton convenience
_client: MoexClient | None = None


def get_moex_client() -> MoexClient:
    global _client
    if _client is None:
        _client = MoexClient(
            algopack_token=settings.MOEX_ALGOPACK_TOKEN,
            max_concurrent=settings.MOEX_MAX_CONCURRENT,
            timeout=settings.MOEX_TIMEOUT,
            cache_ttl=settings.MOEX_CACHE_TTL,
        )
    return _client