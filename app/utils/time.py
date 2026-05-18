"""Time helpers. MOEX trading is anchored to Europe/Moscow."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    MSK = ZoneInfo("Europe/Moscow")
except Exception:  # pragma: no cover - fallback for unusual envs
    MSK = timezone(timedelta(hours=3))


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def now_msk() -> datetime:
    return datetime.now(tz=MSK)


def to_msk(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=MSK)
    return dt.astimezone(MSK)


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=MSK).astimezone(timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_moex_datetime(value: Optional[str | datetime | date]) -> Optional[datetime]:
    """Parse a MOEX-style date / time / datetime string.

    Returns a *naive MSK* datetime (because that is how ISS reports them).
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    text = str(value).strip()
    if not text:
        return None
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%H:%M:%S",
        "%H:%M",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def merge_date_time(d: Optional[str], t: Optional[str]) -> Optional[datetime]:
    """Combine ISS ``tradedate`` and ``tradetime`` strings into one MSK datetime."""
    if not d:
        return None
    if not t:
        t = "00:00:00"
    return parse_moex_datetime(f"{d} {t}")


def msk_today() -> date:
    return now_msk().date()


def msk_yesterday() -> date:
    return now_msk().date() - timedelta(days=1)


def is_msk_trading_window(dt: Optional[datetime] = None) -> bool:
    """Rough check whether MSK now() is inside the equity main session.

    MOEX TQBR main session is 09:50 - 18:50 MSK, plus evening session
    19:05 - 23:50. This is intentionally permissive (the strategy itself
    also checks freshness of data).
    """
    dt = to_msk(dt) if dt else now_msk()
    if dt.weekday() >= 5:
        return False
    hm = dt.hour * 60 + dt.minute
    main_open = 9 * 60 + 50
    main_close = 18 * 60 + 50
    evening_open = 19 * 60 + 5
    evening_close = 23 * 60 + 50
    return (main_open <= hm <= main_close) or (evening_open <= hm <= evening_close)


def age_seconds(dt: Optional[datetime]) -> Optional[float]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=MSK)
    return (now_msk() - dt.astimezone(MSK)).total_seconds()
