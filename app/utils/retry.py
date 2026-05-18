"""Tiny retry helpers used across HTTP clients."""

from __future__ import annotations

import random
import time
from typing import Callable, Iterable, Optional, Type, TypeVar

T = TypeVar("T")


def exponential_backoff_iter(
    attempts: int,
    base: float = 0.5,
    cap: float = 8.0,
    jitter: float = 0.25,
) -> Iterable[float]:
    """Yield retry delays in seconds for ``attempts`` retries (not including the first try)."""
    for i in range(attempts):
        delay = min(cap, base * (2 ** i))
        delay += random.uniform(0, jitter)
        yield delay


def retry_call(
    fn: Callable[[], T],
    attempts: int = 4,
    retry_on: tuple[Type[BaseException], ...] = (Exception,),
    is_retryable: Optional[Callable[[BaseException], bool]] = None,
    base: float = 0.5,
    cap: float = 8.0,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> T:
    """Call ``fn`` with up to ``attempts`` total tries and exponential backoff.

    The function is invoked once initially; on failure it retries up to
    ``attempts - 1`` additional times.
    """
    last_exc: Optional[BaseException] = None
    delays = list(exponential_backoff_iter(max(attempts - 1, 0), base=base, cap=cap))
    for attempt in range(attempts):
        try:
            return fn()
        except retry_on as exc:  # type: ignore[misc]
            last_exc = exc
            retryable = is_retryable(exc) if is_retryable else True
            if not retryable or attempt >= attempts - 1:
                raise
            delay = delays[attempt] if attempt < len(delays) else cap
            if on_retry is not None:
                try:
                    on_retry(attempt + 1, exc, delay)
                except Exception:
                    pass
            time.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError("retry_call exited without value")  # pragma: no cover
