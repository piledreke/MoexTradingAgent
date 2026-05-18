"""Retry helpers with exponential backoff."""
from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import aiohttp

from utils.logger import logger

T = TypeVar("T")

RETRYABLE_EXC = (
    aiohttp.ClientConnectionError,
    aiohttp.ClientPayloadError,
    aiohttp.ServerDisconnectedError,
    asyncio.TimeoutError,
)


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    attempts: int = 3,
    base: float = 1.0,
    max_wait: float = 10.0,
    **kwargs: Any,
) -> Any:
    """Run async callable with retry on transient errors."""
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=base, max=max_wait),
            retry=retry_if_exception_type(RETRYABLE_EXC),
            reraise=True,
        ):
            with attempt:
                return await func(*args, **kwargs)
    except RetryError as e:
        logger.error(f"retry_async exhausted: {e}")
        raise