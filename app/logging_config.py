"""Structured logging configured once per process.

Emits JSON to stdout by default so the agent plays nicely with Docker log
collection (Datadog, Loki, etc.). Set ``LOG_JSON=false`` for human format.
"""

from __future__ import annotations

import logging
import logging.config
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import orjson


_LOGGING_INITIALIZED = False
_BASE_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Minimal, allocation-light JSON formatter using orjson."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: Dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k in _BASE_RESERVED or k.startswith("_"):
                continue
            try:
                orjson.dumps(v)
                payload[k] = v
            except Exception:
                payload[k] = repr(v)
        return orjson.dumps(payload).decode("utf-8")


class PlainFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Initialise logging only once per process."""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else PlainFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    try:
        root.setLevel(level.upper())
    except ValueError:
        root.setLevel(logging.INFO)
    # Tone down noisy libraries.
    for noisy in ("httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(max(root.level, logging.WARNING))
    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
