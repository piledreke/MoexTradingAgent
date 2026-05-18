"""Structured logging via loguru (JSON format)."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from config import settings


def setup_logger() -> None:
    """Configure loguru with console + rotating JSON file sinks."""
    _logger.remove()

    _logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{function}:{line}</cyan> | "
            "<level>{message}</level>"
        ),
        enqueue=True,
    )

    log_file = Path(settings.LOG_DIR) / "analyst.log"
    _logger.add(
        str(log_file),
        level=settings.LOG_LEVEL,
        rotation=f"{settings.LOG_ROTATION_MB} MB",
        retention=settings.LOG_RETENTION_COUNT,
        compression="gz",
        serialize=True,  # JSON
        enqueue=True,
    )


setup_logger()
logger = _logger