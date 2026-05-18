"""SQLite database engine with WAL mode + storage abstraction.

Default backend is local SQLite. The :class:`Database` API only exposes a
SQLAlchemy ``Engine`` and a ``session()`` context manager, which makes it
trivial to swap for PostgreSQL in the future: switch the URL, keep the rest.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.storage.models import Base

_LOG = get_logger(__name__)


def _ensure_parent_dir(path: str) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)


def _make_sqlalchemy_url(db_path: str) -> str:
    if db_path.startswith("sqlite:") or "://" in db_path:
        return db_path
    return f"sqlite:///{db_path}"


def _apply_sqlite_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cur = dbapi_connection.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA foreign_keys=ON;")
            cur.execute("PRAGMA busy_timeout=5000;")
            cur.execute("PRAGMA cache_size=-20000;")  # ~20 MB page cache
        finally:
            cur.close()


class Database:
    """Thin wrapper around a SQLAlchemy engine with init-once semantics."""

    _lock = threading.Lock()
    _instance: Optional["Database"] = None

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        url = _make_sqlalchemy_url(db_path)
        self.url = url
        is_sqlite = url.startswith("sqlite")
        if is_sqlite and ":memory:" not in url:
            # Strip the sqlite:/// prefix for filesystem prep.
            fs_path = url.replace("sqlite:///", "", 1)
            _ensure_parent_dir(fs_path)
        connect_args = {"check_same_thread": False} if is_sqlite else {}
        self.engine: Engine = create_engine(
            url,
            future=True,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        if is_sqlite:
            _apply_sqlite_pragmas(self.engine)
        self._SessionLocal = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False, future=True
        )

    # ------------------------------------------------------------------
    def init_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    # ------------------------------------------------------------------
    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ------------------------------------------------------------------
    def healthcheck(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # pragma: no cover
            _LOG.error("db_healthcheck_failed", extra={"error": str(exc)})
            return False

    # ------------------------------------------------------------------
    def size_bytes(self) -> Optional[int]:
        url = self.url
        if not url.startswith("sqlite") or ":memory:" in url:
            return None
        fs_path = url.replace("sqlite:///", "", 1)
        try:
            return os.path.getsize(fs_path)
        except OSError:
            return None


def get_database(settings: Optional[Settings] = None) -> Database:
    """Module-level singleton accessor."""
    settings = settings or get_settings()
    with Database._lock:
        if Database._instance is None:
            db = Database(settings.db_path)
            db.init_schema()
            Database._instance = db
        return Database._instance


def reset_database(new_path: Optional[str] = None) -> Database:
    """Helper for tests: dispose the current engine and rebuild."""
    with Database._lock:
        if Database._instance is not None:
            try:
                Database._instance.engine.dispose()
            except Exception:
                pass
            Database._instance = None
        if new_path is None:
            new_path = get_settings().db_path
        db = Database(new_path)
        db.init_schema()
        Database._instance = db
        return db
