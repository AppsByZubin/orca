"""Database engine / session plumbing (SQLAlchemy 2.0, SQLite by default)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orca.config import PROJECT_ROOT, get_settings
from orca.logging_setup import get_logger

log = get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _normalize_url(url: str) -> str:
    """Make relative sqlite paths resolve against the project root, not cwd."""
    prefix = "sqlite:///./"
    if url.startswith(prefix):
        target = (PROJECT_ROOT / url[len(prefix) :]).resolve()
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{target}"
    return url


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = _normalize_url(get_settings().database_url)
        kwargs: dict = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if url in {"sqlite://", "sqlite:///:memory:"}:
                # One shared connection, otherwise every thread gets its own
                # empty in-memory database.
                kwargs["poolclass"] = StaticPool
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

        log.info("database: %s", url)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts, jobs and services."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def db_session() -> Iterator[Session]:
    """FastAPI dependency."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create tables. Swap for Alembic migrations once the schema settles."""
    from orca import models  # noqa: F401  (import registers the mappers)
    from orca.models.base import Base

    Base.metadata.create_all(bind=get_engine())
    log.info("schema ready (%d tables)", len(Base.metadata.tables))
