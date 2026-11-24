"""Shared fixtures: in-memory DB, mock feed, paper gateway."""

from __future__ import annotations

import os
from datetime import date, timedelta

import pytest

os.environ.setdefault("ORCA_DATABASE_URL", "sqlite://")  # in-memory
os.environ.setdefault("ORCA_FEED_PROVIDER", "mock")
os.environ.setdefault("ORCA_EXECUTION_GATEWAY", "paper")
os.environ.setdefault("ORCA_SCHEDULER_ENABLED", "false")
os.environ.setdefault("ORCA_MAX_OPEN_TRADES", "10")


@pytest.fixture(scope="session", autouse=True)
def _settings_env():
    from orca.config import reload_config

    reload_config()
    yield


@pytest.fixture
def session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from orca import models  # noqa: F401
    from orca.models.base import Base

    engine = create_engine("sqlite://", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    db = factory()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def feed():
    from orca.feeds.mock import MockProvider

    return MockProvider(seed="test")


@pytest.fixture
def nifty():
    from orca.config import get_universe

    return get_universe().get("NIFTY")


@pytest.fixture
def chain(feed, nifty):
    expiry = date.today() + timedelta(days=3)
    return feed.get_option_chain(nifty, expiry)
