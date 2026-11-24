"""Market-data providers + the factory that picks one from config."""

from __future__ import annotations

from orca.config import get_settings
from orca.feeds.base import FeedError, MarketDataProvider
from orca.feeds.mock import MockProvider
from orca.logging_setup import get_logger

log = get_logger(__name__)

__all__ = ["FeedError", "MarketDataProvider", "MockProvider", "get_feed", "reset_feed"]

_cached: MarketDataProvider | None = None


def build_feed(provider: str | None = None) -> MarketDataProvider:
    name = (provider or get_settings().feed_provider).lower()
    if name == "upstox":
        from orca.feeds.upstox import UpstoxProvider

        try:
            return UpstoxProvider()
        except FeedError as exc:
            log.error("upstox feed unavailable (%s); falling back to mock", exc)
            return MockProvider()
    if name == "mock":
        return MockProvider()
    raise ValueError(f"unknown feed provider {name!r} (expected 'upstox' or 'mock')")


def get_feed() -> MarketDataProvider:
    """Process-wide feed singleton."""
    global _cached
    if _cached is None:
        _cached = build_feed()
        log.info("market feed: %s", _cached.name)
    return _cached


def reset_feed() -> None:
    global _cached
    if _cached is not None:
        _cached.close()
    _cached = None
