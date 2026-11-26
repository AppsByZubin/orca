"""Market-data providers + the factory that picks one from config."""

from __future__ import annotations

from orca.config import get_settings
from orca.feeds.base import FeedError, MarketDataProvider
from orca.feeds.mock import MockProvider
from orca.logging_setup import get_logger

log = get_logger(__name__)

__all__ = ["FeedError", "MarketDataProvider", "MockProvider", "get_feed", "reset_feed"]

_cached: MarketDataProvider | None = None
_cached_token: str | None = None  # the Upstox token _cached was built with


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


def _upstox_token() -> str | None:
    if get_settings().feed_provider.lower() != "upstox":
        return None
    from orca.services.broker_tokens import current_token

    return current_token("upstox")


def get_feed() -> MarketDataProvider:
    """Process-wide feed singleton.

    Rebuilt when the stored Upstox token changes (a new one saved, or the old
    one expired), so a token entered in the UI takes effect without a restart
    - including recovering from the mock fallback used while there was none.
    """
    global _cached, _cached_token
    token = _upstox_token()
    if _cached is not None and token != _cached_token:
        log.info("upstox token changed; rebuilding the market feed")
        reset_feed()
    if _cached is None:
        _cached_token = token
        _cached = build_feed()
        log.info("market feed: %s", _cached.name)
    return _cached


def reset_feed() -> None:
    global _cached
    if _cached is not None:
        _cached.close()
    _cached = None
