"""Order gateways + the factory that picks one from config."""

from __future__ import annotations

from orca.config import get_settings
from orca.execution.base import (
    ExecutionError,
    OrderGateway,
    OrderRequest,
    OrderResult,
    TradingDisabled,
)
from orca.execution.paper import PaperGateway
from orca.logging_setup import get_logger

log = get_logger(__name__)

__all__ = [
    "ExecutionError",
    "OrderGateway",
    "OrderRequest",
    "OrderResult",
    "PaperGateway",
    "TradingDisabled",
    "get_gateway",
    "reset_gateway",
]

_cached: OrderGateway | None = None


def build_gateway(name: str | None = None) -> OrderGateway:
    name = (name or get_settings().execution_gateway).lower()
    if name == "kite":
        from orca.execution.kite import KiteGateway

        try:
            return KiteGateway()
        except ExecutionError as exc:
            log.error("kite gateway unavailable (%s); falling back to paper", exc)
            return PaperGateway()
    if name == "paper":
        return PaperGateway()
    raise ValueError(f"unknown execution gateway {name!r} (expected 'kite' or 'paper')")


_cached_token: str | None = None  # the Kite token _cached was built with


def _kite_token() -> str | None:
    if get_settings().execution_gateway.lower() != "kite":
        return None
    from orca.services.broker_tokens import current_token

    return current_token("kite")


def get_gateway() -> OrderGateway:
    """Process-wide gateway, rebuilt when the stored Kite token changes."""
    global _cached, _cached_token
    token = _kite_token()
    if _cached is not None and token != _cached_token:
        log.info("kite token changed; rebuilding the order gateway")
        _cached = None
    if _cached is None:
        _cached_token = token
        _cached = build_gateway()
        log.info("execution gateway: %s (live=%s)", _cached.name, _cached.live)
    return _cached


def reset_gateway() -> None:
    global _cached
    _cached = None
