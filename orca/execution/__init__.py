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


def get_gateway() -> OrderGateway:
    global _cached
    if _cached is None:
        _cached = build_gateway()
        log.info("execution gateway: %s (live=%s)", _cached.name, _cached.live)
    return _cached


def reset_gateway() -> None:
    global _cached
    _cached = None
