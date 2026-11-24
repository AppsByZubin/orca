"""The order-placement contract (Zerodha Kite in production, paper in dev)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from orca.domain import OrderStatus


@dataclass
class OrderRequest:
    tradingsymbol: str
    exchange: str  # NFO (NSE options) | BFO (BSE options)
    transaction_type: str  # BUY | SELL
    quantity: int
    order_type: str = "MARKET"  # MARKET | LIMIT | SL | SL-M
    price: float | None = None
    trigger_price: float | None = None
    product: str = "NRML"  # NRML carry-forward | MIS intraday
    validity: str = "DAY"
    tag: str = "orca"
    instrument_key: str = ""  # feed-side id, used by the paper gateway to price fills
    purpose: str = "ENTRY"  # ENTRY | EXIT | ADJUST


@dataclass
class OrderResult:
    ok: bool
    order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float | None = None
    message: str = ""
    raw: dict = field(default_factory=dict)


class OrderGateway(ABC):
    """Anything that can put orders in the market."""

    name: str = "base"
    live: bool = False

    @abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResult: ...

    def place_basket(self, requests: list[OrderRequest]) -> list[OrderResult]:
        """Default: sequential. Order matters - buy protection before selling.

        Gateways with a real basket/multi-leg endpoint should override this.
        """
        ordered = sorted(requests, key=lambda r: 0 if r.transaction_type == "BUY" else 1)
        return [self.place_order(r) for r in ordered]

    @abstractmethod
    def get_order(self, order_id: str) -> OrderResult: ...

    def cancel_order(self, order_id: str) -> OrderResult:
        return OrderResult(ok=False, message="cancel not supported by this gateway")

    def get_positions(self) -> list[dict]:
        return []

    def basket_margin(self, requests: list[OrderRequest]) -> float | None:
        """Broker-computed margin for the whole basket, if available."""
        return None

    def health(self) -> dict:
        return {"gateway": self.name, "live": self.live, "ok": True}


class ExecutionError(RuntimeError):
    pass


class TradingDisabled(ExecutionError):
    """Raised when a live order is attempted with ORCA_TRADING_ENABLED=false."""
