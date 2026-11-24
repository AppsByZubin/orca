"""Paper gateway: simulates fills against the live feed. The default.

Nothing here touches a broker, so the full entry -> tracking -> exit loop can
be exercised safely, including during market hours.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime

from orca.domain import OrderStatus
from orca.execution.base import OrderGateway, OrderRequest, OrderResult
from orca.logging_setup import get_logger

log = get_logger(__name__)

SLIPPAGE_PCT = 0.002  # 20 bps against us, per leg


class PaperGateway(OrderGateway):
    name = "paper"
    live = False

    def __init__(self, feed=None) -> None:  # noqa: ANN001 - MarketDataProvider, lazy to avoid cycle
        self._feed = feed
        self._seq = itertools.count(1)
        self._orders: dict[str, OrderResult] = {}

    @property
    def feed(self):  # noqa: ANN201
        if self._feed is None:
            from orca.feeds import get_feed

            self._feed = get_feed()
        return self._feed

    def _reference_price(self, request: OrderRequest) -> float:
        if request.order_type == "LIMIT" and request.price:
            return float(request.price)
        if request.instrument_key:
            quote = self.feed.get_quotes([request.instrument_key]).get(request.instrument_key)
            if quote and quote.ltp:
                return quote.ltp
        return float(request.price or 0.0)

    def place_order(self, request: OrderRequest) -> OrderResult:
        price = self._reference_price(request)
        # Buys fill a touch higher, sells a touch lower.
        drift = SLIPPAGE_PCT if request.transaction_type == "BUY" else -SLIPPAGE_PCT
        fill = round(max(price * (1 + drift), 0.05), 2)
        order_id = f"PAPER-{datetime.now(UTC):%Y%m%d}-{next(self._seq):05d}"
        result = OrderResult(
            ok=True,
            order_id=order_id,
            status=OrderStatus.COMPLETE,
            filled_quantity=request.quantity,
            average_price=fill,
            message="paper fill",
            raw={
                "tradingsymbol": request.tradingsymbol,
                "transaction_type": request.transaction_type,
                "reference_price": price,
            },
        )
        self._orders[order_id] = result
        log.info(
            "paper %s %s x%d @ %.2f",
            request.transaction_type,
            request.tradingsymbol,
            request.quantity,
            fill,
        )
        return result

    def get_order(self, order_id: str) -> OrderResult:
        return self._orders.get(
            order_id, OrderResult(ok=False, message=f"unknown order {order_id}")
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        result = self._orders.get(order_id)
        if not result:
            return OrderResult(ok=False, message="unknown order")
        result.status = OrderStatus.CANCELLED
        return result

    def basket_margin(self, requests: list[OrderRequest]) -> float | None:
        return None  # analytics.margin.estimate_margin is used instead
