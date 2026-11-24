"""Multi-leg trade book: Trade -> TradeLeg -> BrokerOrder, plus MTM history.

A Trade is the *strategy* (e.g. one NIFTY short strangle). Everything about
P&L, greeks and exits is aggregated at this level, which is what makes
multi-leg tracking work: legs can be exited independently, but risk is always
read as one position.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from orca.domain import LegStatus, Side, TradeStatus
from orca.models.base import Base, TimestampMixin, utcnow


class Trade(Base, TimestampMixin):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    underlying: Mapped[str] = mapped_column(String(32), index=True)
    expiry: Mapped[date] = mapped_column(Date, index=True)
    strategy_id: Mapped[str] = mapped_column(String(48), default="manual")
    template: Mapped[str] = mapped_column(String(48), default="custom")
    status: Mapped[str] = mapped_column(String(16), default=TradeStatus.DRAFT.value, index=True)

    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"), nullable=True)
    lots: Mapped[int] = mapped_column(Integer, default=1)
    lot_size: Mapped[int] = mapped_column(Integer, default=1)

    entry_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    entry_spot: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_spot: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Cash: credit is positive when premium was received.
    entry_credit: Mapped[float] = mapped_column(Float, default=0.0)  # total, all lots
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    charges: Mapped[float] = mapped_column(Float, default=0.0)
    margin_blocked: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Risk management
    stop_loss_amount: Mapped[float | None] = mapped_column(Float, nullable=True)  # abs INR loss
    target_amount: Mapped[float | None] = mapped_column(Float, nullable=True)  # abs INR profit
    trail_to_breakeven: Mapped[bool] = mapped_column(Boolean, default=False)
    auto_exit: Mapped[bool] = mapped_column(Boolean, default=False)
    max_profit_seen: Mapped[float] = mapped_column(Float, default=0.0)
    max_loss_seen: Mapped[float] = mapped_column(Float, default=0.0)

    # Live aggregates, refreshed by the tracker
    net_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_gamma: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_theta: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_vega: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_tracked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    alerts: Mapped[list] = mapped_column(JSON, default=list)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    legs: Mapped[list[TradeLeg]] = relationship(
        back_populates="trade", cascade="all, delete-orphan", lazy="selectin"
    )
    orders: Mapped[list[BrokerOrder]] = relationship(
        back_populates="trade", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (Index("ix_trade_status_underlying", "status", "underlying"),)

    # -- derived ---------------------------------------------------------
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.charges

    @property
    def is_open(self) -> bool:
        return self.status in {
            TradeStatus.OPEN.value,
            TradeStatus.PENDING.value,
            TradeStatus.CLOSING.value,
        }

    @property
    def open_legs(self) -> list[TradeLeg]:
        return [leg for leg in self.legs if leg.status == LegStatus.OPEN.value]


class TradeLeg(Base, TimestampMixin):
    __tablename__ = "trade_legs"

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_id: Mapped[int] = mapped_column(ForeignKey("trades.id", ondelete="CASCADE"), index=True)

    instrument_key: Mapped[str] = mapped_column(String(64), default="")  # feed id (Upstox)
    tradingsymbol: Mapped[str] = mapped_column(String(64), default="")  # order id (Kite)
    exchange: Mapped[str] = mapped_column(String(8), default="NFO")
    strike: Mapped[float] = mapped_column(Float)
    option_type: Mapped[str] = mapped_column(String(2))
    side: Mapped[str] = mapped_column(String(4))
    lots: Mapped[int] = mapped_column(Integer, default=1)
    lot_size: Mapped[int] = mapped_column(Integer, default=1)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(12), default=LegStatus.PENDING.value)

    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    ltp: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)

    iv: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    gamma: Mapped[float | None] = mapped_column(Float, nullable=True)
    theta: Mapped[float | None] = mapped_column(Float, nullable=True)
    vega: Mapped[float | None] = mapped_column(Float, nullable=True)

    entry_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    trade: Mapped[Trade] = relationship(back_populates="legs")
    orders: Mapped[list[BrokerOrder]] = relationship(back_populates="leg", lazy="selectin")

    @property
    def sign(self) -> int:
        return Side(self.side).sign

    def mark(self, price: float) -> float:
        """Set LTP and return this leg's unrealized P&L in rupees."""
        self.ltp = price
        if self.entry_price is None or self.status != LegStatus.OPEN.value:
            self.unrealized_pnl = 0.0
        else:
            self.unrealized_pnl = (price - self.entry_price) * self.sign * self.quantity
        return self.unrealized_pnl

    def label(self) -> str:
        return f"{self.side} {int(self.strike)}{self.option_type}"


class BrokerOrder(Base, TimestampMixin):
    """Audit trail of every order ORCA sent (paper or live)."""

    __tablename__ = "broker_orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_id: Mapped[int | None] = mapped_column(
        ForeignKey("trades.id", ondelete="CASCADE"), nullable=True, index=True
    )
    leg_id: Mapped[int | None] = mapped_column(
        ForeignKey("trade_legs.id", ondelete="CASCADE"), nullable=True, index=True
    )
    gateway: Mapped[str] = mapped_column(String(16), default="paper")
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    tradingsymbol: Mapped[str] = mapped_column(String(64), default="")
    exchange: Mapped[str] = mapped_column(String(8), default="NFO")
    transaction_type: Mapped[str] = mapped_column(String(4), default="SELL")
    product: Mapped[str] = mapped_column(String(8), default="NRML")
    order_type: Mapped[str] = mapped_column(String(8), default="MARKET")
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    average_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="PENDING")
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    purpose: Mapped[str] = mapped_column(String(12), default="ENTRY")  # ENTRY | EXIT | ADJUST

    trade: Mapped[Trade | None] = relationship(back_populates="orders")
    leg: Mapped[TradeLeg | None] = relationship(back_populates="orders")


class TradeSnapshot(Base):
    """Time series of a trade's MTM - drives the P&L chart and ML outcome labels."""

    __tablename__ = "trade_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_id: Mapped[int] = mapped_column(ForeignKey("trades.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    spot: Mapped[float | None] = mapped_column(Float, nullable=True)
    mtm: Mapped[float] = mapped_column(Float, default=0.0)
    net_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_theta: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_vega: Mapped[float | None] = mapped_column(Float, nullable=True)
    leg_marks: Mapped[dict] = mapped_column(JSON, default=dict)  # {leg_id: ltp}
