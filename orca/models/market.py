"""Captured market data - this is the raw material the ML layer trains on."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from orca.models.base import Base, TimestampMixin


class ChainSnapshot(Base, TimestampMixin):
    """One capture of one underlying/expiry chain. Rows accumulate all day."""

    __tablename__ = "chain_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    underlying: Mapped[str] = mapped_column(String(32), index=True)
    expiry: Mapped[date] = mapped_column(Date, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    spot: Mapped[float] = mapped_column(Float)
    futures_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    atm_iv: Mapped[float | None] = mapped_column(Float, nullable=True)
    dte: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(16), default="mock")

    quotes: Mapped[list[OptionQuoteRow]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (Index("ix_chain_underlying_ts", "underlying", "ts"),)


class OptionQuoteRow(Base):
    """One option contract inside a ChainSnapshot."""

    __tablename__ = "option_quotes"

    id: Mapped[int] = mapped_column(primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("chain_snapshots.id", ondelete="CASCADE"), index=True
    )
    instrument_key: Mapped[str] = mapped_column(String(64), default="")
    tradingsymbol: Mapped[str] = mapped_column(String(64), default="")
    strike: Mapped[float] = mapped_column(Float, index=True)
    option_type: Mapped[str] = mapped_column(String(2))
    ltp: Mapped[float] = mapped_column(Float, default=0.0)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[int] = mapped_column(Integer, default=0)
    oi: Mapped[int] = mapped_column(Integer, default=0)
    oi_change: Mapped[int] = mapped_column(Integer, default=0)
    iv: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    gamma: Mapped[float | None] = mapped_column(Float, nullable=True)
    theta: Mapped[float | None] = mapped_column(Float, nullable=True)
    vega: Mapped[float | None] = mapped_column(Float, nullable=True)

    snapshot: Mapped[ChainSnapshot] = relationship(back_populates="quotes")


class UnderlyingDaily(Base, TimestampMixin):
    """Daily bar + vol stats per underlying; feeds IV-rank and realized-vol features."""

    __tablename__ = "underlying_daily"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    d: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    atm_iv_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_vol_20d: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (Index("ix_daily_symbol_d", "symbol", "d", unique=True),)
