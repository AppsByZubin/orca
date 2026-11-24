"""Scored candidates produced by the scanner - kept so we can audit the model."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from orca.models.base import Base, TimestampMixin, utcnow


class Signal(Base, TimestampMixin):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    underlying: Mapped[str] = mapped_column(String(32), index=True)
    expiry: Mapped[date] = mapped_column(Date)
    strategy_id: Mapped[str] = mapped_column(String(48), index=True)
    template: Mapped[str] = mapped_column(String(48))
    spot: Mapped[float] = mapped_column(Float)

    score: Mapped[float | None] = mapped_column(Float, index=True)
    expected_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_source: Mapped[str] = mapped_column(String(48), default="heuristic")

    net_credit: Mapped[float] = mapped_column(Float, default=0.0)
    max_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    margin_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_of_profit: Mapped[float | None] = mapped_column(Float, nullable=True)

    legs: Mapped[list] = mapped_column(JSON, default=list)  # serialized LegSpec list
    breakevens: Mapped[list] = mapped_column(JSON, default=list)
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    reasons: Mapped[list] = mapped_column(JSON, default=list)

    taken: Mapped[bool] = mapped_column(Boolean, default=False)  # did we trade it?

    __table_args__ = (Index("ix_signal_underlying_score", "underlying", "score"),)
