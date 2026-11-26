"""Daily broker access tokens, entered through the UI's Tokens page."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from orca.models.base import Base, TimestampMixin, utcnow


class BrokerToken(Base, TimestampMixin):
    __tablename__ = "broker_tokens"

    broker: Mapped[str] = mapped_column(String(16), primary_key=True)  # kite | upstox
    access_token: Mapped[str] = mapped_column(Text)
    saved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # The broker's daily cutoff after saved_at - the token is dead from then on.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Set when the broker answered 401/403 to this token before its cutoff
    # (revoked, or logged in elsewhere). Treated exactly like expiry.
    rejected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
