"""Pre-trade guardrails. Everything here is a hard block, not a suggestion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from orca.config import get_settings
from orca.domain import Candidate, TradeStatus
from orca.models import Trade


def open_trade_count(session: Session) -> int:
    return len(
        session.scalars(
            select(Trade).where(
                Trade.status.in_(
                    [TradeStatus.OPEN.value, TradeStatus.PENDING.value, TradeStatus.CLOSING.value]
                )
            )
        ).all()
    )


def realized_pnl_today(session: Session) -> float:
    start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    trades = session.scalars(select(Trade).where(Trade.created_at >= start)).all()
    return sum(t.realized_pnl + t.unrealized_pnl - t.charges for t in trades)


def day_loss_breached(session: Session) -> tuple[bool, float]:
    settings = get_settings()
    pnl = realized_pnl_today(session)
    return (pnl <= -abs(settings.max_day_loss), pnl)


def check_candidate(session: Session, candidate: Candidate, lots: int) -> list[str]:
    """Return a list of violations; empty means the trade may proceed."""
    settings = get_settings()
    problems: list[str] = []

    if lots > settings.max_lots_per_trade:
        problems.append(
            f"{lots} lots exceeds ORCA_MAX_LOTS_PER_TRADE={settings.max_lots_per_trade}"
        )
    if open_trade_count(session) >= settings.max_open_trades:
        problems.append(f"already at ORCA_MAX_OPEN_TRADES={settings.max_open_trades}")

    breached, pnl = day_loss_breached(session)
    if breached:
        problems.append(f"day loss limit hit (P&L {pnl:,.0f})")

    if candidate.max_loss is None and candidate.margin_estimate is None:
        problems.append("undefined risk and no margin estimate")

    return problems


def recent_entry_exists(session: Session, underlying: str, minutes: int = 5) -> bool:
    """Crude duplicate-entry guard."""
    cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
    return bool(
        session.scalars(
            select(Trade).where(Trade.underlying == underlying, Trade.created_at >= cutoff)
        ).first()
    )
