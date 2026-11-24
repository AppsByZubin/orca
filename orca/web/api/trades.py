"""Trade book: create, enter, track, adjust, exit."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from orca.analytics.greeks import year_fraction
from orca.analytics.payoff import payoff_profile
from orca.db import db_session
from orca.domain import LegStatus, OptionType, Side, TradeStatus
from orca.models import Signal, TradeSnapshot
from orca.schemas import (
    ManualTradeRequest,
    PayoffOut,
    TradeFromSignalRequest,
    TradeOut,
    TradeUpdateRequest,
    trade_to_out,
)
from orca.services.risk import check_candidate
from orca.services.tracker import Tracker, portfolio_summary
from orca.services.trade_service import (
    TradeService,
    candidate_from_signal,
    get_trade,
    list_trades,
)

router = APIRouter(tags=["trades"])
IST = ZoneInfo("Asia/Kolkata")


@router.get("/trades", response_model=list[TradeOut])
def get_trades(
    status: str | None = Query(None, description="open | CLOSED | DRAFT ..."),
    limit: int = Query(100, ge=1, le=500),
    session: Session = Depends(db_session),
) -> list[TradeOut]:
    return [trade_to_out(t) for t in list_trades(session, status=status, limit=limit)]


@router.get("/trades/{trade_id}", response_model=TradeOut)
def get_one(trade_id: int, session: Session = Depends(db_session)) -> TradeOut:
    trade = get_trade(session, trade_id)
    if trade is None:
        raise HTTPException(404, f"trade {trade_id} not found")
    return trade_to_out(trade)


@router.post("/trades/from-signal", response_model=TradeOut)
def from_signal(
    request: TradeFromSignalRequest, session: Session = Depends(db_session)
) -> TradeOut:
    signal = session.scalar(select(Signal).where(Signal.id == request.signal_id))
    if signal is None:
        raise HTTPException(404, f"signal {request.signal_id} not found")

    candidate = candidate_from_signal(signal)
    violations = check_candidate(session, candidate, request.lots)
    if violations:
        raise HTTPException(409, {"error": "risk checks failed", "violations": violations})

    service = TradeService()
    trade = service.create_from_candidate(
        session,
        candidate,
        lots=request.lots,
        auto_exit=request.auto_exit,
        signal_id=signal.id,
        notes=request.notes,
    )
    signal.taken = True
    if request.place:
        service.enter(session, trade)
    session.commit()
    return trade_to_out(trade)


@router.post("/trades/manual", response_model=TradeOut)
def manual(request: ManualTradeRequest, session: Session = Depends(db_session)) -> TradeOut:
    service = TradeService()
    try:
        trade = service.create_manual(
            session,
            underlying_symbol=request.underlying,
            expiry=request.expiry,
            legs=[leg.model_dump() for leg in request.legs],
            lots=request.lots,
            strategy_id=request.strategy_id,
            auto_exit=request.auto_exit,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if request.place:
        service.enter(session, trade)
    session.commit()
    return trade_to_out(trade)


@router.post("/trades/{trade_id}/enter", response_model=TradeOut)
def enter(trade_id: int, session: Session = Depends(db_session)) -> TradeOut:
    trade = get_trade(session, trade_id)
    if trade is None:
        raise HTTPException(404, f"trade {trade_id} not found")
    if trade.status != TradeStatus.DRAFT.value:
        raise HTTPException(409, f"trade {trade_id} is {trade.status}, not DRAFT")
    TradeService().enter(session, trade)
    session.commit()
    return trade_to_out(trade)


@router.patch("/trades/{trade_id}", response_model=TradeOut)
def update(
    trade_id: int, request: TradeUpdateRequest, session: Session = Depends(db_session)
) -> TradeOut:
    trade = get_trade(session, trade_id)
    if trade is None:
        raise HTTPException(404, f"trade {trade_id} not found")
    for field, value in request.model_dump(exclude_unset=True).items():
        setattr(trade, field, value)
    session.commit()
    return trade_to_out(trade)


@router.post("/trades/{trade_id}/exit", response_model=TradeOut)
def exit_trade(
    trade_id: int, reason: str = "manual", session: Session = Depends(db_session)
) -> TradeOut:
    trade = get_trade(session, trade_id)
    if trade is None:
        raise HTTPException(404, f"trade {trade_id} not found")
    TradeService().exit_trade(session, trade, reason=reason)
    session.commit()
    return trade_to_out(trade)


@router.post("/trades/{trade_id}/legs/{leg_id}/exit", response_model=TradeOut)
def exit_leg(
    trade_id: int, leg_id: int, reason: str = "manual-leg", session: Session = Depends(db_session)
) -> TradeOut:
    """Close one leg only - how a strangle gets converted into a naked short."""
    trade = get_trade(session, trade_id)
    if trade is None:
        raise HTTPException(404, f"trade {trade_id} not found")
    leg = next((lg for lg in trade.legs if lg.id == leg_id), None)
    if leg is None:
        raise HTTPException(404, f"leg {leg_id} not on trade {trade_id}")
    TradeService().exit_leg(session, leg, reason=reason)
    trade.realized_pnl = round(sum(lg.realized_pnl for lg in trade.legs), 2)
    if not trade.open_legs:
        trade.status = TradeStatus.CLOSED.value
        trade.exit_ts = datetime.now(IST)
    session.commit()
    return trade_to_out(trade)


@router.post("/trades/{trade_id}/track", response_model=TradeOut)
def track_now(trade_id: int, session: Session = Depends(db_session)) -> TradeOut:
    trade = get_trade(session, trade_id)
    if trade is None:
        raise HTTPException(404, f"trade {trade_id} not found")
    Tracker().track(session, trade)
    session.commit()
    return trade_to_out(trade)


@router.get("/trades/{trade_id}/payoff", response_model=PayoffOut)
def payoff(trade_id: int, session: Session = Depends(db_session)) -> PayoffOut:
    trade = get_trade(session, trade_id)
    if trade is None:
        raise HTTPException(404, f"trade {trade_id} not found")

    legs = [
        (
            leg.strike,
            OptionType(leg.option_type),
            Side(leg.side),
            leg.quantity,
            leg.entry_price or 0.0,
        )
        for leg in trade.legs
        if leg.status in {LegStatus.OPEN.value, LegStatus.CLOSED.value}
    ]
    spot = trade.entry_spot or (trade.legs[0].strike if trade.legs else 0.0)
    snapshot = session.scalars(
        select(TradeSnapshot)
        .where(TradeSnapshot.trade_id == trade_id)
        .order_by(TradeSnapshot.ts.desc())
        .limit(1)
    ).first()
    if snapshot and snapshot.spot:
        spot = snapshot.spot

    profile = payoff_profile(
        legs, spot, iv=0.13, t_years=year_fraction(trade.expiry, datetime.now(IST))
    )
    return PayoffOut(
        spot=spot,
        net_credit=round(profile.net_credit, 2),
        max_profit=profile.max_profit,
        max_loss=profile.max_loss,
        breakevens=profile.breakevens,
        prob_of_profit=profile.prob_of_profit,
        curve=[[round(x, 2), round(y, 2)] for x, y in profile.curve],
        current_mtm=round(trade.total_pnl, 2),
    )


@router.get("/trades/{trade_id}/history")
def history(trade_id: int, session: Session = Depends(db_session)) -> list[dict]:
    """MTM time series for the trade's P&L chart."""
    rows = session.scalars(
        select(TradeSnapshot).where(TradeSnapshot.trade_id == trade_id).order_by(TradeSnapshot.ts)
    ).all()
    return [
        {
            "ts": r.ts.isoformat(),
            "mtm": r.mtm,
            "spot": r.spot,
            "net_delta": r.net_delta,
            "net_theta": r.net_theta,
        }
        for r in rows
    ]


@router.get("/portfolio")
def portfolio(session: Session = Depends(db_session)) -> dict:
    return portfolio_summary(session)


@router.post("/portfolio/track")
def track_all(session: Session = Depends(db_session)) -> dict:
    results = Tracker().track_all(session)
    session.commit()
    return {
        "tracked": len(results),
        "actions": [
            {"trade_id": r.trade_id, "action": r.action, "mtm": r.mtm, "alerts": r.alerts}
            for r in results
            if r.action != "hold"
        ],
        "summary": portfolio_summary(session),
    }
