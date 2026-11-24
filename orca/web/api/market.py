"""Market data endpoints."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from orca.config import get_universe
from orca.db import db_session
from orca.feeds import FeedError
from orca.schemas import ChainOut, UnderlyingOut, chain_to_out
from orca.services.market_data import MarketDataService

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/underlyings", response_model=list[UnderlyingOut])
def underlyings() -> list[UnderlyingOut]:
    return [
        UnderlyingOut(
            symbol=u.symbol,
            name=u.name,
            lot_size=u.lot_size,
            strike_step=u.strike_step,
            option_exchange=u.option_exchange,
            enabled=u.enabled,
        )
        for u in get_universe().underlyings
    ]


@router.get("/expiries", response_model=list[date])
def expiries(symbol: str = Query(..., description="e.g. NIFTY")) -> list[date]:
    try:
        underlying = get_universe().get(symbol)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return MarketDataService().expiries(underlying)


@router.get("/chain", response_model=ChainOut)
def chain(
    symbol: str = Query(...),
    expiry: date | None = None,
    strikes: int = Query(12, ge=1, le=60, description="strikes to keep either side of ATM"),
    persist: bool = False,
    session: Session = Depends(db_session),
) -> ChainOut:
    try:
        underlying = get_universe().get(symbol)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    service = MarketDataService()
    expiry = expiry or service.nearest_expiry(underlying, max_dte=60)
    if expiry is None:
        raise HTTPException(404, f"no expiry available for {symbol}")
    try:
        snapshot = service.chain(underlying, expiry)
    except FeedError as exc:
        raise HTTPException(502, f"feed error: {exc}") from exc

    if persist:
        service.persist_chain(session, snapshot)
        session.commit()

    # Trim to a window around ATM so the UI stays light.
    atm = snapshot.atm_strike()
    window = strikes * underlying.strike_step
    snapshot.options = [o for o in snapshot.options if abs(o.strike - atm) <= window]
    return chain_to_out(snapshot)


@router.get("/context")
def context(
    symbol: str = Query(...),
    expiry: date | None = None,
    session: Session = Depends(db_session),
) -> dict:
    """Regime features (IV rank, realized vol, trend...) for the nearest expiry."""
    try:
        underlying = get_universe().get(symbol)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    service = MarketDataService()
    expiry = expiry or service.nearest_expiry(underlying, max_dte=60)
    if expiry is None:
        raise HTTPException(404, f"no expiry available for {symbol}")
    snapshot = service.chain(underlying, expiry)
    return {
        "underlying": symbol,
        "expiry": expiry.isoformat(),
        "spot": snapshot.spot,
        "context": service.context(session, underlying, snapshot),
    }
