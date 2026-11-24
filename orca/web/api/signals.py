"""Candidate scanning / signal history."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from orca.db import db_session
from orca.schemas import CandidateOut, ScanRequest, candidate_to_out, signal_to_out
from orca.services.scanner import Scanner, recent_signals

router = APIRouter(prefix="/signals", tags=["signals"])


@router.post("/scan", response_model=list[CandidateOut])
def scan(request: ScanRequest, session: Session = Depends(db_session)) -> list[CandidateOut]:
    """Pull fresh chains, build candidates, score and rank them."""
    scanner = Scanner()
    candidates = scanner.scan(
        session,
        symbols=request.symbols,
        lots=request.lots,
        persist=request.persist,
        max_dte=request.max_dte,
    )
    if request.persist:
        session.commit()  # ids are stamped onto the candidates by Scanner.persist
    return [candidate_to_out(c, c.signal_id) for c in candidates[: request.limit]]


@router.get("", response_model=list[CandidateOut])
def list_signals(
    limit: int = Query(50, ge=1, le=500),
    underlying: str | None = None,
    session: Session = Depends(db_session),
) -> list[CandidateOut]:
    return [signal_to_out(s) for s in recent_signals(session, limit=limit, underlying=underlying)]
