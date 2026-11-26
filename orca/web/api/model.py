"""Model inspection and (re)training, plus the market-implied probability model."""

from __future__ import annotations

import math
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from orca.analytics.greeks import year_fraction
from orca.analytics.probability import density_curve, p_below, p_touch, sigma_ranges
from orca.config import get_universe
from orca.db import db_session
from orca.domain import OptionType
from orca.feeds import FeedError
from orca.ml.dataset import build_training_frame, synthetic_frame
from orca.ml.model import get_model
from orca.ml.train import train_model
from orca.schemas.api import (
    CurvePointOut,
    ModelInfoOut,
    ProbabilityOut,
    SigmaRangeOut,
    StrikeProbabilityOut,
    TrainRequest,
)
from orca.services.market_data import MarketDataService
from orca.web.api.health import model_info

router = APIRouter(prefix="/model", tags=["model"])


@router.get("", response_model=ModelInfoOut)
def info() -> ModelInfoOut:
    return model_info()


@router.post("/train")
def train(request: TrainRequest, session: Session = Depends(db_session)) -> dict:
    """Train on captured snapshots, or on synthetic data to smoke-test the path."""
    if request.synthetic:
        frame = synthetic_frame(request.rows)
    else:
        frame = build_training_frame(
            session,
            underlying=request.underlying,
            target_pct=request.target_pct,
            stop_multiple=request.stop_multiple,
        )
    if frame.empty:
        raise HTTPException(
            409,
            "no training data yet - let the snapshot job run for a few sessions, "
            "or POST again with synthetic=true to validate the pipeline",
        )
    _model, metrics = train_model(
        frame,
        name=request.name,
        target_pct=request.target_pct,
        stop_multiple=request.stop_multiple,
        synthetic=request.synthetic,
    )
    get_model(reload=True)
    return {"name": request.name, "synthetic": request.synthetic, "metrics": metrics}


@router.get("/features")
def features() -> dict:
    model = get_model()
    return {
        "is_fitted": model.is_fitted,
        "importance": model.feature_importance(),
        "metadata": model.metadata.__dict__,
    }


@router.get("/probability", response_model=ProbabilityOut)
def probability(
    symbol: str = Query(..., description="e.g. NIFTY"),
    expiry: date | None = None,
    strikes: int = Query(12, ge=1, le=60, description="strikes to keep either side of ATM"),
) -> ProbabilityOut:
    """Where the market's implied vol says the underlying expires.

    Zero-drift lognormal - the model behind every candidate's prob_of_profit.
    The distribution and sigma ranges use ATM IV; per-strike probabilities use
    that strike's out-of-the-money option IV, so the smile is respected.
    """
    try:
        underlying = get_universe().get(symbol)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    service = MarketDataService()
    expiry = expiry or service.nearest_expiry(underlying, max_dte=60)
    if expiry is None:
        raise HTTPException(404, f"no expiry available for {symbol}")
    try:
        chain = service.chain(underlying, expiry)
    except FeedError as exc:
        raise HTTPException(502, f"feed error: {exc}") from exc

    atm_iv = chain.atm_iv()
    if not atm_iv or atm_iv <= 0:
        raise HTTPException(422, f"no implied volatility in the {symbol} {expiry} chain")
    spot, atm = chain.spot, chain.atm_strike()
    t = year_fraction(expiry)
    root_t = math.sqrt(t)
    sigma = atm_iv * root_t

    window = strikes * underlying.strike_step
    rows: list[StrikeProbabilityOut] = []
    for k in chain.strikes():
        if abs(k - atm) > window:
            continue
        ce, pe = chain.get(k, OptionType.CE), chain.get(k, OptionType.PE)
        otm, itm = (pe, ce) if k < spot else (ce, pe)
        if otm and otm.iv:
            iv, src = otm.iv, "otm"
        elif itm and itm.iv:
            iv, src = itm.iv, "itm"
        else:
            iv, src = atm_iv, "atm"
        below = p_below(k, spot, iv * root_t)
        rows.append(
            StrikeProbabilityOut(
                strike=k,
                iv=iv,
                iv_source=src,
                p_below=below,
                p_above=1.0 - below,
                p_touch=p_touch(k, spot, iv * root_t),
                call_delta=ce.delta if ce else None,
                put_delta=pe.delta if pe else None,
            )
        )

    return ProbabilityOut(
        underlying=chain.underlying,
        expiry=expiry,
        ts=chain.ts,
        source=chain.source,
        spot=spot,
        atm_strike=atm,
        atm_iv=atm_iv,
        dte=chain.dte(),
        t_years=t,
        sigma=sigma,
        ranges=[SigmaRangeOut(**r.__dict__) for r in sigma_ranges(spot, sigma)],
        strikes=rows,
        curve=[
            CurvePointOut(price=x, density=d, p_below=b) for x, d, b in density_curve(spot, sigma)
        ],
    )
