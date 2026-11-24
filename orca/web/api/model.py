"""Model inspection and (re)training."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from orca.db import db_session
from orca.ml.dataset import build_training_frame, synthetic_frame
from orca.ml.model import get_model
from orca.ml.train import train_model
from orca.schemas.api import ModelInfoOut, TrainRequest
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
