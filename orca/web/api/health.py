"""Health / system status."""

from __future__ import annotations

from fastapi import APIRouter

from orca import __version__
from orca.config import get_settings, get_universe
from orca.execution import get_gateway
from orca.feeds import get_feed
from orca.jobs.scheduler import is_market_open
from orca.ml.model import get_model
from orca.schemas.api import HealthOut, ModelInfoOut

router = APIRouter(tags=["system"])


def model_info() -> ModelInfoOut:
    model = get_model()
    meta = model.metadata
    return ModelInfoOut(
        name=meta.name,
        is_fitted=model.is_fitted,
        trained_at=meta.trained_at,
        n_samples=meta.n_samples,
        synthetic=meta.synthetic,
        metrics=meta.metrics,
        label_definition=meta.label_definition,
        top_features=dict(list(model.feature_importance().items())[:10]),
    )


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    settings = get_settings()
    return HealthOut(
        ok=True,
        env=settings.env,
        version=__version__,
        market_open=is_market_open(),
        feed=get_feed().health(),
        gateway=get_gateway().health(),
        trading_enabled=settings.trading_enabled,
        model=model_info(),
        universe=[u.symbol for u in get_universe().enabled],
    )
