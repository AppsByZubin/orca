"""API routers, mounted under /api by orca.web.app."""

from fastapi import APIRouter

from orca.web.api import auth, health, market, model, signals, trades

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(market.router)
api_router.include_router(signals.router)
api_router.include_router(trades.router)
api_router.include_router(model.router)
api_router.include_router(auth.router)

__all__ = ["api_router"]
