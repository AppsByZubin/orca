"""API routers, mounted under /api by orca.web.app.

Everything requires a dashboard login (orca.web.security.require_user) except
/health (container probes, smoke tests), /session/login, and the broker OAuth
callbacks under /auth, which the broker's browser redirect hits without our
token. auth.py guards its own login-URL routes.
"""

from fastapi import APIRouter, Depends

from orca.web.api import auth, health, market, model, session, signals, tokens, trades
from orca.web.security import require_user

_protected = [Depends(require_user)]

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(session.router)
api_router.include_router(market.router, dependencies=_protected)
api_router.include_router(signals.router, dependencies=_protected)
api_router.include_router(trades.router, dependencies=_protected)
api_router.include_router(model.router, dependencies=_protected)
api_router.include_router(tokens.router, dependencies=_protected)
api_router.include_router(auth.router)

__all__ = ["api_router"]
