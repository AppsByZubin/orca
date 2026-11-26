"""FastAPI application factory.

    uvicorn orca.web.app:app --reload        # or: orca serve

Serves the JSON API under /api. There is no bundled front end - the dashboard
was removed in favour of a separate UI project, so every view lives in whatever
client consumes this API. Interactive docs at /docs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from orca import __version__
from orca.config import get_settings
from orca.db import init_db
from orca.jobs.scheduler import start_scheduler, stop_scheduler
from orca.logging_setup import get_logger, setup_logging
from orca.web.api import api_router
from orca.web.api import auth as broker_auth

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    log.info("ORCA %s starting (env=%s)", __version__, settings.env)
    init_db()
    start_scheduler()
    try:
        yield
    finally:
        stop_scheduler()
        log.info("ORCA stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ORCA",
        version=__version__,
        summary="ML-ranked option-selling candidates with multi-leg trade tracking",
        lifespan=lifespan,
    )
    # The UI is a separate project on its own origin, so CORS is load-bearing
    # now rather than a convenience. Access control is the bearer-token login
    # (orca.web.security); still narrow "*" to the UI's origin in deployments.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api")
    # Broker apps are registered with redirect URIs like
    # http://127.0.0.1:8000/auth/upstox/callback (no /api), so serve the
    # broker auth routes there too. Same handlers, same protection.
    app.include_router(broker_auth.router, include_in_schema=False)

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": "ORCA",
            "version": __version__,
            "api": "/api",
            "docs": "/docs",
        }

    return app


app = create_app()
