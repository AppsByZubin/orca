"""Daily broker login helpers.

Both brokers hand out a short-lived access token via a browser redirect:

    1. open the login URL,
    2. the broker redirects back with ?code= (Upstox) or ?request_token= (Kite),
    3. we exchange that for an access token.

The token is returned here for you to paste into the dashboard's Tokens page.
The callbacks are unauthenticated, so they deliberately do not store it
themselves - otherwise anyone could plant their own broker session in ORCA.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from orca.config import get_settings
from orca.execution.kite import KiteAuth
from orca.feeds.upstox import UpstoxAuth
from orca.web.security import require_user

router = APIRouter(prefix="/auth", tags=["auth"])

# The callbacks stay open: the broker redirects the browser to them, and that
# request cannot carry the dashboard's bearer token. The codes they accept are
# single-use and short-lived. Starting a login, though, needs a dashboard login.
_protected = [Depends(require_user)]


@router.get("/upstox/login", dependencies=_protected)
def upstox_login() -> dict:
    if not get_settings().upstox_api_key:
        raise HTTPException(400, "UPSTOX_API_KEY is not configured")
    return {"login_url": UpstoxAuth.login_url()}


@router.get("/upstox/callback")
def upstox_callback(code: str) -> dict:
    try:
        payload = UpstoxAuth.exchange_code(code)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"token exchange failed: {exc}") from exc
    token = payload.get("access_token", "")
    return {
        "access_token": token,
        "next_step": "paste this into the Upstox card on the dashboard's Tokens page",
        "raw": {k: v for k, v in payload.items() if k != "access_token"},
    }


@router.get("/kite/login", dependencies=_protected)
def kite_login() -> dict:
    if not get_settings().kite_api_key:
        raise HTTPException(400, "KITE_API_KEY is not configured")
    return {"login_url": KiteAuth.login_url()}


@router.get("/kite/callback")
def kite_callback(request_token: str) -> dict:
    try:
        payload = KiteAuth.exchange_request_token(request_token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"token exchange failed: {exc}") from exc
    return {
        "access_token": payload.get("access_token", ""),
        "user_id": payload.get("user_id"),
        "next_step": "paste this into the Zerodha Kite card on the dashboard's Tokens page",
    }
