"""Daily broker login helpers.

Both brokers hand out a short-lived access token via a browser redirect:

    1. open the login URL,
    2. the broker redirects back with ?code= (Upstox) or ?request_token= (Kite),
    3. we exchange that for an access token.

The token is returned here so you can paste it into .env - ORCA deliberately
does not write secrets to disk on your behalf.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from orca.config import get_settings
from orca.execution.kite import KiteAuth
from orca.feeds.upstox import UpstoxAuth

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/upstox/login")
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
        "next_step": "put this in .env as UPSTOX_ACCESS_TOKEN and restart, "
        "or export it before launching",
        "raw": {k: v for k, v in payload.items() if k != "access_token"},
    }


@router.get("/kite/login")
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
        "next_step": "put this in .env as KITE_ACCESS_TOKEN",
    }
