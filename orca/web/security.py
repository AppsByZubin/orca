"""Dashboard login: one configured operator account, stateless bearer tokens.

A token is `<base64url(json payload)>.<base64url(hmac-sha256)>` where the
payload carries the username and an expiry. Nothing is stored server-side, so
tokens survive restarts and work across replicas that share the same secret.
Standard library only - no JWT dependency for a single-user login.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from orca.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _secret(settings: Settings) -> bytes:
    if settings.auth_secret:
        return settings.auth_secret.encode()
    # No explicit secret: derive one from the credentials, so rotating the
    # password invalidates every token issued under the old one.
    seed = f"orca-auth:{settings.auth_username}:{settings.auth_password}"
    return hashlib.sha256(seed.encode()).digest()


def _sign(body: str, settings: Settings) -> str:
    return _b64(hmac.new(_secret(settings), body.encode(), hashlib.sha256).digest())


def check_credentials(username: str, password: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    # Compare both fields in constant time and don't short-circuit, so timing
    # doesn't reveal which one was wrong.
    user_ok = hmac.compare_digest(username.encode(), settings.auth_username.encode())
    pass_ok = hmac.compare_digest(password.encode(), settings.auth_password.encode())
    return user_ok and pass_ok


def issue_token(username: str, settings: Settings | None = None) -> tuple[str, int]:
    """Return (token, expires_at_epoch_seconds)."""
    settings = settings or get_settings()
    exp = int(time.time() + settings.auth_token_ttl_hours * 3600)
    body = _b64(json.dumps({"sub": username, "exp": exp}, separators=(",", ":")).encode())
    return f"{body}.{_sign(body, settings)}", exp


def verify_token(token: str, settings: Settings | None = None) -> str | None:
    """Return the username for a valid, unexpired token, else None."""
    settings = settings or get_settings()
    try:
        body, sig = token.split(".", 1)
        if not hmac.compare_digest(sig, _sign(body, settings)):
            return None
        payload = json.loads(_unb64(body))
        if int(payload["exp"]) < time.time():
            return None
        sub = str(payload["sub"])
    except (ValueError, KeyError, TypeError):
        return None
    # A token for a since-renamed account is no longer valid.
    return sub if hmac.compare_digest(sub.encode(), settings.auth_username.encode()) else None


def require_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """FastAPI dependency: the logged-in username, or 401."""
    user = verify_token(creds.credentials) if creds else None
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
