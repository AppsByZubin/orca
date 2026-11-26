"""Dashboard login: exchange the operator credentials for a bearer token."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from orca.schemas.api import LoginRequest, SessionOut
from orca.web.security import check_credentials, issue_token, require_user

router = APIRouter(prefix="/session", tags=["session"])


@router.post("/login", response_model=SessionOut)
def login(body: LoginRequest) -> SessionOut:
    if not check_credentials(body.username, body.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid username or password")
    token, exp = issue_token(body.username)
    return SessionOut(access_token=token, username=body.username, expires_at=exp)


@router.get("/me")
def me(user: str = Depends(require_user)) -> dict:
    return {"username": user}
