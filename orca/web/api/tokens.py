"""Daily broker access tokens (Zerodha Kite, Upstox), managed from the UI."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from orca.schemas.api import BrokerTokenIn, BrokerTokenOut
from orca.services import broker_tokens as svc
from orca.services.broker_tokens import Broker

router = APIRouter(prefix="/tokens", tags=["tokens"])


@router.get("", response_model=list[BrokerTokenOut])
def list_tokens() -> list[dict]:
    """Presence and expiry of each broker token. No broker calls - cheap to poll."""
    return svc.all_statuses()


@router.put("/{broker}", response_model=BrokerTokenOut)
def save_token(broker: Broker, body: BrokerTokenIn, verify: bool = True) -> dict:
    """Store today's token. With verify (default) the broker is asked first, and
    a token it rejects is not saved; if it cannot be reached the token is saved
    with verified=null."""
    try:
        return svc.save_token(broker, body.access_token, verify=verify)
    except svc.TokenRejected as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{broker}/verify", response_model=BrokerTokenOut)
def verify_token(broker: Broker) -> dict:
    """Ask the broker whether the current token still works."""
    return svc.verify_saved(broker)


@router.delete("/{broker}", response_model=BrokerTokenOut)
def delete_token(broker: Broker) -> dict:
    return svc.delete_token(broker)
