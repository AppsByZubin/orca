"""Daily broker access tokens: storage, expiry and live verification.

Both brokers mint an access token per login, and it dies at a fixed time the
next morning (IST):

    Upstox  03:30     https://upstox.com/developer/api-documentation/authentication
    Kite    06:00     https://kite.trade/docs/connect/v3/user/#login-flow

The operator pastes each day's token into the UI's Tokens page; it is stored
in the database, so the API and the worker container (which share it) both see
it without a restart. A stored token is current until its cutoff, or until the
broker rejects it. With nothing stored, the old UPSTOX_ACCESS_TOKEN /
KITE_ACCESS_TOKEN env values are still used - their age is unknown, so they
report no expiry.
"""

from __future__ import annotations

import time as _time
from datetime import UTC, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import httpx

from orca.config import get_settings
from orca.db import session_scope
from orca.logging_setup import get_logger
from orca.models.base import utcnow
from orca.models.tokens import BrokerToken

log = get_logger(__name__)

Broker = Literal["kite", "upstox"]
BROKERS: tuple[Broker, ...] = ("kite", "upstox")
LABELS = {"kite": "Zerodha Kite", "upstox": "Upstox"}
IST = ZoneInfo("Asia/Kolkata")
DAILY_CUTOFF = {"upstox": time(3, 30), "kite": time(6, 0)}

# current_token() is read on every get_feed()/get_gateway() call, so it is
# cached briefly. A save in this process clears it; other processes (the
# worker) pick a new token up within this many seconds.
_CACHE_TTL = 30.0
_cache: dict[str, tuple[float, str]] = {}


def _aware(dt: datetime | None) -> datetime | None:
    # SQLite hands timezone-aware columns back naive (in UTC).
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=UTC)


def next_cutoff(broker: Broker, after: datetime) -> datetime:
    """The first daily cutoff strictly after `after`, in UTC."""
    local = after.astimezone(IST)
    cutoff = datetime.combine(local.date(), DAILY_CUTOFF[broker], tzinfo=IST)
    if cutoff <= local:
        cutoff += timedelta(days=1)
    return cutoff.astimezone(UTC)


def _env_token(broker: Broker) -> str:
    s = get_settings()
    return s.upstox_access_token if broker == "upstox" else s.kite_access_token


def _is_live(row: BrokerToken, now: datetime) -> bool:
    return row.rejected_at is None and _aware(row.expires_at) > now  # type: ignore[operator]


def current_token(broker: Broker) -> str:
    """The token ORCA should use right now ('' if none is usable)."""
    hit = _cache.get(broker)
    if hit and _time.monotonic() - hit[0] < _CACHE_TTL:
        return hit[1]
    token = ""
    try:
        with session_scope() as db:
            row = db.get(BrokerToken, broker)
            if row is None:
                token = _env_token(broker)
            elif _is_live(row, utcnow()):
                token = row.access_token
    except Exception as exc:  # noqa: BLE001 - never let token lookup take the feed down
        log.warning("broker token lookup failed (%s); using env value", exc)
        token = _env_token(broker)
    _cache[broker] = (_time.monotonic(), token)
    return token


def _forget_cache() -> None:
    _cache.clear()


def _in_use(broker: Broker) -> bool:
    s = get_settings()
    if broker == "upstox":
        return s.feed_provider.lower() == "upstox"
    return s.execution_gateway.lower() == "kite"


def status(broker: Broker) -> dict:
    s = get_settings()
    now = utcnow()
    with session_scope() as db:
        row = db.get(BrokerToken, broker)
        if row is not None:
            token, source = row.access_token, "saved"
            saved_at, expires_at = _aware(row.saved_at), _aware(row.expires_at)
            rejected = row.rejected_at is not None
            expired = rejected or expires_at <= now  # type: ignore[operator]
        else:
            token = _env_token(broker)
            source = "env" if token else None
            saved_at = expires_at = None
            rejected = expired = False
    has_token = bool(token)
    return {
        "broker": broker,
        "label": LABELS[broker],
        "has_token": has_token,
        "source": source,
        "token_hint": f"…{token[-4:]}" if len(token) >= 8 else None,
        "saved_at": saved_at,
        "expires_at": expires_at,
        "expired": expired,
        "rejected": rejected,
        "in_use": _in_use(broker),
        "api_key_configured": bool(s.kite_api_key if broker == "kite" else s.upstox_api_key),
        # An env token's age is unknown, so it is never trusted as current:
        # ask for one on the Tokens page, where its expiry can be tracked.
        "needs_attention": not has_token or expired or source == "env",
    }


def all_statuses() -> list[dict]:
    return [status(b) for b in BROKERS]


def probe(broker: Broker, token: str) -> tuple[bool | None, str]:
    """Ask the broker whether `token` works: (True|False, detail), or None if
    the check could not be made (no API key, network error, broker outage)."""
    try:
        if broker == "upstox":
            from orca.feeds.upstox import API_BASE

            resp = httpx.get(
                f"{API_BASE}/user/profile",
                headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
                timeout=8,
            )
        else:
            from orca.execution.kite import KiteAuth

            check = KiteAuth.verify_token(token, required_exchanges=_option_exchanges())
            return check.ok, check.detail
    except httpx.HTTPError as exc:
        return None, f"could not reach {LABELS[broker]}: {exc}"
    if resp.status_code in (401, 403):
        return False, f"{LABELS[broker]} rejected the token (HTTP {resp.status_code})"
    if resp.status_code >= 400:
        return None, f"{LABELS[broker]} answered HTTP {resp.status_code}"
    return True, f"{LABELS[broker]} accepted the token"


def _option_exchanges() -> list[str]:
    """The option segments ORCA trades (NFO for NIFTY, BFO for SENSEX, ...)."""
    from orca.config import get_universe

    try:
        return sorted({u.option_exchange for u in get_universe().enabled})
    except Exception:  # noqa: BLE001 - a config problem shouldn't block a token check
        return []


def normalize(broker: Broker, token: str) -> str:
    if broker == "kite":
        from orca.execution.kite import normalize_kite_token

        return normalize_kite_token(token, get_settings().kite_api_key)
    return token.strip()


class TokenRejected(ValueError):
    pass


def save_token(broker: Broker, token: str, verify: bool = True) -> dict:
    token = normalize(broker, token)
    if not token:
        raise ValueError("token is empty")
    verified, detail = probe(broker, token) if verify else (None, "not checked")
    if verified is False:
        raise TokenRejected(detail)
    now = utcnow()
    with session_scope() as db:
        row = db.get(BrokerToken, broker) or BrokerToken(broker=broker)
        row.access_token = token
        row.saved_at = now
        row.expires_at = next_cutoff(broker, now)
        row.rejected_at = None
        db.add(row)
    _forget_cache()
    log.info("%s token saved (expires %s)", broker, next_cutoff(broker, now).isoformat())
    return {**status(broker), "verified": verified, "verify_detail": detail}


def verify_saved(broker: Broker) -> dict:
    token = current_token(broker) or ""
    if not token:
        return {**status(broker), "verified": False, "verify_detail": "no usable token"}
    verified, detail = probe(broker, token)
    if verified is False:
        mark_rejected(broker, token)
    return {**status(broker), "verified": verified, "verify_detail": detail}


def recheck_saved(brokers: tuple[Broker, ...] = ("kite",)) -> None:
    """Scheduler job: re-ask the broker about each saved, still-current token,
    so one revoked mid-day (a new login elsewhere invalidates Kite's) is marked
    and the UI prompts before an order fails on it. Kite by default - the
    Upstox feed already notices on its next quote request."""
    for broker in brokers:
        token = current_token(broker)
        if not token or status(broker)["source"] != "saved":
            continue
        ok, detail = probe(broker, token)
        if ok is False:
            mark_rejected(broker, token)
        log.info("%s token re-check: %s", broker, detail)


def delete_token(broker: Broker) -> dict:
    with session_scope() as db:
        row = db.get(BrokerToken, broker)
        if row is not None:
            db.delete(row)
    _forget_cache()
    return status(broker)


def mark_rejected(broker: Broker, token: str) -> None:
    """The broker refused `token` (401/403): stop using it and flag it in the UI."""
    try:
        with session_scope() as db:
            row = db.get(BrokerToken, broker)
            if row is not None and row.access_token == token and row.rejected_at is None:
                row.rejected_at = utcnow()
                log.warning("%s rejected the saved token; marked expired", broker)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not mark %s token rejected: %s", broker, exc)
    _forget_cache()
