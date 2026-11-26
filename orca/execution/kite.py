"""Zerodha Kite Connect order gateway.

Uses the REST API directly (no SDK dependency required; install the optional
`kiteconnect` extra if you prefer it). Endpoints:

    POST   /orders/regular        place
    DELETE /orders/regular/{id}   cancel
    GET    /orders                order book
    GET    /positions             positions
    POST   /margins/basket        margin for a multi-leg basket

Auth header:  Authorization: token <api_key>:<access_token>   (+ X-Kite-Version: 3)
The access token is daily; mint it from the request_token in the login redirect.

SAFETY: every call goes through `_assert_enabled()`, so nothing reaches the
exchange unless ORCA_TRADING_ENABLED=true *and* ORCA_EXECUTION_GATEWAY=kite.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import httpx

from orca.config import get_settings
from orca.domain import OrderStatus
from orca.execution.base import (
    ExecutionError,
    OrderGateway,
    OrderRequest,
    OrderResult,
    TradingDisabled,
)
from orca.logging_setup import get_logger

log = get_logger(__name__)

KITE_BASE = "https://api.kite.trade"
KITE_LOGIN = "https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"

STATUS_MAP = {
    "COMPLETE": OrderStatus.COMPLETE,
    "REJECTED": OrderStatus.REJECTED,
    "CANCELLED": OrderStatus.CANCELLED,
    "OPEN": OrderStatus.PENDING,
    "TRIGGER PENDING": OrderStatus.PENDING,
    "VALIDATION PENDING": OrderStatus.PENDING,
    "PUT ORDER REQ RECEIVED": OrderStatus.PENDING,
}


@dataclass
class KiteTokenCheck:
    """Outcome of asking Kite whether an access token works.

    ok is True (Kite accepted it), False (Kite rejected it - expired, revoked,
    or minted for a different api_key) or None (could not tell: no api_key,
    network trouble, rate limit, Kite outage).
    """

    ok: bool | None
    detail: str
    user_id: str | None = None
    user_name: str | None = None
    exchanges: list[str] = field(default_factory=list)
    missing_exchanges: list[str] = field(default_factory=list)


def normalize_kite_token(raw: str, api_key: str = "") -> str:
    """Undo the usual paste mistakes: the whole `token <api_key>:<access_token>`
    header, or `<api_key>:<access_token>`, instead of just the access token."""
    token = raw.strip().strip("\"'")
    if token.lower().startswith("token "):
        token = token[6:].strip()
    if ":" in token:
        prefix, _, rest = token.partition(":")
        if not api_key or prefix == api_key:
            token = rest.strip()
    return token


class KiteAuth:
    @staticmethod
    def verify_token(
        access_token: str,
        api_key: str | None = None,
        required_exchanges: list[str] | None = None,
        timeout: float = 8.0,
    ) -> KiteTokenCheck:
        """GET /user/profile with the token - the cheapest authenticated call.

        A pass also reports whose account the token belongs to and whether that
        account can trade `required_exchanges` (the option segments ORCA uses);
        a token for an account without NFO/BFO is accepted but flagged.
        """
        api_key = api_key if api_key is not None else get_settings().kite_api_key
        if not api_key:
            return KiteTokenCheck(
                None, "KITE_API_KEY is not configured, so the token cannot be checked"
            )
        token = normalize_kite_token(access_token, api_key)
        if not token or any(c.isspace() for c in token):
            return KiteTokenCheck(False, "that does not look like a Kite access token")
        try:
            resp = httpx.get(
                f"{KITE_BASE}/user/profile",
                headers={"X-Kite-Version": "3", "Authorization": f"token {api_key}:{token}"},
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            return KiteTokenCheck(None, f"could not reach Zerodha Kite: {exc}")
        try:
            body = resp.json()
        except ValueError:
            body = {}
        message = body.get("message") or f"HTTP {resp.status_code}"
        error_type = body.get("error_type")

        if resp.status_code == 403 or error_type == "TokenException":
            # Kite uses the same TokenException for an expired token, a revoked
            # one, and a token minted under a different api_key.
            return KiteTokenCheck(
                False,
                f"Zerodha Kite rejected the token ({message}). It has expired or been revoked, "
                "or it was generated for a different app than KITE_API_KEY.",
            )
        if resp.status_code == 429:
            return KiteTokenCheck(None, "Zerodha Kite rate-limited the check; try again shortly")
        if resp.status_code >= 400 or body.get("status") == "error":
            return KiteTokenCheck(None, f"Zerodha Kite could not check the token: {message}")

        data = body.get("data") or {}
        user_id, user_name = data.get("user_id"), data.get("user_name")
        exchanges = [str(e) for e in data.get("exchanges") or []]
        missing = [e for e in required_exchanges or [] if exchanges and e not in exchanges]
        who = f"{user_id} ({user_name})" if user_name else str(user_id or "unknown account")
        detail = f"Zerodha Kite accepted the token for {who}"
        if missing:
            detail += f", but the account is not enabled for {', '.join(missing)}"
        return KiteTokenCheck(True, detail, user_id, user_name, exchanges, missing)

    @staticmethod
    def login_url() -> str:
        return KITE_LOGIN.format(api_key=get_settings().kite_api_key)

    @staticmethod
    def exchange_request_token(request_token: str) -> dict:
        """POST /session/token with checksum = sha256(api_key + request_token + secret)."""
        s = get_settings()
        checksum = hashlib.sha256(
            f"{s.kite_api_key}{request_token}{s.kite_api_secret}".encode()
        ).hexdigest()
        resp = httpx.post(
            f"{KITE_BASE}/session/token",
            headers={"X-Kite-Version": "3"},
            data={
                "api_key": s.kite_api_key,
                "request_token": request_token,
                "checksum": checksum,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json().get("data", {})


class KiteGateway(OrderGateway):
    name = "kite"
    live = True

    def __init__(self, api_key: str | None = None, access_token: str | None = None) -> None:
        from orca.services.broker_tokens import current_token

        s = get_settings()
        self.api_key = api_key or s.kite_api_key
        self.access_token = access_token or current_token("kite")
        if not self.api_key or not self.access_token:
            raise ExecutionError(
                "KITE_API_KEY or a current Kite token is missing - add today's token on the "
                "UI's Tokens page, or use ORCA_EXECUTION_GATEWAY=paper."
            )
        self._client = httpx.Client(
            base_url=KITE_BASE,
            timeout=20.0,
            headers={
                "X-Kite-Version": "3",
                "Authorization": f"token {self.api_key}:{self.access_token}",
            },
        )

    # -- guardrail -------------------------------------------------------
    @staticmethod
    def _assert_enabled() -> None:
        if not get_settings().trading_enabled:
            raise TradingDisabled(
                "ORCA_TRADING_ENABLED is false - refusing to send a live order. "
                "Flip it in .env once you have verified the basket by hand."
            )

    def _request(self, method: str, path: str, data: dict | None = None) -> dict:
        resp = self._client.request(method, path, data=data)
        body: dict
        try:
            body = resp.json()
        except ValueError:
            raise ExecutionError(f"kite {path} -> {resp.status_code}: {resp.text[:300]}") from None
        if resp.status_code == 403 and body.get("error_type") == "TokenException":
            from orca.services.broker_tokens import mark_rejected

            mark_rejected("kite", self.access_token)
        if resp.status_code >= 400 or body.get("status") == "error":
            raise ExecutionError(f"kite {path}: {body.get('message') or resp.text[:200]}")
        return body.get("data", {})

    # -- orders ----------------------------------------------------------
    def place_order(self, request: OrderRequest) -> OrderResult:
        self._assert_enabled()
        payload = {
            "tradingsymbol": request.tradingsymbol,
            "exchange": request.exchange,
            "transaction_type": request.transaction_type,
            "order_type": request.order_type,
            "quantity": request.quantity,
            "product": request.product,
            "validity": request.validity,
            "tag": request.tag[:20],
        }
        if request.price is not None and request.order_type in {"LIMIT", "SL"}:
            payload["price"] = request.price
        if request.trigger_price is not None:
            payload["trigger_price"] = request.trigger_price

        try:
            data = self._request("POST", "/orders/regular", payload)
        except ExecutionError as exc:
            log.error("kite order rejected: %s", exc)
            return OrderResult(ok=False, status=OrderStatus.REJECTED, message=str(exc))

        order_id = data.get("order_id")
        log.info("kite order %s placed for %s", order_id, request.tradingsymbol)
        return OrderResult(ok=True, order_id=order_id, status=OrderStatus.PENDING, raw=data)

    def get_order(self, order_id: str) -> OrderResult:
        orders = self._request("GET", "/orders") or []
        rows = [o for o in orders if str(o.get("order_id")) == str(order_id)]
        if not rows:
            return OrderResult(ok=False, message=f"order {order_id} not in order book")
        row = rows[-1]
        return OrderResult(
            ok=True,
            order_id=order_id,
            status=STATUS_MAP.get(row.get("status", ""), OrderStatus.PENDING),
            filled_quantity=int(row.get("filled_quantity") or 0),
            average_price=float(row.get("average_price") or 0) or None,
            message=row.get("status_message") or "",
            raw=row,
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        self._assert_enabled()
        data = self._request("DELETE", f"/orders/regular/{order_id}")
        return OrderResult(ok=True, order_id=order_id, status=OrderStatus.CANCELLED, raw=data)

    def get_positions(self) -> list[dict]:
        data = self._request("GET", "/positions") or {}
        return data.get("net", [])

    def basket_margin(self, requests: list[OrderRequest]) -> float | None:
        """Real SPAN+exposure for the basket, including hedge benefit."""
        orders = [
            {
                "exchange": r.exchange,
                "tradingsymbol": r.tradingsymbol,
                "transaction_type": r.transaction_type,
                "variety": "regular",
                "product": r.product,
                "order_type": r.order_type,
                "quantity": r.quantity,
                "price": r.price or 0,
                "trigger_price": r.trigger_price or 0,
            }
            for r in requests
        ]
        try:
            # Kite expects a JSON array body for this one.
            resp = self._client.post(
                "/margins/basket", params={"consider_positions": "true"}, json=orders
            )
            body = resp.json()
            if body.get("status") == "error":
                raise ExecutionError(body.get("message", "basket margin failed"))
            initial = (body.get("data") or {}).get("initial") or {}
            return float(initial.get("total") or 0) or None
        except (httpx.HTTPError, ValueError, ExecutionError) as exc:
            log.warning("basket margin unavailable: %s", exc)
            return None

    def health(self) -> dict:
        try:
            profile = self._request("GET", "/user/profile")
            return {
                "gateway": self.name,
                "live": True,
                "ok": True,
                "user": profile.get("user_id"),
                "trading_enabled": get_settings().trading_enabled,
            }
        except ExecutionError as exc:
            return {"gateway": self.name, "live": True, "ok": False, "error": str(exc)}
