"""Upstox market-data provider (data feed only - orders go through Kite).

Endpoints used, checked against https://upstox.com/developer/api-documentation/
(Nov 2026). The v2 LTP and v2 historical-candle routes are deprecated there, so
those calls use v3; option chain and contracts are still v2-only.

    GET /v3/market-quote/ltp?instrument_key=k1,k2                    -> last prices
    GET /v2/option/contract?instrument_key=...                       -> expiries
    GET /v2/option/chain?instrument_key=...&expiry_date=YYYY-MM-DD   -> chain w/ greeks
    GET /v3/historical-candle/{key}/days/1/{to}/{from}               -> daily candles
    GET /v2/user/profile                                             -> token check

Instrument keys look like "NSE_INDEX|Nifty 50"; inside a URL path they must be
percent-encoded (NSE_INDEX%7CNifty%2050). Query params are encoded by httpx.

Auth: Upstox access tokens are *daily* - they expire at 03:30 IST. Paste each
day's token on the UI's Tokens page (see orca.services.broker_tokens).
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx

from orca.analytics.greeks import greeks as bs_greeks
from orca.analytics.greeks import implied_vol, year_fraction
from orca.config import Underlying, get_settings
from orca.domain import ChainSnapshot, OptionQuote, OptionType, Quote
from orca.feeds.base import FeedError, MarketDataProvider
from orca.logging_setup import get_logger

log = get_logger(__name__)

API_ROOT = "https://api.upstox.com"
API_BASE = f"{API_ROOT}/v2"  # login/token/profile routes

# /option/contract returns every contract of the underlying (thousands of rows)
# and only changes when the exchange lists a new expiry - cache the expiry list.
EXPIRY_CACHE_SEC = 600
# v2 LTP documented 500 keys per call; v3 states a limit without a number.
LTP_CHUNK = 100


def _key_path(instrument_key: str) -> str:
    """Percent-encode an instrument key for use as a URL path segment."""
    return quote(instrument_key, safe="")


def _iv_decimal(raw: Any) -> float | None:
    """Upstox's option_greeks.iv, as a decimal (0.152 == 15.2%).

    The docs don't state the unit; their samples (50.78, 262.31) and live data
    are percentages. Index IV is never below 3%, and a decimal IV is never
    above 3 (300%), so a value <= 3 is taken as already being a decimal.
    """
    try:
        iv = float(raw)
    except (TypeError, ValueError):
        return None
    if iv <= 0:
        return None
    return iv / 100.0 if iv > 3 else iv


class UpstoxAuth:
    """Helpers for the one-time-per-day login handshake."""

    @staticmethod
    def login_url() -> str:
        s = get_settings()
        return (
            f"{API_BASE}/login/authorization/dialog?response_type=code"
            f"&client_id={s.upstox_api_key}&redirect_uri={s.upstox_redirect_uri}"
        )

    @staticmethod
    def exchange_code(code: str) -> dict:
        """Swap the ?code= from the redirect for an access token."""
        s = get_settings()
        resp = httpx.post(
            f"{API_BASE}/login/authorization/token",
            headers={
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "code": code,
                "client_id": s.upstox_api_key,
                "client_secret": s.upstox_api_secret,
                "redirect_uri": s.upstox_redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()


class UpstoxProvider(MarketDataProvider):
    name = "upstox"

    def __init__(
        self,
        access_token: str | None = None,
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,  # tests inject a mock
    ) -> None:
        from orca.services.broker_tokens import current_token

        settings = get_settings()
        self.token = access_token or current_token("upstox")
        if not self.token:
            raise FeedError(
                "no current Upstox token - add today's on the UI's Tokens page "
                "(or set ORCA_FEED_PROVIDER=mock for offline development)"
            )
        self.r = settings.risk_free_rate
        self._client = httpx.Client(
            base_url=API_ROOT,
            timeout=timeout,
            transport=transport,
            headers={"Accept": "application/json", "Authorization": f"Bearer {self.token}"},
        )
        self._expiries: dict[str, tuple[float, list[date]]] = {}

    # -- low level -------------------------------------------------------
    def _get(self, path: str, params: dict | None = None) -> Any:
        try:
            resp = self._client.get(path, params=params)
        except httpx.HTTPError as exc:  # network level
            raise FeedError(f"upstox request failed: {exc}") from exc
        if resp.status_code == 401:
            from orca.services.broker_tokens import mark_rejected

            mark_rejected("upstox", self.token)
            raise FeedError("upstox token rejected (401) - the daily token has expired")
        if resp.status_code == 429:
            raise FeedError(f"upstox rate limit hit on {path}; slow the polling down")
        if resp.status_code >= 400:
            raise FeedError(f"upstox {path} -> {resp.status_code}: {resp.text[:300]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise FeedError(f"upstox {path} returned non-JSON: {resp.text[:200]}") from exc
        if isinstance(body, dict) and body.get("status") == "error":
            raise FeedError(f"upstox error: {body.get('errors') or body}")
        return body.get("data") if isinstance(body, dict) else body

    # -- spot ------------------------------------------------------------
    def _ltp(self, keys: list[str]) -> dict[str, dict]:
        """v3 LTP for `keys`, keyed by the instrument key we asked with.

        The response map is keyed "EXCHANGE:SYMBOL" (e.g. "NSE_INDEX:Nifty 50",
        "NSE_FO:NIFTY25D0226000CE"); each entry carries instrument_token
        ("NSE_FO|51834"), which is how entries are matched back.
        """
        out: dict[str, dict] = {}
        for i in range(0, len(keys), LTP_CHUNK):
            chunk = keys[i : i + LTP_CHUNK]
            data = self._get("/v3/market-quote/ltp", {"instrument_key": ",".join(chunk)}) or {}
            by_token = {p.get("instrument_token"): p for p in data.values()}
            for key in chunk:
                payload = (
                    by_token.get(key)
                    or data.get(key.replace("|", ":", 1))
                    or (next(iter(data.values())) if len(chunk) == 1 and len(data) == 1 else None)
                )
                if payload:
                    out[key] = payload
        return out

    def get_spot(self, underlying: Underlying) -> float:
        key = underlying.upstox_instrument_key
        price = (self._ltp([key]).get(key) or {}).get("last_price")
        if not price:
            raise FeedError(f"no LTP returned for {key}")
        return float(price)

    # -- expiries --------------------------------------------------------
    def get_expiries(self, underlying: Underlying) -> list[date]:
        key = underlying.upstox_instrument_key
        today = date.today()
        hit = self._expiries.get(key)
        if hit and time.monotonic() - hit[0] < EXPIRY_CACHE_SEC:
            return [e for e in hit[1] if e >= today]
        data = self._get("/v2/option/contract", {"instrument_key": key})
        expiries = set()
        for row in data or []:
            raw = row.get("expiry")
            if raw:
                expiries.add(datetime.strptime(raw[:10], "%Y-%m-%d").date())
        result = sorted(e for e in expiries if e >= today)
        self._expiries[key] = (time.monotonic(), result)
        return result

    # -- chain -----------------------------------------------------------
    def get_option_chain(self, underlying: Underlying, expiry: date) -> ChainSnapshot:
        data = self._get(
            "/v2/option/chain",
            {
                "instrument_key": underlying.upstox_instrument_key,
                "expiry_date": expiry.isoformat(),
            },
        )
        if not data:
            raise FeedError(f"empty chain for {underlying.symbol} {expiry}")

        ts = datetime.now(UTC)
        spot = float(data[0].get("underlying_spot_price") or 0) or self.get_spot(underlying)
        t = year_fraction(expiry)
        options: list[OptionQuote] = []

        for row in data:
            strike = float(row.get("strike_price") or 0)
            for field, option_type in (
                ("call_options", OptionType.CE),
                ("put_options", OptionType.PE),
            ):
                node = row.get(field) or {}
                options.append(
                    self._parse_leg(underlying, expiry, strike, option_type, node, spot, t)
                )

        return ChainSnapshot(
            underlying=underlying.symbol,
            expiry=expiry,
            spot=spot,
            ts=ts,
            options=[o for o in options if o.ltp > 0 or o.oi > 0],
            source=self.name,
        )

    def _parse_leg(
        self,
        underlying: Underlying,
        expiry: date,
        strike: float,
        option_type: OptionType,
        node: dict,
        spot: float,
        t: float,
    ) -> OptionQuote:
        md = node.get("market_data") or {}
        gk = node.get("option_greeks") or {}
        ltp = float(md.get("ltp") or 0.0)
        iv = _iv_decimal(gk.get("iv"))
        # Outside market hours Upstox sends zero greeks; back IV out of the price.
        if iv is None and ltp > 0:
            iv = implied_vol(ltp, spot, strike, t, option_type, self.r)

        delta, gamma, theta, vega = (
            gk.get("delta"),
            gk.get("gamma"),
            gk.get("theta"),
            gk.get("vega"),
        )
        if not delta and iv:  # missing, or the 0.0 Upstox sends when greeks are off
            g = bs_greeks(spot, strike, t, iv, option_type, self.r)
            delta, gamma, theta, vega = g["delta"], g["gamma"], g["theta"], g["vega"]

        from orca.symbols import kite_tradingsymbol

        return OptionQuote(
            underlying=underlying.symbol,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            instrument_key=node.get("instrument_key") or "",
            tradingsymbol=kite_tradingsymbol(underlying, expiry, strike, option_type),
            ltp=ltp,
            bid=float(md.get("bid_price") or 0) or None,
            ask=float(md.get("ask_price") or 0) or None,
            volume=int(md.get("volume") or 0),
            oi=int(md.get("oi") or 0),
            oi_change=int((md.get("oi") or 0) - (md.get("prev_oi") or 0)),
            iv=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
        )

    # -- quotes ----------------------------------------------------------
    def get_quotes(self, instrument_keys: list[str]) -> dict[str, Quote]:
        now = datetime.now(UTC)
        return {
            key: Quote(instrument_key=key, ltp=float(payload.get("last_price") or 0), ts=now)
            for key, payload in self._ltp(instrument_keys).items()
        }

    def get_daily_closes(self, underlying: Underlying, days: int = 60) -> list[tuple[date, float]]:
        to_d = date.today()
        from_d = to_d - timedelta(days=int(days * 1.6) + 5)
        key = _key_path(underlying.upstox_instrument_key)
        data = self._get(f"/v3/historical-candle/{key}/days/1/{to_d}/{from_d}") or {}
        candles = data.get("candles") or []
        out: list[tuple[date, float]] = []
        for c in candles:
            try:
                out.append((datetime.fromisoformat(c[0]).date(), float(c[4])))
            except (ValueError, IndexError, TypeError):
                continue
        return sorted(out)[-days:]

    def health(self) -> dict:
        try:
            self._get("/v2/user/profile")
            return {"provider": self.name, "ok": True}
        except FeedError as exc:
            return {"provider": self.name, "ok": False, "error": str(exc)}

    def close(self) -> None:
        self._client.close()
