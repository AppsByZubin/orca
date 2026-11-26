"""Upstox market-data provider (data feed only - orders go through Kite).

Endpoints used (Upstox API v2). Verify against the current docs at
https://upstox.com/developer/api-documentation/ before going live; Upstox has
been moving some market-data routes to v3.

    GET /v2/option/contract?instrument_key=...                 -> expiries + contracts
    GET /v2/option/chain?instrument_key=...&expiry_date=...     -> chain w/ greeks
    GET /v2/market-quote/ltp?instrument_key=k1,k2               -> last prices
    GET /v2/historical-candle/{key}/day/{to}/{from}             -> daily candles

Auth: Upstox access tokens are *daily* - they expire around 03:30 IST. Use
`UpstoxAuth` to mint one, then put it in .env as UPSTOX_ACCESS_TOKEN (or have
the web app refresh it via /auth/upstox/*).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from orca.analytics.greeks import greeks as bs_greeks
from orca.analytics.greeks import implied_vol, year_fraction
from orca.config import Underlying, get_settings
from orca.domain import ChainSnapshot, OptionQuote, OptionType, Quote
from orca.feeds.base import FeedError, MarketDataProvider
from orca.logging_setup import get_logger

log = get_logger(__name__)

API_BASE = "https://api.upstox.com/v2"


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

    def __init__(self, access_token: str | None = None, timeout: float = 15.0) -> None:
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
            base_url=API_BASE,
            timeout=timeout,
            headers={"Accept": "application/json", "Authorization": f"Bearer {self.token}"},
        )

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
        if resp.status_code >= 400:
            raise FeedError(f"upstox {path} -> {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        if isinstance(body, dict) and body.get("status") == "error":
            raise FeedError(f"upstox error: {body.get('errors') or body}")
        return body.get("data") if isinstance(body, dict) else body

    # -- spot ------------------------------------------------------------
    def get_spot(self, underlying: Underlying) -> float:
        key = underlying.upstox_instrument_key
        data = self._get("/market-quote/ltp", {"instrument_key": key}) or {}
        for payload in data.values():
            price = payload.get("last_price")
            if price:
                return float(price)
        raise FeedError(f"no LTP returned for {key}")

    # -- expiries --------------------------------------------------------
    def get_expiries(self, underlying: Underlying) -> list[date]:
        data = self._get("/option/contract", {"instrument_key": underlying.upstox_instrument_key})
        expiries = set()
        for row in data or []:
            raw = row.get("expiry")
            if raw:
                expiries.add(datetime.strptime(raw[:10], "%Y-%m-%d").date())
        today = date.today()
        return sorted(e for e in expiries if e >= today)

    # -- chain -----------------------------------------------------------
    def get_option_chain(self, underlying: Underlying, expiry: date) -> ChainSnapshot:
        data = self._get(
            "/option/chain",
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
        iv = gk.get("iv")
        # Upstox reports IV in percent; normalise to a decimal.
        iv = float(iv) / 100.0 if iv not in (None, 0) else None
        if iv is None and ltp > 0:
            iv = implied_vol(ltp, spot, strike, t, option_type, self.r)

        delta, gamma, theta, vega = (
            gk.get("delta"),
            gk.get("gamma"),
            gk.get("theta"),
            gk.get("vega"),
        )
        if delta is None and iv:
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
        out: dict[str, Quote] = {}
        now = datetime.now(UTC)
        # Upstox caps the number of keys per request; chunk to stay well under it.
        for i in range(0, len(instrument_keys), 25):
            chunk = instrument_keys[i : i + 25]
            data = self._get("/market-quote/ltp", {"instrument_key": ",".join(chunk)}) or {}
            by_key = {
                (payload.get("instrument_token") or name): payload for name, payload in data.items()
            }
            for key in chunk:
                payload = by_key.get(key)
                if payload is None:
                    # Response keys look like "NSE_FO:NIFTY25N1125000CE"; match on suffix.
                    payload = next((p for n, p in data.items() if key.split("|")[-1] in n), None)
                if payload:
                    out[key] = Quote(
                        instrument_key=key, ltp=float(payload.get("last_price") or 0), ts=now
                    )
        return out

    def get_daily_closes(self, underlying: Underlying, days: int = 60) -> list[tuple[date, float]]:
        to_d = date.today()
        from_d = to_d - timedelta(days=int(days * 1.6) + 5)
        key = underlying.upstox_instrument_key
        data = self._get(f"/historical-candle/{key}/day/{to_d}/{from_d}") or {}
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
            self._get("/user/profile")
            return {"provider": self.name, "ok": True}
        except FeedError as exc:
            return {"provider": self.name, "ok": False, "error": str(exc)}

    def close(self) -> None:
        self._client.close()
