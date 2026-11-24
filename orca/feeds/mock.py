"""Synthetic feed so the whole stack runs with no broker credentials.

Prices come from Black-Scholes on a simple vol smile, with a deterministic
intraday random walk for the spot. Deterministic per (symbol, date, minute),
so repeated API calls do not teleport the market.
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, date, datetime, timedelta

from orca.analytics.greeks import black_scholes, greeks, year_fraction
from orca.config import Underlying, get_settings
from orca.domain import ChainSnapshot, OptionQuote, OptionType, Quote
from orca.feeds.base import MarketDataProvider
from orca.symbols import kite_tradingsymbol

BASE_SPOT = {"NIFTY": 25000.0, "SENSEX": 82000.0, "BANKNIFTY": 54000.0}
BASE_IV = {"NIFTY": 0.12, "SENSEX": 0.13, "BANKNIFTY": 0.15}


def _jitter(seed: str, lo: float = -1.0, hi: float = 1.0) -> float:
    """Deterministic pseudo-random value in [lo, hi] from a string seed."""
    digest = hashlib.sha256(seed.encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return lo + unit * (hi - lo)


class MockProvider(MarketDataProvider):
    name = "mock"

    def __init__(self, seed: str = "orca") -> None:
        self.seed = seed
        self.r = get_settings().risk_free_rate

    # -- spot ------------------------------------------------------------
    def get_spot(self, underlying: Underlying) -> float:
        base = BASE_SPOT.get(underlying.symbol, 20000.0)
        now = datetime.now(UTC)
        day_drift = _jitter(f"{self.seed}:{underlying.symbol}:{now:%Y-%m-%d}", -0.012, 0.012)
        minute_noise = _jitter(
            f"{self.seed}:{underlying.symbol}:{now:%Y-%m-%d-%H-%M}", -0.002, 0.002
        )
        return round(base * (1 + day_drift + minute_noise), 2)

    # -- expiries --------------------------------------------------------
    def get_expiries(self, underlying: Underlying, count: int = 4) -> list[date]:
        today = date.today()
        out: list[date] = []
        cursor = today
        while len(out) < count:
            delta = (underlying.weekly_expiry_weekday - cursor.weekday()) % 7
            candidate = cursor + timedelta(days=delta)
            if candidate >= today and candidate not in out:
                out.append(candidate)
            cursor = candidate + timedelta(days=1)
        return out

    # -- chain -----------------------------------------------------------
    def _iv_for(self, underlying: Underlying, strike: float, spot: float, t: float) -> float:
        """A crude smile: IV rises with moneyness distance and with short DTE."""
        atm_iv = BASE_IV.get(underlying.symbol, 0.14)
        atm_iv *= 1 + _jitter(f"{self.seed}:{underlying.symbol}:iv:{date.today()}", -0.15, 0.25)
        moneyness = abs(math.log(max(strike, 1) / max(spot, 1)))
        smile = 1 + 7.0 * moneyness**1.5
        term = 1 + max(0.0, 0.03 - t) * 6.0
        return max(0.03, atm_iv * smile * term)

    def get_option_chain(self, underlying: Underlying, expiry: date) -> ChainSnapshot:
        spot = self.get_spot(underlying)
        step = underlying.strike_step
        atm = round(spot / step) * step
        t = year_fraction(expiry)
        ts = datetime.now(UTC)
        options: list[OptionQuote] = []

        for i in range(-20, 21):
            strike = atm + i * step
            if strike <= 0:
                continue
            for option_type in (OptionType.CE, OptionType.PE):
                iv = self._iv_for(underlying, strike, spot, t)
                price = black_scholes(spot, strike, t, iv, option_type, self.r)
                price = max(round(price, 2), 0.05)
                g = greeks(spot, strike, t, iv, option_type, self.r)
                half_spread = max(0.05, price * 0.004 + abs(i) * 0.02)
                dist = abs(i)
                oi = int(max(1000, 900_000 * math.exp(-0.08 * dist * dist / 4)))
                options.append(
                    OptionQuote(
                        underlying=underlying.symbol,
                        expiry=expiry,
                        strike=float(strike),
                        option_type=option_type,
                        instrument_key=f"MOCK|{underlying.symbol}|{expiry:%y%m%d}|{int(strike)}|{option_type.value}",
                        tradingsymbol=kite_tradingsymbol(
                            underlying, expiry, float(strike), option_type
                        ),
                        ltp=price,
                        bid=round(max(price - half_spread, 0.05), 2),
                        ask=round(price + half_spread, 2),
                        volume=int(oi * 0.35),
                        oi=oi,
                        oi_change=int(_jitter(f"{strike}{option_type}{ts:%H%M}", -50_000, 50_000)),
                        iv=round(iv, 4),
                        delta=round(g["delta"], 4),
                        gamma=round(g["gamma"], 6),
                        theta=round(g["theta"], 3),
                        vega=round(g["vega"], 3),
                    )
                )

        return ChainSnapshot(
            underlying=underlying.symbol,
            expiry=expiry,
            spot=spot,
            ts=ts,
            options=options,
            futures_price=round(spot * 1.0008, 2),
            source=self.name,
        )

    # -- quotes ----------------------------------------------------------
    def get_quotes(self, instrument_keys: list[str]) -> dict[str, Quote]:
        """Re-price mock instrument keys (MOCK|SYM|YYMMDD|STRIKE|CE)."""
        from orca.config import get_universe

        out: dict[str, Quote] = {}
        now = datetime.now(UTC)
        for key in instrument_keys:
            parts = key.split("|")
            if len(parts) != 5 or parts[0] != "MOCK":
                out[key] = Quote(instrument_key=key, ltp=0.0, ts=now)
                continue
            _, symbol, exp, strike, opt = parts
            try:
                underlying = get_universe().get(symbol)
            except KeyError:
                continue
            expiry = datetime.strptime(exp, "%y%m%d").date()
            spot = self.get_spot(underlying)
            t = year_fraction(expiry)
            option_type = OptionType(opt)
            iv = self._iv_for(underlying, float(strike), spot, t)
            price = max(
                round(black_scholes(spot, float(strike), t, iv, option_type, self.r), 2), 0.05
            )
            out[key] = Quote(
                instrument_key=key,
                ltp=price,
                ts=now,
                bid=round(max(price - 0.1, 0.05), 2),
                ask=round(price + 0.1, 2),
            )
        return out

    def get_daily_closes(self, underlying: Underlying, days: int = 60) -> list[tuple[date, float]]:
        base = BASE_SPOT.get(underlying.symbol, 20000.0)
        out: list[tuple[date, float]] = []
        level = base * 0.97
        for i in range(days, 0, -1):
            d = date.today() - timedelta(days=i)
            if d.weekday() >= 5:
                continue
            level *= 1 + _jitter(f"{self.seed}:{underlying.symbol}:{d}", -0.009, 0.0095)
            out.append((d, round(level, 2)))
        return out
