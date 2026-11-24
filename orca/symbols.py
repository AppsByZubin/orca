"""Symbol translation between the data feed and the order gateway.

ORCA reads prices from Upstox and sends orders to Zerodha Kite, so every leg
carries two identifiers:

    instrument_key   "NSE_FO|12345"            Upstox - what we quote
    tradingsymbol    "NIFTY25N1125000CE"       Kite   - what we trade

Kite option tradingsymbols:
    weekly   <NAME><YY><M><DD><STRIKE><CE|PE>   NIFTY25N1125000CE   (M: 1-9,O,N,D)
    monthly  <NAME><YY><MMM><STRIKE><CE|PE>     NIFTY25NOV25000CE

The formatter below is a convenience for display and for the paper gateway.
For live orders prefer `resolve_from_instruments()`, which looks the contract
up in Kite's own instrument dump - that is the only authoritative mapping and
it protects you from format changes.
"""

from __future__ import annotations

import csv
import io
from datetime import date, timedelta
from functools import lru_cache

import httpx

from orca.config import Underlying
from orca.domain import OptionType
from orca.logging_setup import get_logger

log = get_logger(__name__)

MONTH_CODE = {10: "O", 11: "N", 12: "D"}
KITE_INSTRUMENTS_URL = "https://api.kite.trade/instruments/{exchange}"


def is_monthly_expiry(expiry: date) -> bool:
    """True when no later weekly expiry exists in the same calendar month."""
    return (expiry + timedelta(days=7)).month != expiry.month


def _fmt_strike(strike: float) -> str:
    return str(int(strike)) if float(strike).is_integer() else str(strike)


def kite_tradingsymbol(
    underlying: Underlying,
    expiry: date,
    strike: float,
    option_type: OptionType,
    monthly: bool | None = None,
) -> str:
    """Format a Kite option tradingsymbol (best-effort, see module docstring)."""
    name = underlying.kite_symbol_prefix
    yy = f"{expiry.year % 100:02d}"
    if monthly is None:
        monthly = is_monthly_expiry(expiry)
    if monthly:
        return f"{name}{yy}{expiry.strftime('%b').upper()}{_fmt_strike(strike)}{option_type.value}"
    month = MONTH_CODE.get(expiry.month, str(expiry.month))
    return f"{name}{yy}{month}{expiry.day:02d}{_fmt_strike(strike)}{option_type.value}"


@lru_cache(maxsize=4)
def _kite_instruments(exchange: str) -> list[dict]:
    """Download and cache Kite's instrument dump for an exchange (public CSV)."""
    url = KITE_INSTRUMENTS_URL.format(exchange=exchange)
    try:
        resp = httpx.get(url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("could not fetch kite instruments for %s: %s", exchange, exc)
        return []
    return list(csv.DictReader(io.StringIO(resp.text)))


def resolve_from_instruments(
    underlying: Underlying, expiry: date, strike: float, option_type: OptionType
) -> str | None:
    """Authoritative lookup of a tradingsymbol in Kite's instrument dump."""
    for row in _kite_instruments(underlying.option_exchange):
        if (
            row.get("name") == underlying.kite_symbol_prefix
            and row.get("expiry") == expiry.isoformat()
            and row.get("instrument_type") == option_type.value
            and abs(float(row.get("strike") or 0) - strike) < 1e-6
        ):
            return row.get("tradingsymbol")
    return None


def clear_instrument_cache() -> None:
    _kite_instruments.cache_clear()
