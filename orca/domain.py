"""Core in-memory domain types shared by every layer.

These are deliberately plain dataclasses, not DB rows and not API schemas, so
that the feed, analytics, strategy and ML code can be unit-tested without a
database or a web server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum


class OptionType(StrEnum):
    CE = "CE"
    PE = "PE"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """+1 for long, -1 for short - multiply premium/greeks by this."""
        return 1 if self is Side.BUY else -1


class TradeStatus(StrEnum):
    DRAFT = "DRAFT"  # built from a candidate, nothing sent
    PENDING = "PENDING"  # orders sent, not fully filled
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"


class LegStatus(StrEnum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Quote:
    """Last traded snapshot of any instrument (index, future, option)."""

    instrument_key: str
    ltp: float
    ts: datetime | None = None
    bid: float | None = None
    ask: float | None = None
    volume: int | None = None
    oi: int | None = None
    prev_close: float | None = None

    @property
    def mid(self) -> float:
        if self.bid and self.ask and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.ltp


@dataclass
class OptionQuote:
    """One option contract in a chain snapshot."""

    underlying: str
    expiry: date
    strike: float
    option_type: OptionType
    instrument_key: str = ""  # feed-side id (Upstox)
    tradingsymbol: str = ""  # execution-side id (Kite)
    ltp: float = 0.0
    bid: float | None = None
    ask: float | None = None
    volume: int = 0
    oi: int = 0
    oi_change: int = 0
    iv: float | None = None  # as a decimal, 0.14 == 14%
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None

    @property
    def mid(self) -> float:
        if self.bid and self.ask and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.ltp

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as a fraction of mid - the liquidity filter."""
        if not self.bid or not self.ask or self.mid <= 0:
            return 1.0
        return (self.ask - self.bid) / self.mid

    @property
    def abs_delta(self) -> float:
        return abs(self.delta) if self.delta is not None else 0.0


@dataclass
class ChainSnapshot:
    """A full option chain for one underlying/expiry at one point in time."""

    underlying: str
    expiry: date
    spot: float
    ts: datetime
    options: list[OptionQuote] = field(default_factory=list)
    futures_price: float | None = None
    source: str = "mock"

    # -- lookups ---------------------------------------------------------
    def strikes(self) -> list[float]:
        return sorted({o.strike for o in self.options})

    def get(self, strike: float, option_type: OptionType) -> OptionQuote | None:
        for o in self.options:
            if abs(o.strike - strike) < 1e-6 and o.option_type is option_type:
                return o
        return None

    def side(self, option_type: OptionType) -> list[OptionQuote]:
        return sorted(
            (o for o in self.options if o.option_type is option_type), key=lambda o: o.strike
        )

    def atm_strike(self) -> float:
        return min(self.strikes(), key=lambda s: abs(s - self.spot), default=self.spot)

    def atm_iv(self) -> float | None:
        atm = self.atm_strike()
        ivs = [
            o.iv
            for o in self.options
            if abs(o.strike - atm) < 1e-6 and o.iv is not None and o.iv > 0
        ]
        return sum(ivs) / len(ivs) if ivs else None

    def nearest_by_delta(self, option_type: OptionType, target_delta: float) -> OptionQuote | None:
        """Strike whose |delta| is closest to `target_delta` (e.g. 0.15)."""
        pool = [o for o in self.side(option_type) if o.delta is not None]
        if not pool:
            return None
        return min(pool, key=lambda o: abs(o.abs_delta - abs(target_delta)))

    def dte(self, as_of: date | None = None) -> int:
        as_of = as_of or self.ts.date()
        return max((self.expiry - as_of).days, 0)


@dataclass
class LegSpec:
    """A leg we intend to trade (pre-execution intent)."""

    underlying: str
    expiry: date
    strike: float
    option_type: OptionType
    side: Side
    lots: int = 1
    lot_size: int = 1
    instrument_key: str = ""
    tradingsymbol: str = ""
    ref_price: float = 0.0  # price at the time the candidate was built

    @property
    def quantity(self) -> int:
        return self.lots * self.lot_size

    @property
    def signed_premium(self) -> float:
        """Cash impact per unit: negative when we pay, positive when we receive."""
        return -self.side.sign * self.ref_price

    def label(self) -> str:
        return f"{self.side.value} {int(self.strike)}{self.option_type.value} x{self.lots}"


@dataclass
class PayoffProfile:
    max_profit: float | None
    max_loss: float | None
    breakevens: list[float]
    net_credit: float
    curve: list[tuple[float, float]] = field(default_factory=list)
    prob_of_profit: float | None = None


@dataclass
class Candidate:
    """A scored, tradable idea produced by the scanner."""

    strategy_id: str
    template: str
    underlying: str
    expiry: date
    legs: list[LegSpec]
    spot: float
    net_credit: float = 0.0  # per lot, in rupees
    max_loss: float | None = None
    breakevens: list[float] = field(default_factory=list)
    prob_of_profit: float | None = None
    margin_estimate: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    score: float | None = None  # ML probability of a profitable outcome
    expected_return: float | None = None
    score_source: str = "none"  # "model:<name>" or "heuristic"
    reasons: list[str] = field(default_factory=list)
    ts: datetime | None = None
    signal_id: int | None = None  # set once the candidate is persisted as a Signal

    @property
    def credit_to_risk(self) -> float | None:
        if not self.max_loss:
            return None
        return self.net_credit / self.max_loss if self.max_loss else None

    def describe(self) -> str:
        return f"{self.underlying} {self.expiry:%d-%b} " + " / ".join(
            leg.label() for leg in self.legs
        )
