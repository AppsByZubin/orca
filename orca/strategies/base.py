"""Strategy templates turn a chain snapshot into candidate leg-sets.

A template is a function:

    (chain, underlying, params, lots) -> list[Candidate]

Register it with @template("name") and reference that name from
config/strategies.yaml. All rupee amounts on a Candidate are for the *full*
size (lots x lot_size), not per unit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from orca.analytics.greeks import year_fraction
from orca.analytics.margin import estimate_margin
from orca.analytics.payoff import legs_to_tuples, payoff_profile
from orca.config import Underlying
from orca.domain import Candidate, ChainSnapshot, LegSpec, OptionQuote, Side
from orca.logging_setup import get_logger

log = get_logger(__name__)
IST = ZoneInfo("Asia/Kolkata")

TemplateFn = Callable[[ChainSnapshot, Underlying, dict, int], list[Candidate]]
_REGISTRY: dict[str, TemplateFn] = {}


def template(name: str) -> Callable[[TemplateFn], TemplateFn]:
    def decorator(fn: TemplateFn) -> TemplateFn:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_template(name: str) -> TemplateFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy template {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_templates() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def make_leg(quote: OptionQuote, side: Side, underlying: Underlying, lots: int) -> LegSpec:
    return LegSpec(
        underlying=underlying.symbol,
        expiry=quote.expiry,
        strike=quote.strike,
        option_type=quote.option_type,
        side=side,
        lots=lots,
        lot_size=underlying.lot_size,
        instrument_key=quote.instrument_key,
        tradingsymbol=quote.tradingsymbol,
        # Sells are marked at bid, buys at ask: assume we cross the spread.
        ref_price=(quote.bid or quote.ltp) if side is Side.SELL else (quote.ask or quote.ltp),
    )


def passes_liquidity(quotes: Iterable[OptionQuote], defaults: dict) -> tuple[bool, str]:
    max_spread = float(defaults.get("max_spread_pct", 1.0))
    min_oi = int(defaults.get("min_open_interest", 0))
    for q in quotes:
        if q.ltp <= 0:
            return False, f"{int(q.strike)}{q.option_type.value} has no price"
        if q.spread_pct > max_spread:
            return False, f"{int(q.strike)}{q.option_type.value} spread {q.spread_pct:.1%}"
        if q.oi < min_oi:
            return False, f"{int(q.strike)}{q.option_type.value} OI {q.oi}"
    return True, ""


def build_candidate(
    strategy_id: str,
    template_name: str,
    chain: ChainSnapshot,
    underlying: Underlying,
    legs: Sequence[LegSpec],
    reasons: list[str] | None = None,
) -> Candidate:
    """Attach payoff, risk and margin numbers to a leg-set."""
    tuples = legs_to_tuples(legs)
    t = year_fraction(chain.expiry, datetime.now(IST))
    profile = payoff_profile(tuples, chain.spot, iv=chain.atm_iv(), t_years=t, span_pct=0.10)
    return Candidate(
        strategy_id=strategy_id,
        template=template_name,
        underlying=underlying.symbol,
        expiry=chain.expiry,
        legs=list(legs),
        spot=chain.spot,
        net_credit=round(profile.net_credit, 2),
        max_loss=None if profile.max_loss is None else round(abs(min(profile.max_loss, 0.0)), 2),
        breakevens=[round(b, 2) for b in profile.breakevens],
        prob_of_profit=profile.prob_of_profit,
        margin_estimate=estimate_margin(legs, chain.spot),
        reasons=reasons or [],
        ts=chain.ts,
    )


def meets_credit_floor(candidate: Candidate, defaults: dict, lots: int) -> bool:
    floor = float(defaults.get("min_credit_per_lot", 0)) * max(lots, 1)
    return candidate.net_credit >= floor
