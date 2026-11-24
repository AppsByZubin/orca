"""Expiry payoff of an arbitrary multi-leg position.

Works for both a Candidate's LegSpec list and a live Trade's legs, which is
why it takes a small structural tuple instead of either class.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from orca.analytics.greeks import norm_cdf
from orca.domain import LegSpec, OptionType, PayoffProfile, Side

# (strike, option_type, side, quantity, entry_price)
LegTuple = tuple[float, OptionType, Side, int, float]


def legs_to_tuples(legs: Iterable[LegSpec]) -> list[LegTuple]:
    return [(leg.strike, leg.option_type, leg.side, leg.quantity, leg.ref_price) for leg in legs]


def intrinsic(spot: float, strike: float, option_type: OptionType) -> float:
    if option_type is OptionType.CE:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def pnl_at(spot: float, legs: Sequence[LegTuple]) -> float:
    """Position P&L (rupees) if the underlying expires at `spot`."""
    total = 0.0
    for strike, option_type, side, qty, entry in legs:
        value = intrinsic(spot, strike, option_type)
        total += (value - entry) * side.sign * qty
    return total


def net_credit(legs: Sequence[LegTuple]) -> float:
    """Positive when the position collects premium."""
    return sum(-side.sign * entry * qty for _s, _t, side, qty, entry in legs)


def payoff_profile(
    legs: Sequence[LegTuple],
    spot: float,
    span_pct: float = 0.12,
    points: int = 161,
    iv: float | None = None,
    t_years: float | None = None,
) -> PayoffProfile:
    """Payoff curve, breakevens and max profit/loss over +/- span_pct of spot."""
    if not legs:
        return PayoffProfile(0.0, 0.0, [], 0.0, [])

    lo, hi = spot * (1 - span_pct), spot * (1 + span_pct)
    step = (hi - lo) / (points - 1)
    curve = [(lo + i * step, pnl_at(lo + i * step, legs)) for i in range(points)]

    # Breakevens: linear interpolation wherever the curve crosses zero.
    breakevens: list[float] = []
    for (x0, y0), (x1, y1) in zip(curve, curve[1:], strict=False):
        if y0 == 0.0:
            breakevens.append(x0)
        elif y0 * y1 < 0:
            breakevens.append(x0 + (x1 - x0) * (-y0) / (y1 - y0))

    ys = [y for _x, y in curve]
    max_profit: float | None = max(ys)
    max_loss: float | None = min(ys)

    # Naked short legs have unbounded risk beyond the plotted window.
    shorts = [lg for lg in legs if lg[2] is Side.SELL]
    longs = [lg for lg in legs if lg[2] is Side.BUY]
    naked_call = sum(q for _s, t, sd, q, _e in shorts if t is OptionType.CE) > sum(
        q for _s, t, sd, q, _e in longs if t is OptionType.CE
    )
    naked_put = sum(q for _s, t, sd, q, _e in shorts if t is OptionType.PE) > sum(
        q for _s, t, sd, q, _e in longs if t is OptionType.PE
    )
    if naked_call or naked_put:
        max_loss = None  # undefined / theoretically unlimited

    pop = (
        prob_of_profit(breakevens, spot, iv, t_years, legs)
        if iv and t_years and breakevens
        else None
    )
    return PayoffProfile(
        max_profit=max_profit,
        max_loss=max_loss,
        breakevens=sorted(breakevens),
        net_credit=net_credit(legs),
        curve=curve,
        prob_of_profit=pop,
    )


def prob_of_profit(
    breakevens: Sequence[float],
    spot: float,
    iv: float,
    t_years: float,
    legs: Sequence[LegTuple] | None = None,
) -> float | None:
    """Lognormal probability that expiry lands in the profitable region.

    Assumes zero drift. Good enough to rank candidates; it is not a risk model.
    """
    if not breakevens or iv <= 0 or t_years <= 0:
        return None
    sigma = iv * math.sqrt(t_years)

    def cdf(x: float) -> float:
        if x <= 0:
            return 0.0
        return norm_cdf((math.log(x / spot) + 0.5 * sigma**2 * 0) / sigma)

    bes = sorted(breakevens)
    if len(bes) == 1:
        # One crossing: profitable on whichever side currently pays.
        probe = bes[0] * 0.95
        profitable_below = legs is not None and pnl_at(probe, legs) > 0
        return cdf(bes[0]) if profitable_below else 1 - cdf(bes[0])

    lower, upper = bes[0], bes[-1]
    mid = 0.5 * (lower + upper)
    inside_is_profit = legs is None or pnl_at(mid, legs) > 0
    inside = cdf(upper) - cdf(lower)
    return inside if inside_is_profit else 1 - inside
