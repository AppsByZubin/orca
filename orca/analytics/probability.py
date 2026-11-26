"""Market-implied probabilities for where an underlying expires.

The same zero-drift lognormal model `payoff.prob_of_profit` uses to give every
candidate its probability of profit: log(S_T / S) ~ N(0, sigma^2) with
sigma = IV * sqrt(t). Implied vol is the market's price of movement, not a
forecast - treat these as "what the options are pricing", not a risk model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from orca.analytics.greeks import norm_cdf, norm_pdf


def p_below(level: float, spot: float, sigma: float) -> float:
    """P(expiry price < level)."""
    if level <= 0:
        return 0.0
    if sigma <= 0:
        return 1.0 if level > spot else 0.0
    return norm_cdf(math.log(level / spot) / sigma)


def p_touch(level: float, spot: float, sigma: float) -> float:
    """Approximate P(price trades through `level` before expiry).

    Reflection principle for driftless Brownian motion: roughly twice the
    probability of finishing beyond the level.
    """
    beyond = 1.0 - p_below(level, spot, sigma) if level >= spot else p_below(level, spot, sigma)
    return min(1.0, 2.0 * beyond)


def density(price: float, spot: float, sigma: float) -> float:
    """Lognormal probability density of the expiry price (per unit of price)."""
    if price <= 0 or sigma <= 0:
        return 0.0
    return norm_pdf(math.log(price / spot) / sigma) / (price * sigma)


@dataclass
class SigmaRange:
    sigmas: int
    lower: float
    upper: float
    prob: float  # P(lower <= expiry <= upper)


def sigma_ranges(spot: float, sigma: float, ks: tuple[int, ...] = (1, 2, 3)) -> list[SigmaRange]:
    return [
        SigmaRange(k, spot * math.exp(-k * sigma), spot * math.exp(k * sigma), 2 * norm_cdf(k) - 1)
        for k in ks
    ]


def density_curve(
    spot: float, sigma: float, points: int = 121, span: float = 3.5
) -> list[tuple[float, float, float]]:
    """(price, density, P(below price)) from -span to +span sigma."""
    lo, hi = spot * math.exp(-span * sigma), spot * math.exp(span * sigma)
    step = (hi - lo) / (points - 1)
    out = []
    for i in range(points):
        x = lo + i * step
        out.append((x, density(x, spot, sigma), p_below(x, spot, sigma)))
    return out
