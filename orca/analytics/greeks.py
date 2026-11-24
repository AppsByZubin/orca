"""Black-Scholes pricing, greeks and implied vol.

Deliberately dependency-free (math only) so it can run anywhere. Index options
on NSE/BSE are European and cash-settled, which is exactly what BS assumes.
"""

from __future__ import annotations

import math
from datetime import date, datetime

from orca.domain import OptionType

SQRT_2PI = math.sqrt(2.0 * math.pi)
TRADING_DAYS = 252.0
MIN_T = 1.0 / (365.0 * 24.0)  # floor so expiry-day math stays finite


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def year_fraction(expiry: date | datetime, now: datetime | None = None) -> float:
    """Time to expiry in years, measured to 15:30 IST on expiry day."""
    from zoneinfo import ZoneInfo

    ist = ZoneInfo("Asia/Kolkata")
    now = now or datetime.now(ist)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ist)
    if isinstance(expiry, datetime):
        exp_dt = expiry if expiry.tzinfo else expiry.replace(tzinfo=ist)
    else:
        exp_dt = datetime.combine(expiry, datetime.min.time(), tzinfo=ist).replace(
            hour=15, minute=30
        )
    seconds = (exp_dt - now.astimezone(ist)).total_seconds()
    return max(seconds / (365.0 * 24.0 * 3600.0), MIN_T)


def _d1_d2(s: float, k: float, t: float, r: float, sigma: float) -> tuple[float, float]:
    vol = max(sigma, 1e-6) * math.sqrt(max(t, MIN_T))
    d1 = (math.log(max(s, 1e-9) / max(k, 1e-9)) + (r + 0.5 * sigma * sigma) * t) / vol
    return d1, d1 - vol


def black_scholes(
    s: float, k: float, t: float, sigma: float, option_type: OptionType, r: float = 0.065
) -> float:
    """Theoretical premium of a European option."""
    if t <= MIN_T or sigma <= 0:
        intrinsic = max(s - k, 0.0) if option_type is OptionType.CE else max(k - s, 0.0)
        return intrinsic
    d1, d2 = _d1_d2(s, k, t, r, sigma)
    disc = math.exp(-r * t)
    if option_type is OptionType.CE:
        return s * norm_cdf(d1) - k * disc * norm_cdf(d2)
    return k * disc * norm_cdf(-d2) - s * norm_cdf(-d1)


def greeks(
    s: float, k: float, t: float, sigma: float, option_type: OptionType, r: float = 0.065
) -> dict[str, float]:
    """Delta, gamma, theta (per calendar day), vega (per 1 vol point), rho."""
    t = max(t, MIN_T)
    sigma = max(sigma, 1e-6)
    d1, d2 = _d1_d2(s, k, t, r, sigma)
    disc = math.exp(-r * t)
    pdf = norm_pdf(d1)
    sqrt_t = math.sqrt(t)

    if option_type is OptionType.CE:
        delta = norm_cdf(d1)
        theta_year = -s * pdf * sigma / (2 * sqrt_t) - r * k * disc * norm_cdf(d2)
        rho = k * t * disc * norm_cdf(d2) / 100.0
    else:
        delta = norm_cdf(d1) - 1.0
        theta_year = -s * pdf * sigma / (2 * sqrt_t) + r * k * disc * norm_cdf(-d2)
        rho = -k * t * disc * norm_cdf(-d2) / 100.0

    return {
        "delta": delta,
        "gamma": pdf / (s * sigma * sqrt_t),
        "theta": theta_year / 365.0,
        "vega": s * pdf * sqrt_t / 100.0,
        "rho": rho,
    }


def implied_vol(
    price: float,
    s: float,
    k: float,
    t: float,
    option_type: OptionType,
    r: float = 0.065,
    lo: float = 0.005,
    hi: float = 5.0,
    tol: float = 1e-5,
    max_iter: int = 80,
) -> float | None:
    """Bisection solve for sigma. Returns None if the price is not arbitrage-free."""
    if price <= 0 or t <= MIN_T:
        return None
    intrinsic = max(s - k, 0.0) if option_type is OptionType.CE else max(k - s, 0.0)
    if price < intrinsic - 0.01:
        return None
    f_lo = black_scholes(s, k, t, lo, option_type, r) - price
    f_hi = black_scholes(s, k, t, hi, option_type, r) - price
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = black_scholes(s, k, t, mid, option_type, r) - price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def realized_vol(closes: list[float], window: int = 20, annualize: bool = True) -> float | None:
    """Close-to-close realized volatility from a price series."""
    if len(closes) < window + 1:
        return None
    rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(len(closes) - window, len(closes))
        if closes[i - 1] > 0
    ]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    return sd * math.sqrt(TRADING_DAYS) if annualize else sd
