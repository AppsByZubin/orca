"""Pricing, greeks and payoff invariants."""

from __future__ import annotations

import math

import pytest

from orca.analytics.greeks import black_scholes, greeks, implied_vol, year_fraction
from orca.analytics.payoff import net_credit, payoff_profile, pnl_at
from orca.domain import OptionType, Side


def test_put_call_parity() -> None:
    s, k, t, vol, r = 25000.0, 25000.0, 0.05, 0.13, 0.065
    call = black_scholes(s, k, t, vol, OptionType.CE, r)
    put = black_scholes(s, k, t, vol, OptionType.PE, r)
    assert math.isclose(call - put, s - k * math.exp(-r * t), abs_tol=0.5)


def test_implied_vol_round_trip() -> None:
    s, k, t, vol = 25000.0, 25300.0, 0.02, 0.145
    price = black_scholes(s, k, t, vol, OptionType.CE)
    recovered = implied_vol(price, s, k, t, OptionType.CE)
    assert recovered is not None
    assert abs(recovered - vol) < 1e-3


def test_greeks_signs() -> None:
    call = greeks(25000, 25200, 0.02, 0.13, OptionType.CE)
    put = greeks(25000, 24800, 0.02, 0.13, OptionType.PE)
    assert 0 < call["delta"] < 1
    assert -1 < put["delta"] < 0
    assert call["gamma"] > 0 and call["vega"] > 0
    assert call["theta"] < 0  # long options bleed


def test_year_fraction_is_positive() -> None:
    from datetime import date, timedelta

    assert year_fraction(date.today() + timedelta(days=7)) > 0


def test_short_strangle_payoff_shape() -> None:
    legs = [
        (25300.0, OptionType.CE, Side.SELL, 75, 60.0),
        (24700.0, OptionType.PE, Side.SELL, 75, 55.0),
    ]
    assert net_credit(legs) == (60.0 + 55.0) * 75
    # Max profit between the strikes, losses outside.
    assert pnl_at(25000, legs) > 0
    assert pnl_at(26500, legs) < 0
    assert pnl_at(23500, legs) < 0

    profile = payoff_profile(legs, 25000.0, iv=0.13, t_years=0.02)
    assert len(profile.breakevens) == 2
    assert profile.max_loss is None  # naked shorts: undefined
    assert profile.prob_of_profit is not None


def test_iron_condor_has_defined_loss() -> None:
    legs = [
        (25300.0, OptionType.CE, Side.SELL, 75, 60.0),
        (25500.0, OptionType.CE, Side.BUY, 75, 25.0),
        (24700.0, OptionType.PE, Side.SELL, 75, 55.0),
        (24500.0, OptionType.PE, Side.BUY, 75, 22.0),
    ]
    profile = payoff_profile(legs, 25000.0, span_pct=0.2)
    assert profile.max_loss is not None and profile.max_loss < 0
    assert profile.max_profit > 0


# --- probability model ---------------------------------------------------------
def test_probability_math() -> None:
    from orca.analytics.probability import density_curve, p_below, p_touch, sigma_ranges

    spot, sigma = 100.0, 0.1
    assert p_below(spot, spot, sigma) == pytest.approx(0.5)
    assert p_below(spot * math.exp(sigma), spot, sigma) == pytest.approx(0.8413, abs=1e-4)
    assert p_touch(spot * math.exp(sigma), spot, sigma) == pytest.approx(2 * 0.1587, abs=1e-4)
    assert p_touch(spot, spot, sigma) == 1.0
    one, two, _ = sigma_ranges(spot, sigma)
    assert one.prob == pytest.approx(0.6827, abs=1e-4)
    assert two.prob == pytest.approx(0.9545, abs=1e-4)
    assert one.lower == pytest.approx(spot / math.exp(sigma))
    curve = density_curve(spot, sigma, points=401, span=6)
    step = curve[1][0] - curve[0][0]
    assert sum(d for _x, d, _p in curve) * step == pytest.approx(1.0, abs=1e-3)  # a density
    assert [p for *_, p in curve] == sorted(p for *_, p in curve)
