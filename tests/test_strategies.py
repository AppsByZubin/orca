"""Candidate generation from a chain snapshot."""

from __future__ import annotations

from orca.domain import Side
from orca.strategies import available_templates, get_template


def test_templates_registered() -> None:
    assert {"short_strangle", "iron_condor", "credit_spread"} <= set(available_templates())


def test_short_strangle_builds_two_short_legs(chain, nifty) -> None:
    candidates = get_template("short_strangle")(
        chain, nifty, {"short_delta": [0.15], "_defaults": {}}, 1
    )
    assert candidates
    cand = candidates[0]
    assert len(cand.legs) == 2
    assert all(leg.side is Side.SELL for leg in cand.legs)
    assert cand.net_credit > 0
    assert len(cand.breakevens) == 2
    assert cand.margin_estimate and cand.margin_estimate > 0


def test_iron_condor_is_defined_risk(chain, nifty) -> None:
    candidates = get_template("iron_condor")(
        chain, nifty, {"short_delta": [0.15], "wing_width_strikes": [4], "_defaults": {}}, 1
    )
    assert candidates
    cand = candidates[0]
    assert len(cand.legs) == 4
    assert cand.max_loss is not None and cand.max_loss > 0
    assert cand.net_credit > 0
    # Wings must be further out than the shorts.
    shorts = [leg for leg in cand.legs if leg.side is Side.SELL]
    longs = [leg for leg in cand.legs if leg.side is Side.BUY]
    assert max(leg.strike for leg in longs) > max(s.strike for s in shorts)
    assert min(leg.strike for leg in longs) < min(s.strike for s in shorts)


def test_credit_spread_pairs_one_short_one_long(chain, nifty) -> None:
    candidates = get_template("credit_spread")(
        chain,
        nifty,
        {"side": ["PE"], "short_delta": [0.25], "wing_width_strikes": [4], "_defaults": {}},
        1,
    )
    assert candidates
    cand = candidates[0]
    assert [leg.side for leg in cand.legs].count(Side.SELL) == 1
    assert [leg.side for leg in cand.legs].count(Side.BUY) == 1
    assert cand.max_loss is not None
