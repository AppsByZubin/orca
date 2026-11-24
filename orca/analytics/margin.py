"""Margin estimation.

This is an approximation only. Real SPAN+exposure depends on the exchange risk
arrays of the day, so before trading real size replace this with the broker's
margin API:

    Kite:   POST https://api.kite.trade/margins/basket      (basket margins)
    Upstox: POST /v2/charges/margin                         (margin for a list of legs)

`OrderGateway.basket_margin()` is the hook for that - the Kite gateway wires it
up and falls back to this estimate when no credentials are configured.
"""

from __future__ import annotations

from collections.abc import Sequence

from orca.domain import LegSpec, OptionType, Side

# Rough SPAN+exposure as a fraction of contract notional for index options.
SHORT_OPTION_NOTIONAL_PCT = 0.11
HEDGE_BENEFIT = 0.70  # fraction of margin released by a protective long leg


def estimate_margin(legs: Sequence[LegSpec], spot: float) -> float:
    """Blocked margin estimate in rupees for a multi-leg position."""
    if not legs:
        return 0.0

    short_margin = 0.0
    premium_paid = 0.0
    hedged_pairs = 0

    shorts = [lg for lg in legs if lg.side is Side.SELL]
    longs = [lg for lg in legs if lg.side is Side.BUY]

    for leg in shorts:
        notional = spot * leg.quantity
        short_margin += notional * SHORT_OPTION_NOTIONAL_PCT
        premium_received = leg.ref_price * leg.quantity
        short_margin -= premium_received * 0.5  # premium partly offsets requirement

    for leg in longs:
        premium_paid += leg.ref_price * leg.quantity
        same_type_short = next((s for s in shorts if s.option_type is leg.option_type), None)
        if same_type_short is not None:
            hedged_pairs += 1

    if hedged_pairs:
        covered = min(hedged_pairs / max(len(shorts), 1), 1.0)
        short_margin *= 1 - HEDGE_BENEFIT * covered

    return round(max(short_margin, 0.0) + premium_paid, 2)


def spread_max_loss(legs: Sequence[LegSpec]) -> float | None:
    """Defined max loss for vertical/condor structures, else None."""
    by_type: dict[OptionType, list[LegSpec]] = {}
    for leg in legs:
        by_type.setdefault(leg.option_type, []).append(leg)

    total = 0.0
    for _option_type, group in by_type.items():
        shorts = [lg for lg in group if lg.side is Side.SELL]
        longs = [lg for lg in group if lg.side is Side.BUY]
        if not shorts:
            continue
        if not longs:
            return None  # naked short -> undefined
        width = abs(longs[0].strike - shorts[0].strike)
        qty = min(shorts[0].quantity, longs[0].quantity)
        credit = (shorts[0].ref_price - longs[0].ref_price) * qty
        total += width * qty - credit
    return round(total, 2) if total else None
