"""Outcome labelling: what actually happened after we would have entered.

The model answers one question: *if I sell this structure now, do I reach my
profit target before my stop?* To label that from history we replay the stored
chain snapshots forward and mark the position on every tick.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from orca.analytics.payoff import pnl_at
from orca.domain import ChainSnapshot, LegSpec


@dataclass
class Outcome:
    label: int  # 1 = target reached (or expired profitable) before the stop
    final_pnl: float
    max_favourable: float
    max_adverse: float
    hit_target: bool
    hit_stop: bool
    bars: int
    return_on_credit: float  # final_pnl / credit received


def position_mtm(legs: Sequence[LegSpec], chain: ChainSnapshot) -> float | None:
    """Mark a leg-set against a later snapshot. None if any leg is missing."""
    total = 0.0
    for leg in legs:
        quote = chain.get(leg.strike, leg.option_type)
        if quote is None or quote.ltp <= 0:
            return None
        total += (quote.ltp - leg.ref_price) * leg.side.sign * leg.quantity
    return total


def simulate_outcome(
    legs: Sequence[LegSpec],
    forward: Sequence[ChainSnapshot],
    target_pct: float = 0.5,
    stop_multiple: float = 2.0,
    expiry_spot: float | None = None,
) -> Outcome:
    """Replay `forward` snapshots (chronological) and classify the result."""
    credit = sum(-leg.side.sign * leg.ref_price * leg.quantity for leg in legs)
    credit = max(credit, 1e-6)
    target = credit * target_pct
    stop = -credit * stop_multiple

    mtm = 0.0
    best, worst = 0.0, 0.0
    hit_target = hit_stop = False
    bars = 0

    for chain in forward:
        marked = position_mtm(legs, chain)
        if marked is None:
            continue
        bars += 1
        mtm = marked
        best, worst = max(best, mtm), min(worst, mtm)
        if mtm >= target:
            hit_target = True
            break
        if mtm <= stop:
            hit_stop = True
            break

    if not hit_target and not hit_stop and expiry_spot is not None:
        mtm = pnl_at(
            expiry_spot,
            [(leg.strike, leg.option_type, leg.side, leg.quantity, leg.ref_price) for leg in legs],
        )
        best, worst = max(best, mtm), min(worst, mtm)

    label = 1 if hit_target or (not hit_stop and mtm > 0) else 0
    return Outcome(
        label=label,
        final_pnl=round(mtm, 2),
        max_favourable=round(best, 2),
        max_adverse=round(worst, 2),
        hit_target=hit_target,
        hit_stop=hit_stop,
        bars=bars,
        return_on_credit=round(mtm / credit, 4),
    )
