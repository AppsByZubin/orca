"""Short strangle: sell an OTM call and an OTM put at a target delta.

The bread-and-butter premium-selling trade on index weeklies. Undefined risk,
so the tracker's stop-loss is what keeps it survivable.
"""

from __future__ import annotations

from orca.config import Underlying
from orca.domain import Candidate, ChainSnapshot, OptionType, Side
from orca.strategies.base import (
    build_candidate,
    make_leg,
    passes_liquidity,
    template,
)


@template("short_strangle")
def build(
    chain: ChainSnapshot, underlying: Underlying, params: dict, lots: int = 1
) -> list[Candidate]:
    deltas = params.get("short_delta") or [0.15]
    defaults = params.get("_defaults", {})
    out: list[Candidate] = []

    for target in deltas:
        call = chain.nearest_by_delta(OptionType.CE, target)
        put = chain.nearest_by_delta(OptionType.PE, target)
        if not call or not put or call.strike <= put.strike:
            continue
        ok, _why = passes_liquidity([call, put], defaults)
        if not ok:
            continue
        legs = [
            make_leg(call, Side.SELL, underlying, lots),
            make_leg(put, Side.SELL, underlying, lots),
        ]
        out.append(
            build_candidate(
                strategy_id=f"short_strangle_{int(target * 100)}d",
                template_name="short_strangle",
                chain=chain,
                underlying=underlying,
                legs=legs,
                reasons=[
                    f"short {target:.0%} delta strangle",
                    f"{int(put.strike)}PE / {int(call.strike)}CE",
                ],
            )
        )
    return out
