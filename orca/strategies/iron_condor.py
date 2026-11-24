"""Iron condor: short strangle with long wings n strikes further out.

Defined risk, so margin is far lower and the ML score can be compared against
a known max loss.
"""

from __future__ import annotations

from orca.config import Underlying
from orca.domain import Candidate, ChainSnapshot, OptionType, Side
from orca.strategies.base import build_candidate, make_leg, passes_liquidity, template


@template("iron_condor")
def build(
    chain: ChainSnapshot, underlying: Underlying, params: dict, lots: int = 1
) -> list[Candidate]:
    deltas = params.get("short_delta") or [0.15]
    widths = params.get("wing_width_strikes") or [4]
    defaults = params.get("_defaults", {})
    step = underlying.strike_step
    out: list[Candidate] = []

    for target in deltas:
        short_call = chain.nearest_by_delta(OptionType.CE, target)
        short_put = chain.nearest_by_delta(OptionType.PE, target)
        if not short_call or not short_put:
            continue
        for width in widths:
            long_call = chain.get(short_call.strike + width * step, OptionType.CE)
            long_put = chain.get(short_put.strike - width * step, OptionType.PE)
            if not long_call or not long_put:
                continue
            ok, _why = passes_liquidity([short_call, short_put], defaults)
            if not ok:
                continue
            legs = [
                make_leg(short_call, Side.SELL, underlying, lots),
                make_leg(long_call, Side.BUY, underlying, lots),
                make_leg(short_put, Side.SELL, underlying, lots),
                make_leg(long_put, Side.BUY, underlying, lots),
            ]
            out.append(
                build_candidate(
                    strategy_id=f"iron_condor_{int(target * 100)}d_{width}w",
                    template_name="iron_condor",
                    chain=chain,
                    underlying=underlying,
                    legs=legs,
                    reasons=[
                        f"{target:.0%} delta condor, {width}-strike wings",
                        f"{int(long_put.strike)}/{int(short_put.strike)}"
                        f" - {int(short_call.strike)}/{int(long_call.strike)}",
                    ],
                )
            )
    return out
