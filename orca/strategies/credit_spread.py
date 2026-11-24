"""Vertical credit spread: sell a strike, buy one further out, same expiry.

Directional premium selling - a put spread is mildly bullish, a call spread
mildly bearish. The ML score is what decides whether the direction is worth it.
"""

from __future__ import annotations

from orca.config import Underlying
from orca.domain import Candidate, ChainSnapshot, OptionType, Side
from orca.strategies.base import build_candidate, make_leg, passes_liquidity, template


@template("credit_spread")
def build(
    chain: ChainSnapshot, underlying: Underlying, params: dict, lots: int = 1
) -> list[Candidate]:
    sides = [OptionType(s) for s in (params.get("side") or ["PE"])]
    deltas = params.get("short_delta") or [0.25]
    widths = params.get("wing_width_strikes") or [4]
    defaults = params.get("_defaults", {})
    step = underlying.strike_step
    out: list[Candidate] = []

    for option_type in sides:
        for target in deltas:
            short = chain.nearest_by_delta(option_type, target)
            if not short:
                continue
            for width in widths:
                offset = width * step * (1 if option_type is OptionType.CE else -1)
                long = chain.get(short.strike + offset, option_type)
                if not long:
                    continue
                ok, _why = passes_liquidity([short], defaults)
                if not ok:
                    continue
                legs = [
                    make_leg(short, Side.SELL, underlying, lots),
                    make_leg(long, Side.BUY, underlying, lots),
                ]
                kind = "call" if option_type is OptionType.CE else "put"
                out.append(
                    build_candidate(
                        strategy_id=f"{kind}_credit_spread_{int(target * 100)}d_{width}w",
                        template_name="credit_spread",
                        chain=chain,
                        underlying=underlying,
                        legs=legs,
                        reasons=[
                            f"{kind} credit spread at {target:.0%} delta",
                            f"short {int(short.strike)} / long {int(long.strike)}",
                        ],
                    )
                )
    return out
