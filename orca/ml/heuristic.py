"""Rule-based fallback score, used until a trained model exists.

It is intentionally simple and transparent: a weighted blend of the things an
options seller actually cares about. It also gives the ML model something to
beat - if a trained model cannot outperform this, do not trade it.
"""

from __future__ import annotations

WEIGHTS = {
    "iv_rank": 0.25,  # selling expensive vol
    "iv_rv_spread": 0.15,  # implied above realized
    "prob_of_profit": 0.25,
    "credit_to_margin": 0.15,
    "theta_per_vega": 0.10,
    "delta_neutrality": 0.10,
}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score(features: dict[str, float]) -> tuple[float, list[str]]:
    """Return (score in 0..1, human-readable reasons)."""
    f = features
    reasons: list[str] = []

    iv_rank = _clip(f.get("iv_rank", 0.5))
    if iv_rank > 0.6:
        reasons.append(f"IV rank {iv_rank:.0%} - premium is rich")
    elif iv_rank < 0.25:
        reasons.append(f"IV rank {iv_rank:.0%} - thin premium")

    spread = f.get("iv_rv_spread", 0.0)
    spread_n = _clip(0.5 + spread * 8)
    if spread > 0.02:
        reasons.append(f"IV {spread:.1%} above realized vol")

    pop = _clip(f.get("prob_of_profit", 0.5))
    ctm = _clip(f.get("credit_to_margin", 0.0) * 12)
    tpv = _clip(f.get("theta_per_vega", 0.0) * 2 + 0.5)

    net_delta = abs(f.get("net_delta", 0.0))
    neutrality = _clip(1 - net_delta / 50.0)
    if net_delta > 40:
        reasons.append(f"net delta {f.get('net_delta', 0):.0f} - directional")

    dte = f.get("dte", 5)
    total = (
        WEIGHTS["iv_rank"] * iv_rank
        + WEIGHTS["iv_rv_spread"] * spread_n
        + WEIGHTS["prob_of_profit"] * pop
        + WEIGHTS["credit_to_margin"] * ctm
        + WEIGHTS["theta_per_vega"] * tpv
        + WEIGHTS["delta_neutrality"] * neutrality
    )

    # Expiry-day gamma is brutal for sellers; haircut the score.
    if dte <= 0:
        total *= 0.8
        reasons.append("expiry day - gamma risk")

    return round(_clip(total), 4), reasons
