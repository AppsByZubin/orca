"""Feature engineering - the part that decides whether the ML is any good.

Two levels:

  `market_context(...)`   once per chain snapshot: the regime (IV rank, realized
                          vol, trend, PCR, time of day).
  `candidate_features(...)` per candidate: how far the strikes are, how much
                          credit relative to risk, greeks of the structure.

FEATURE_NAMES fixes the column order so a saved model keeps working.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from zoneinfo import ZoneInfo

from orca.analytics.greeks import realized_vol, year_fraction
from orca.domain import Candidate, ChainSnapshot, OptionType, Side

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN_MIN = 9 * 60 + 15
MARKET_CLOSE_MIN = 15 * 60 + 30

FEATURE_NAMES: list[str] = [
    # regime
    "atm_iv",
    "iv_rank",
    "iv_percentile",
    "realized_vol_20",
    "iv_rv_spread",
    "iv_skew",
    "pcr_oi",
    "trend_5d",
    "trend_20d",
    "spot_vs_ema20",
    "gap_pct",
    # timing
    "dte",
    "minutes_to_close",
    "day_of_week",
    # structure
    "n_legs",
    "is_defined_risk",
    "short_delta_avg",
    "net_delta",
    "net_theta",
    "net_vega",
    "theta_per_vega",
    "credit_pct_of_spot",
    "credit_to_max_loss",
    "credit_to_margin",
    "breakeven_width_pct",
    "nearest_be_sigma",
    "prob_of_profit",
    "short_strike_distance_pct",
]


def _safe(x: float | None, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        value = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) or math.isinf(value) else value


def _pct_rank(value: float, history: list[float]) -> float | None:
    if not history:
        return None
    below = sum(1 for h in history if h <= value)
    return below / len(history)


def _rank_within_range(value: float, history: list[float]) -> float | None:
    """Classic IV rank: position between the period low and high."""
    if not history:
        return None
    lo, hi = min(history), max(history)
    if hi - lo < 1e-9:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def market_context(
    chain: ChainSnapshot,
    daily_closes: list[tuple[date, float]] | None = None,
    iv_history: list[float] | None = None,
    now: datetime | None = None,
) -> dict[str, float]:
    """Regime features shared by every candidate on this snapshot."""
    now = now or chain.ts
    closes = [c for _d, c in (daily_closes or [])]
    atm_iv = _safe(chain.atm_iv(), 0.0)

    rv = realized_vol(closes, window=20) if len(closes) > 21 else None
    ctx: dict[str, float] = {
        "atm_iv": atm_iv,
        "iv_rank": _safe(_rank_within_range(atm_iv, iv_history or []), 0.5),
        "iv_percentile": _safe(_pct_rank(atm_iv, iv_history or []), 0.5),
        "realized_vol_20": _safe(rv),
        "iv_rv_spread": _safe(atm_iv - rv) if rv else 0.0,
        "iv_skew": _iv_skew(chain),
        "pcr_oi": _pcr(chain),
        "trend_5d": _trend(closes, 5),
        "trend_20d": _trend(closes, 20),
        "spot_vs_ema20": _spot_vs_ema(chain.spot, closes, 20),
        "gap_pct": (chain.spot / closes[-1] - 1) if closes else 0.0,
        "dte": float(chain.dte(now.date())),
        "minutes_to_close": _minutes_to_close(now),
        "day_of_week": float(now.astimezone(IST).weekday()),
    }
    return ctx


def _iv_skew(chain: ChainSnapshot) -> float:
    """Put IV minus call IV at ~15 delta; positive = puts bid up (fear)."""
    put = chain.nearest_by_delta(OptionType.PE, 0.15)
    call = chain.nearest_by_delta(OptionType.CE, 0.15)
    if not put or not call or put.iv is None or call.iv is None:
        return 0.0
    return _safe(put.iv - call.iv)


def _pcr(chain: ChainSnapshot) -> float:
    calls = sum(o.oi for o in chain.side(OptionType.CE))
    puts = sum(o.oi for o in chain.side(OptionType.PE))
    return puts / calls if calls else 0.0


def _trend(closes: list[float], window: int) -> float:
    if len(closes) <= window:
        return 0.0
    return closes[-1] / closes[-1 - window] - 1


def _spot_vs_ema(spot: float, closes: list[float], window: int) -> float:
    if len(closes) < window:
        return 0.0
    k = 2 / (window + 1)
    ema = closes[-window]
    for c in closes[-window + 1 :]:
        ema = c * k + ema * (1 - k)
    return spot / ema - 1 if ema else 0.0


def _minutes_to_close(now: datetime) -> float:
    local = now.astimezone(IST)
    minutes = local.hour * 60 + local.minute
    return float(max(0, min(MARKET_CLOSE_MIN - minutes, MARKET_CLOSE_MIN - MARKET_OPEN_MIN)))


def candidate_features(
    candidate: Candidate, chain: ChainSnapshot, context: dict[str, float] | None = None
) -> dict[str, float]:
    """Full feature row for one candidate: regime + structure."""
    ctx = dict(context or market_context(chain))
    spot = chain.spot or candidate.spot or 1.0
    t = year_fraction(candidate.expiry, chain.ts)
    atm_iv = ctx.get("atm_iv") or 0.0
    sigma_move = spot * atm_iv * math.sqrt(max(t, 1e-6))

    net_delta = net_theta = net_vega = 0.0
    short_deltas: list[float] = []
    short_distances: list[float] = []

    for leg in candidate.legs:
        quote = chain.get(leg.strike, leg.option_type)
        if quote is None:
            continue
        sign = leg.side.sign * leg.quantity
        net_delta += _safe(quote.delta) * sign
        net_theta += _safe(quote.theta) * sign
        net_vega += _safe(quote.vega) * sign
        if leg.side is Side.SELL:
            short_deltas.append(quote.abs_delta)
            short_distances.append(abs(leg.strike - spot) / spot)

    be = candidate.breakevens
    be_width = (max(be) - min(be)) / spot if len(be) >= 2 else 0.0
    nearest_be = min((abs(b - spot) for b in be), default=0.0)

    row = {
        **{name: _safe(ctx.get(name)) for name in FEATURE_NAMES if name in ctx},
        "n_legs": float(len(candidate.legs)),
        "is_defined_risk": 1.0 if candidate.max_loss is not None else 0.0,
        "short_delta_avg": sum(short_deltas) / len(short_deltas) if short_deltas else 0.0,
        "net_delta": net_delta,
        "net_theta": net_theta,
        "net_vega": net_vega,
        "theta_per_vega": net_theta / abs(net_vega) if abs(net_vega) > 1e-6 else 0.0,
        "credit_pct_of_spot": candidate.net_credit / (spot * max(candidate.legs[0].quantity, 1))
        if candidate.legs
        else 0.0,
        "credit_to_max_loss": _safe(candidate.credit_to_risk),
        "credit_to_margin": candidate.net_credit / candidate.margin_estimate
        if candidate.margin_estimate
        else 0.0,
        "breakeven_width_pct": be_width,
        "nearest_be_sigma": nearest_be / sigma_move if sigma_move > 0 else 0.0,
        "prob_of_profit": _safe(candidate.prob_of_profit, 0.5),
        "short_strike_distance_pct": sum(short_distances) / len(short_distances)
        if short_distances
        else 0.0,
    }
    return {name: _safe(row.get(name)) for name in FEATURE_NAMES}


def to_vector(features: dict[str, float]) -> list[float]:
    """Feature dict -> model input, in FEATURE_NAMES order."""
    return [_safe(features.get(name)) for name in FEATURE_NAMES]
