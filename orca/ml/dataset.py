"""Turn captured snapshots into a training table.

Real path:      build_training_frame(session)  - replays stored chain snapshots,
                rebuilds the candidates that existed at each point in time, and
                labels them from what happened next.
Bootstrap path: synthetic_frame(n)             - plausible random regimes with a
                known noisy relationship, so the pipeline is testable on day 1.
                Models trained this way are marked synthetic=True; never trade
                off them.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from orca.config import get_strategy_config, get_universe
from orca.domain import ChainSnapshot, OptionQuote, OptionType
from orca.logging_setup import get_logger
from orca.ml.features import FEATURE_NAMES, candidate_features, market_context
from orca.ml.labels import simulate_outcome
from orca.ml.sources import daily_atm_iv_from_snapshots, daily_closes_from_snapshots
from orca.models import ChainSnapshot as ChainSnapshotRow
from orca.strategies import get_template

log = get_logger(__name__)

LABEL_COL = "label"
RETURN_COL = "return_on_credit"


def _row_to_domain(row: ChainSnapshotRow) -> ChainSnapshot:
    return ChainSnapshot(
        underlying=row.underlying,
        expiry=row.expiry,
        spot=row.spot,
        ts=row.ts,
        futures_price=row.futures_price,
        source=row.source,
        options=[
            OptionQuote(
                underlying=row.underlying,
                expiry=row.expiry,
                strike=q.strike,
                option_type=OptionType(q.option_type),
                instrument_key=q.instrument_key,
                tradingsymbol=q.tradingsymbol,
                ltp=q.ltp,
                bid=q.bid,
                ask=q.ask,
                volume=q.volume,
                oi=q.oi,
                oi_change=q.oi_change,
                iv=q.iv,
                delta=q.delta,
                gamma=q.gamma,
                theta=q.theta,
                vega=q.vega,
            )
            for q in row.quotes
        ],
    )


def load_snapshots(
    session: Session, underlying: str | None = None, since: date | None = None
) -> list[ChainSnapshot]:
    stmt = select(ChainSnapshotRow).order_by(ChainSnapshotRow.ts)
    if underlying:
        stmt = stmt.where(ChainSnapshotRow.underlying == underlying)
    if since:
        stmt = stmt.where(ChainSnapshotRow.ts >= since)
    return [_row_to_domain(r) for r in session.scalars(stmt).all()]


def frame_from_snapshots(
    snapshots: Iterable[ChainSnapshot],
    target_pct: float = 0.5,
    stop_multiple: float = 2.0,
    sample_every: int = 5,
    daily_closes: dict[str, list[tuple[date, float]]] | None = None,
    iv_history: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """One row per (snapshot, candidate), labelled by the forward path.

    Source-agnostic: the snapshots may come from the local `chain_snapshots`
    table, from Parquet in S3, or from anywhere else that can produce domain
    objects.

    `daily_closes` and `iv_history` feed the regime block. If omitted they are
    derived from the snapshots, which matters: passing neither leaves
    realized_vol_20, iv_rv_spread, trend_5d, trend_20d, spot_vs_ema20 and
    gap_pct at zero in every training row while the live scanner computes them
    for real - a train/serve skew that silently wastes six features.
    """
    snapshots = list(snapshots)
    if not snapshots:
        return pd.DataFrame(columns=[*FEATURE_NAMES, LABEL_COL, RETURN_COL])

    if daily_closes is None:
        daily_closes = daily_closes_from_snapshots(snapshots)
    if iv_history is None:
        iv_history = daily_atm_iv_from_snapshots(snapshots)

    groups: dict[tuple[str, date], list[ChainSnapshot]] = defaultdict(list)
    for snap in snapshots:
        groups[(snap.underlying, snap.expiry)].append(snap)

    strategy_cfg = get_strategy_config()
    universe = get_universe()
    records: list[dict] = []

    for (symbol, _expiry), series in groups.items():
        try:
            u = universe.get(symbol)
        except KeyError:
            continue
        series.sort(key=lambda s: s.ts)
        last_spot = series[-1].spot
        closes = daily_closes.get(symbol, [])
        ivs = iv_history.get(symbol, [])
        for i in range(0, len(series) - 1, max(sample_every, 1)):
            chain = series[i]
            forward = series[i + 1 :]
            if not forward:
                continue
            # Only history up to this snapshot: never let the context see the future.
            as_of = chain.ts.date()
            ctx = market_context(
                chain,
                daily_closes=[(d, c) for d, c in closes if d <= as_of],
                iv_history=ivs[: sum(1 for d, _c in closes if d <= as_of)] or None,
                now=chain.ts,
            )
            for spec in strategy_cfg.enabled:
                params = {**spec.params, "_defaults": strategy_cfg.defaults}
                try:
                    candidates = get_template(spec.template)(chain, u, params, 1)
                except KeyError:
                    continue
                for cand in candidates:
                    outcome = simulate_outcome(
                        cand.legs, forward, target_pct, stop_multiple, expiry_spot=last_spot
                    )
                    if outcome.bars == 0:
                        continue
                    row = candidate_features(cand, chain, ctx)
                    row[LABEL_COL] = outcome.label
                    row[RETURN_COL] = outcome.return_on_credit
                    row["ts"] = chain.ts
                    row["underlying"] = symbol
                    row["strategy_id"] = cand.strategy_id
                    records.append(row)

    df = pd.DataFrame.from_records(records)
    log.info("training frame: %d rows", len(df))
    return df


def build_training_frame(
    session: Session,
    underlying: str | None = None,
    since: date | None = None,
    target_pct: float = 0.5,
    stop_multiple: float = 2.0,
    sample_every: int = 5,
) -> pd.DataFrame:
    """Training frame from the local chain_snapshots table."""
    return frame_from_snapshots(
        load_snapshots(session, underlying, since),
        target_pct=target_pct,
        stop_multiple=stop_multiple,
        sample_every=sample_every,
    )


def build_training_frame_from_parquet(
    source: str,
    underlying: str | None = None,
    since: date | None = None,
    until: date | None = None,
    target_pct: float = 0.5,
    stop_multiple: float = 2.0,
    sample_every: int = 1,
    columns: dict[str, str] | None = None,
    storage_options: dict | None = None,
) -> pd.DataFrame:
    """Training frame from Parquet (local path, directory tree or s3:// prefix).

    `sample_every` defaults to 1 here rather than 5: Parquet archives are often
    coarser than ORCA's own 60-second capture, so subsampling again would throw
    away most entry points.
    """
    from orca.ml.sources import load_parquet_snapshots

    return frame_from_snapshots(
        load_parquet_snapshots(
            source,
            underlying=underlying,
            since=since,
            until=until,
            columns=columns,
            storage_options=storage_options,
        ),
        target_pct=target_pct,
        stop_multiple=stop_multiple,
        sample_every=sample_every,
    )


def synthetic_frame(n: int = 4000, seed: int = 11) -> pd.DataFrame:
    """Bootstrap data with a deliberately imperfect signal (~75% separable)."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    rows = []
    for _ in range(n):
        iv_rank = float(rng.beta(2, 2))
        iv_rv = float(rng.normal(0.01, 0.03))
        dte = float(rng.integers(0, 8))
        short_delta = float(rng.uniform(0.05, 0.35))
        pop = 1 - 2 * short_delta + float(rng.normal(0, 0.05))
        credit_to_margin = float(rng.uniform(0.01, 0.12))
        theta_per_vega = float(rng.normal(0.4, 0.25))
        net_delta = float(rng.normal(0, 25))
        be_sigma = float(rng.uniform(0.4, 2.5))
        trend = float(rng.normal(0, 0.02))

        # Latent "truth": rich vol, far strikes, enough time, not directional.
        logit = (
            2.2 * (iv_rank - 0.45)
            + 14 * iv_rv
            + 1.1 * (be_sigma - 1.0)
            - 3.0 * (short_delta - 0.15)
            - 0.05 * abs(net_delta) / 5
            - 0.18 * max(0.0, 1 - dte)
            + 0.7 * theta_per_vega
            - 9 * abs(trend)
            + float(rng.normal(0, 0.9))
        )
        prob = 1 / (1 + np.exp(-logit))
        label = int(rng.random() < prob)
        ret = (0.45 if label else -0.85) + float(rng.normal(0, 0.25))

        rows.append(
            {
                "atm_iv": 0.10 + iv_rank * 0.12,
                "iv_rank": iv_rank,
                "iv_percentile": min(1.0, max(0.0, iv_rank + float(rng.normal(0, 0.08)))),
                "realized_vol_20": max(0.02, 0.11 + iv_rank * 0.08 - iv_rv),
                "iv_rv_spread": iv_rv,
                "iv_skew": float(rng.normal(0.01, 0.02)),
                "pcr_oi": float(rng.uniform(0.6, 1.5)),
                "trend_5d": trend,
                "trend_20d": float(rng.normal(0, 0.04)),
                "spot_vs_ema20": float(rng.normal(0, 0.015)),
                "gap_pct": float(rng.normal(0, 0.004)),
                "dte": dte,
                "minutes_to_close": float(rng.integers(5, 375)),
                "day_of_week": float(rng.integers(0, 5)),
                "n_legs": float(random.choice([2, 2, 4])),
                "is_defined_risk": float(random.choice([0, 1])),
                "short_delta_avg": short_delta,
                "net_delta": net_delta,
                "net_theta": abs(float(rng.normal(900, 350))),
                "net_vega": -abs(float(rng.normal(1500, 500))),
                "theta_per_vega": theta_per_vega,
                "credit_pct_of_spot": float(rng.uniform(0.001, 0.01)),
                "credit_to_max_loss": float(rng.uniform(0.05, 0.6)),
                "credit_to_margin": credit_to_margin,
                "breakeven_width_pct": float(rng.uniform(0.01, 0.06)),
                "nearest_be_sigma": be_sigma,
                "prob_of_profit": min(0.99, max(0.3, pop)),
                "short_strike_distance_pct": float(rng.uniform(0.005, 0.05)),
                LABEL_COL: label,
                RETURN_COL: ret,
            }
        )
    df = pd.DataFrame.from_records(rows)
    return df[[*FEATURE_NAMES, LABEL_COL, RETURN_COL]]
