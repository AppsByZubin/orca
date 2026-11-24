"""The scanner: chain -> candidates -> features -> ML score -> ranked list.

This is the "which option-selling trade should I take" engine. Candidates come
from config-driven templates; the ranking comes from the model when one is
trained, and from the transparent heuristic until then.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from orca.config import Underlying, get_strategy_config, get_universe
from orca.domain import Candidate, ChainSnapshot
from orca.logging_setup import get_logger
from orca.ml.features import candidate_features
from orca.ml.heuristic import score as heuristic_score
from orca.ml.model import ScoringModel, get_model
from orca.models import Signal
from orca.services.market_data import MarketDataService
from orca.strategies import get_template
from orca.strategies.base import meets_credit_floor

log = get_logger(__name__)


def leg_to_dict(leg) -> dict:  # noqa: ANN001 - LegSpec
    d = asdict(leg)
    d["expiry"] = leg.expiry.isoformat()
    d["option_type"] = leg.option_type.value
    d["side"] = leg.side.value
    d["quantity"] = leg.quantity
    return d


class Scanner:
    def __init__(
        self,
        market: MarketDataService | None = None,
        model: ScoringModel | None = None,
    ) -> None:
        self.market = market or MarketDataService()
        self.model = model or get_model()

    # -- generation ------------------------------------------------------
    def candidates_for(
        self, session: Session, underlying: Underlying, chain: ChainSnapshot, lots: int = 1
    ) -> list[Candidate]:
        cfg = get_strategy_config()
        context = self.market.context(session, underlying, chain)
        out: list[Candidate] = []

        for spec in cfg.enabled:
            params = {**spec.params, "_defaults": cfg.defaults}
            try:
                builder = get_template(spec.template)
            except KeyError as exc:
                log.warning("%s", exc)
                continue
            for cand in builder(chain, underlying, params, lots):
                if not meets_credit_floor(cand, cfg.defaults, lots):
                    continue
                cand.features = candidate_features(cand, chain, context)
                out.append(cand)

        self.score(out)
        out.sort(key=lambda c: c.score or 0.0, reverse=True)
        return out

    # -- scoring ---------------------------------------------------------
    def score(self, candidates: list[Candidate]) -> None:
        """Attach score + expected_return in place."""
        if not candidates:
            return
        if self.model.is_fitted:
            results = self.model.score_many([c.features for c in candidates])
            source = f"model:{self.model.metadata.name}"
            if self.model.metadata.synthetic:
                source += " (synthetic)"
            for cand, (prob, expected) in zip(candidates, results, strict=False):
                cand.score = None if prob is None else round(prob, 4)
                cand.expected_return = None if expected is None else round(expected, 4)
                cand.score_source = source
                _, reasons = heuristic_score(cand.features)
                cand.reasons = [*cand.reasons, *reasons]
        else:
            for cand in candidates:
                value, reasons = heuristic_score(cand.features)
                cand.score = value
                cand.score_source = "heuristic"
                cand.reasons = [*cand.reasons, *reasons]

    # -- full scan -------------------------------------------------------
    def scan(
        self,
        session: Session,
        symbols: list[str] | None = None,
        lots: int = 1,
        persist: bool = True,
        max_dte: int | None = None,
    ) -> list[Candidate]:
        cfg = get_strategy_config()
        max_dte = max_dte if max_dte is not None else int(cfg.defaults.get("max_dte", 10))
        min_dte = int(cfg.defaults.get("min_dte", 0))
        universe = get_universe()
        targets = [universe.get(s) for s in symbols] if symbols else universe.enabled

        all_candidates: list[Candidate] = []
        for underlying in targets:
            expiry = self.market.nearest_expiry(underlying, min_dte=min_dte, max_dte=max_dte)
            if expiry is None:
                continue
            try:
                chain = self.market.chain(underlying, expiry)
            except Exception as exc:  # noqa: BLE001
                log.warning("scan skipped %s: %s", underlying.symbol, exc)
                continue
            candidates = self.candidates_for(session, underlying, chain, lots)
            log.info(
                "%s %s: %d candidates (best score %.3f)",
                underlying.symbol,
                expiry,
                len(candidates),
                candidates[0].score or 0.0 if candidates else 0.0,
            )
            all_candidates.extend(candidates)

        all_candidates.sort(key=lambda c: c.score or 0.0, reverse=True)
        if persist:
            self.persist(session, all_candidates)
        return all_candidates

    def persist(self, session: Session, candidates: list[Candidate]) -> list[Signal]:
        rows = [
            Signal(
                ts=c.ts or datetime.now(UTC),
                underlying=c.underlying,
                expiry=c.expiry,
                strategy_id=c.strategy_id,
                template=c.template,
                spot=c.spot,
                score=c.score,
                expected_return=c.expected_return,
                score_source=c.score_source,
                net_credit=c.net_credit,
                max_loss=c.max_loss,
                margin_estimate=c.margin_estimate,
                prob_of_profit=c.prob_of_profit,
                legs=[leg_to_dict(leg) for leg in c.legs],
                breakevens=c.breakevens,
                features=c.features,
                reasons=c.reasons,
            )
            for c in candidates
        ]
        session.add_all(rows)
        session.flush()
        for candidate, row in zip(candidates, rows, strict=False):
            candidate.signal_id = row.id
        return rows


def recent_signals(
    session: Session, limit: int = 50, underlying: str | None = None
) -> list[Signal]:
    stmt = select(Signal).order_by(Signal.ts.desc(), Signal.score.desc()).limit(limit)
    if underlying:
        stmt = stmt.where(Signal.underlying == underlying)
    return list(session.scalars(stmt).all())
