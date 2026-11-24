"""Features, labels and the training pipeline."""

from __future__ import annotations

from orca.ml.dataset import LABEL_COL, synthetic_frame
from orca.ml.features import FEATURE_NAMES, candidate_features, market_context
from orca.ml.heuristic import score as heuristic_score
from orca.ml.labels import simulate_outcome
from orca.ml.train import train_model
from orca.strategies import get_template


def test_feature_row_is_complete(chain, nifty) -> None:
    cand = get_template("short_strangle")(
        chain, nifty, {"short_delta": [0.15], "_defaults": {}}, 1
    )[0]
    ctx = market_context(chain)
    row = candidate_features(cand, chain, ctx)
    assert set(row) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in row.values())
    assert row["n_legs"] == 2
    assert row["net_theta"] > 0  # short premium collects theta


def test_heuristic_scores_in_unit_interval(chain, nifty) -> None:
    cand = get_template("short_strangle")(
        chain, nifty, {"short_delta": [0.15], "_defaults": {}}, 1
    )[0]
    value, reasons = heuristic_score(candidate_features(cand, chain))
    assert 0.0 <= value <= 1.0
    assert isinstance(reasons, list)


def test_outcome_labelling_detects_profit(chain, nifty, feed) -> None:
    from datetime import timedelta

    cand = get_template("short_strangle")(
        chain, nifty, {"short_delta": [0.15], "_defaults": {}}, 1
    )[0]
    # Decay the forward chain by shrinking time to expiry: premium falls, shorts profit.
    forward = [feed.get_option_chain(nifty, chain.expiry) for _ in range(3)]
    for i, snap in enumerate(forward):
        snap.ts = chain.ts + timedelta(minutes=5 * (i + 1))
        for option in snap.options:
            option.ltp *= 0.3
    outcome = simulate_outcome(cand.legs, forward, target_pct=0.5, stop_multiple=2.0)
    assert outcome.hit_target is True
    assert outcome.label == 1
    assert outcome.final_pnl > 0


def test_train_on_synthetic_frame_beats_coin_flip() -> None:
    frame = synthetic_frame(1500, seed=3)
    assert LABEL_COL in frame.columns
    model, metrics = train_model(frame, name="test", synthetic=True, save=False)
    assert model.is_fitted
    assert metrics["roc_auc"] is not None and metrics["roc_auc"] > 0.6
    prob, expected = model.score({name: 0.0 for name in FEATURE_NAMES})
    assert prob is not None and 0.0 <= prob <= 1.0
    assert expected is not None
