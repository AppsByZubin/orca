"""Training entrypoint: frame in, saved model + metrics out.

Split is chronological where a timestamp exists - options data is a time
series and a random split leaks the future into the training set.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)

from orca import __version__
from orca.logging_setup import get_logger
from orca.ml.dataset import LABEL_COL, RETURN_COL
from orca.ml.features import FEATURE_NAMES
from orca.ml.model import ModelMetadata, ScoringModel, new_classifier, new_regressor

log = get_logger(__name__)


def _split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "ts" in df.columns:
        df = df.sort_values("ts")
    cut = int(len(df) * (1 - test_size))
    cut = max(min(cut, len(df) - 1), 1)
    return df.iloc[:cut], df.iloc[cut:]


def train_model(
    df: pd.DataFrame,
    name: str = "default",
    test_size: float = 0.25,
    target_pct: float = 0.5,
    stop_multiple: float = 2.0,
    synthetic: bool = False,
    save: bool = True,
) -> tuple[ScoringModel, dict]:
    if df.empty:
        raise ValueError("empty training frame - capture snapshots first, or use --synthetic")
    if LABEL_COL not in df.columns:
        raise ValueError(f"frame is missing the {LABEL_COL!r} column")

    df = df.dropna(subset=[LABEL_COL])
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    for col in missing:
        df[col] = 0.0
    if missing:
        log.warning("filled %d missing feature columns with 0: %s", len(missing), missing)

    train_df, test_df = _split(df, test_size)
    x_train = train_df[FEATURE_NAMES].to_numpy(dtype=float)
    y_train = train_df[LABEL_COL].to_numpy(dtype=int)
    x_test = test_df[FEATURE_NAMES].to_numpy(dtype=float)
    y_test = test_df[LABEL_COL].to_numpy(dtype=int)

    classifier = new_classifier()
    classifier.fit(x_train, y_train)

    probs = classifier.predict_proba(x_test)[:, 1]
    metrics: dict = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "base_rate": float(np.mean(y_train)) if len(y_train) else 0.0,
        "accuracy": float(accuracy_score(y_test, probs > 0.5)) if len(set(y_test)) > 1 else None,
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(set(y_test)) > 1 else None,
        "brier": float(brier_score_loss(y_test, probs)) if len(y_test) else None,
        "log_loss": float(log_loss(y_test, probs, labels=[0, 1])) if len(set(y_test)) > 1 else None,
    }

    regressor = None
    if RETURN_COL in df.columns and df[RETURN_COL].notna().sum() > 50:
        regressor = new_regressor()
        regressor.fit(x_train, train_df[RETURN_COL].to_numpy(dtype=float))
        preds = regressor.predict(x_test)
        metrics["return_mae"] = float(
            mean_absolute_error(test_df[RETURN_COL].to_numpy(dtype=float), preds)
        )

    model = ScoringModel(
        classifier=classifier,
        regressor=regressor,
        metadata=ModelMetadata(
            name=name,
            trained_at=datetime.now(UTC).isoformat(),
            n_samples=int(len(df)),
            metrics=metrics,
            target_pct=target_pct,
            stop_multiple=stop_multiple,
            synthetic=synthetic,
            orca_version=__version__,
        ),
    )
    if save:
        model.save(name)

    log.info(
        "trained %s on %d rows | auc=%s brier=%s",
        name,
        len(df),
        _fmt(metrics.get("roc_auc")),
        _fmt(metrics.get("brier")),
    )
    return model, metrics


def _fmt(x: float | None) -> str:
    return f"{x:.3f}" if isinstance(x, float) else "n/a"
