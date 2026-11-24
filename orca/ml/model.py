"""The scoring model: P(profitable) plus an expected-return estimate.

Classifier  -> probability the structure reaches target before stop
Regressor   -> expected return as a fraction of credit received

Both are sklearn pipelines persisted with joblib next to a JSON metadata file,
so `data/models/<name>.joblib` + `<name>.json` is a complete, inspectable
artifact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from orca.config import get_settings
from orca.logging_setup import get_logger
from orca.ml.features import FEATURE_NAMES, to_vector

log = get_logger(__name__)


@dataclass
class ModelMetadata:
    name: str = "default"
    trained_at: str = ""
    n_samples: int = 0
    n_features: int = 0
    features: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    label_definition: str = "target reached before stop"
    target_pct: float = 0.5
    stop_multiple: float = 2.0
    synthetic: bool = False
    orca_version: str = ""


def new_classifier() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=4,
                    max_iter=250,
                    learning_rate=0.06,
                    l2_regularization=1.0,
                    min_samples_leaf=15,
                    random_state=7,
                ),
            ),
        ]
    )


def new_regressor() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "reg",
                HistGradientBoostingRegressor(
                    max_depth=4, max_iter=250, learning_rate=0.06, random_state=7
                ),
            ),
        ]
    )


class ScoringModel:
    """Wraps the fitted pipelines and knows how to score a feature dict."""

    def __init__(
        self,
        classifier: Pipeline | None = None,
        regressor: Pipeline | None = None,
        metadata: ModelMetadata | None = None,
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.metadata = metadata or ModelMetadata()

    # -- scoring ---------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self.classifier is not None

    def score(self, features: dict[str, float]) -> tuple[float | None, float | None]:
        """(probability of profit, expected return on credit)."""
        if self.classifier is None:
            return None, None
        x = np.array([to_vector(features)], dtype=float)
        prob = float(self.classifier.predict_proba(x)[0][1])
        expected = float(self.regressor.predict(x)[0]) if self.regressor is not None else None
        return prob, expected

    def score_many(self, rows: list[dict[str, float]]) -> list[tuple[float | None, float | None]]:
        if self.classifier is None or not rows:
            return [(None, None)] * len(rows)
        x = np.array([to_vector(r) for r in rows], dtype=float)
        probs = self.classifier.predict_proba(x)[:, 1]
        if self.regressor is not None:
            expected = self.regressor.predict(x)
            return [(float(p), float(e)) for p, e in zip(probs, expected, strict=False)]
        return [(float(p), None) for p in probs]

    def feature_importance(self) -> dict[str, float]:
        """Permutation importance is the honest measure; this is the cheap proxy."""
        if self.classifier is None:
            return {}
        clf = self.classifier.named_steps.get("clf")
        scores = getattr(clf, "feature_importances_", None)
        if scores is None:
            return {}
        return dict(
            sorted(
                zip(FEATURE_NAMES, (float(s) for s in scores), strict=False),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )

    # -- persistence -----------------------------------------------------
    def save(self, name: str | None = None, model_dir: Path | None = None) -> Path:
        name = name or self.metadata.name
        model_dir = Path(model_dir or get_settings().model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata.name = name
        self.metadata.trained_at = self.metadata.trained_at or datetime.now(UTC).isoformat()
        self.metadata.features = FEATURE_NAMES
        self.metadata.n_features = len(FEATURE_NAMES)
        path = model_dir / f"{name}.joblib"
        joblib.dump({"classifier": self.classifier, "regressor": self.regressor}, path)
        (model_dir / f"{name}.json").write_text(json.dumps(asdict(self.metadata), indent=2))
        log.info("saved model -> %s", path)
        return path

    @classmethod
    def load(cls, name: str | None = None, model_dir: Path | None = None) -> ScoringModel:
        settings = get_settings()
        name = name or settings.active_model
        model_dir = Path(model_dir or settings.model_dir)
        path = model_dir / f"{name}.joblib"
        if not path.exists():
            log.info("no model at %s - scoring will fall back to the heuristic", path)
            return cls(metadata=ModelMetadata(name=name))
        blob = joblib.load(path)
        meta_path = model_dir / f"{name}.json"
        meta = (
            ModelMetadata(**json.loads(meta_path.read_text()))
            if meta_path.exists()
            else ModelMetadata(name=name)
        )
        return cls(blob.get("classifier"), blob.get("regressor"), meta)


_cached: ScoringModel | None = None


def get_model(reload: bool = False) -> ScoringModel:
    """Process-wide model singleton (reload after training)."""
    global _cached
    if _cached is None or reload:
        _cached = ScoringModel.load()
    return _cached
