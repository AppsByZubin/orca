"""Machine-learning layer: features -> labels -> model -> score."""

from orca.ml.features import FEATURE_NAMES, candidate_features, market_context
from orca.ml.heuristic import score as heuristic_score
from orca.ml.labels import Outcome, simulate_outcome
from orca.ml.model import ModelMetadata, ScoringModel, get_model
from orca.ml.train import train_model

__all__ = [
    "FEATURE_NAMES",
    "ModelMetadata",
    "Outcome",
    "ScoringModel",
    "candidate_features",
    "get_model",
    "heuristic_score",
    "market_context",
    "simulate_outcome",
    "train_model",
]
