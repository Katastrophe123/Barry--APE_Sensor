"""Model training: benchmark the candidates, pick the cheapest.

v1 trained RandomForest and XGBoost in notebook cells and eyeballed R². Here
both candidates are trained on the same stratified split, evaluated with the
same :class:`~app.ml.evaluation.Evaluator`, and the winner is chosen by the
**Scania cost metric** - which, with a 50:1 false-negative penalty, is the only
selection criterion that reflects the business problem.

Nothing here invents a metric: every number in a :class:`TrainingResult` comes
from an actual fit on actual data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.logging_config import get_logger
from app.ml.cleaning import DataCleaner
from app.ml.evaluation import EvaluationResult, Evaluator
from app.ml.features import FeaturePipeline

logger = get_logger(__name__)

ModelFactory = Callable[[float], Any]


def _random_forest(scale_pos_weight: float) -> RandomForestClassifier:
    """Class weights, not resampling: cheaper and keeps the data honest."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


def xgboost_available() -> bool:
    """Whether XGBoost's native library can actually be loaded.

    On macOS the wheel needs ``libomp`` (``brew install libomp``); importing
    without it raises at load time rather than at install time. Checking here
    lets the trainer fall back to RandomForest alone instead of failing a whole
    training run, and lets the service start on a host without it.
    """
    try:
        import xgboost  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any load failure means unavailable
        logger.warning("xgboost_unavailable", error=str(exc)[:200])
        return False
    return True


def _xgboost(scale_pos_weight: float) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        # The imbalance correction XGBoost expects: n_negative / n_positive.
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


#: Every candidate this project knows how to train.
ALL_MODEL_FACTORIES: dict[str, ModelFactory] = {
    "random_forest": _random_forest,
    "xgboost": _xgboost,
}


def available_model_factories() -> dict[str, ModelFactory]:
    """Candidates that can actually run on this host.

    Add a model to :data:`ALL_MODEL_FACTORIES` and it is benchmarked
    automatically, provided its runtime is importable.
    """
    factories = dict(ALL_MODEL_FACTORIES)
    if not xgboost_available():
        factories.pop("xgboost", None)
    return factories


@dataclass(slots=True)
class CandidateResult:
    """One trained candidate and how it scored on the held-out split."""

    name: str
    model: Any
    evaluation: EvaluationResult
    tuned_threshold: float
    tuned_cost: int
    fit_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tuned_threshold": self.tuned_threshold,
            "tuned_cost": self.tuned_cost,
            "fit_seconds": round(self.fit_seconds, 3),
            "metrics": self.evaluation.to_dict(),
        }


@dataclass(slots=True)
class TrainingResult:
    """The full outcome of a training run."""

    best_name: str
    best_model: Any
    best_threshold: float
    cleaner: DataCleaner
    pipeline: FeaturePipeline
    candidates: list[CandidateResult] = field(default_factory=list)
    training_rows: int = 0
    test_rows: int = 0
    positive_rate: float = 0.0

    @property
    def best(self) -> CandidateResult:
        return next(item for item in self.candidates if item.name == self.best_name)

    def comparison(self) -> list[dict[str, Any]]:
        """Candidate comparison, cheapest first."""
        return [item.to_dict() for item in sorted(self.candidates, key=lambda c: c.tuned_cost)]


class ModelTrainer:
    """Trains every candidate and selects by held-out Scania cost."""

    def __init__(
        self,
        *,
        test_size: float = 0.2,
        random_state: int = 42,
        evaluator: Evaluator | None = None,
        tune_threshold: bool = True,
        model_factories: dict[str, ModelFactory] | None = None,
    ) -> None:
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")
        self.test_size = test_size
        self.random_state = random_state
        self.evaluator = evaluator or Evaluator()
        self.tune_threshold = tune_threshold
        # `is None` rather than falsy: an explicitly-empty dict is a caller
        # error, not a request for the defaults.
        self.model_factories = (
            available_model_factories() if model_factories is None else dict(model_factories)
        )
        if not self.model_factories:
            raise ValueError("No model candidates are available on this host")

    @staticmethod
    def scale_pos_weight(y: pd.Series | np.ndarray) -> float:
        """``n_negative / n_positive``, clamped so a degenerate split is safe."""
        target = np.asarray(y).astype(int).ravel()
        positives = int((target == 1).sum())
        negatives = int((target == 0).sum())
        if positives == 0:
            return 1.0
        return max(1.0, negatives / positives)

    def train(self, raw: pd.DataFrame) -> TrainingResult:
        """Clean, split, fit every candidate, and pick the cheapest."""
        cleaner = DataCleaner()
        features, target = cleaner.fit_transform(raw)

        if target.nunique() < 2:
            raise ValueError(
                "Training data contains a single class; both pos and neg rows are required"
            )

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target,
        )

        pipeline = FeaturePipeline()
        x_train_t = pipeline.fit_transform(x_train)
        x_test_t = pipeline.transform(x_test)

        weight = self.scale_pos_weight(y_train)
        candidates: list[CandidateResult] = []

        for name, factory in self.model_factories.items():
            started = perf_counter()
            model = factory(weight)
            model.fit(x_train_t, y_train)
            fit_seconds = perf_counter() - started

            scores = model.predict_proba(x_test_t)[:, 1]
            if self.tune_threshold:
                threshold, cost = self.evaluator.best_threshold(y_test.to_numpy(), scores)
            else:
                threshold = 0.5
                cost = self.evaluator.evaluate(y_test.to_numpy(), scores, threshold=0.5).total_cost

            evaluation = self.evaluator.evaluate(y_test.to_numpy(), scores, threshold=threshold)
            candidates.append(
                CandidateResult(
                    name=name,
                    model=model,
                    evaluation=evaluation,
                    tuned_threshold=threshold,
                    tuned_cost=cost,
                    fit_seconds=fit_seconds,
                )
            )
            logger.info(
                "candidate_trained",
                model=name,
                threshold=threshold,
                fit_seconds=round(fit_seconds, 3),
                **{
                    "cost": evaluation.total_cost,
                    "roc_auc": round(evaluation.roc_auc, 4),
                    "pr_auc": round(evaluation.pr_auc, 4),
                    "recall": round(evaluation.recall, 4),
                    "precision": round(evaluation.precision, 4),
                },
            )

        # Selection: lowest cost, tie-broken by PR-AUC (better ranking quality).
        best = min(candidates, key=lambda c: (c.tuned_cost, -c.evaluation.pr_auc))
        logger.info(
            "best_model_selected",
            model=best.name,
            cost=best.tuned_cost,
            threshold=best.tuned_threshold,
        )

        return TrainingResult(
            best_name=best.name,
            best_model=best.model,
            best_threshold=best.tuned_threshold,
            cleaner=cleaner,
            pipeline=pipeline,
            candidates=candidates,
            training_rows=len(x_train),
            test_rows=len(x_test),
            positive_rate=float(np.asarray(target).mean()),
        )
