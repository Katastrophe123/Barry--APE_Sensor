"""Model evaluation.

A note on metric choice, because this is where v1 went wrong
-------------------------------------------------------------
v1 benchmarked RandomForest against XGBoost on **R², RMSE and MAE**. Those are
regression metrics, and APS failure prediction is binary classification on a
severely imbalanced target (~1.7% positive). Applying them to a 0/1 target is a
category error: R² on a binary outcome is not interpretable, and a model that
predicts "never fails" for every truck scores well on MAE while being useless.

:class:`Evaluator` therefore reports proper classification metrics - ROC-AUC,
PR-AUC, precision, recall, F1, the confusion matrix - plus the **official
Scania cost metric** (``10 x FP + 500 x FN``), which is the number the business
actually cares about and the one used for model selection here.

The v1 regression metrics are still computed, clearly namespaced under
``legacy_regression``, so the upgrade can be compared against the old numbers
rather than silently replacing them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from app.ml.schema import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    @property
    def total(self) -> int:
        return (
            self.true_negatives + self.false_positives + self.false_negatives + self.true_positives
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Everything one evaluation produced. All values are measured, not assumed."""

    # --- classification (the metrics that matter here) -------------------
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    balanced_accuracy: float
    confusion: ConfusionCounts
    #: 10 x FP + 500 x FN. Lower is better; this drives model selection.
    total_cost: int
    cost_per_1000_vehicles: float
    threshold: float

    # --- v1 metrics, retained for continuity only ------------------------
    legacy_regression: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confusion"] = asdict(self.confusion)
        return payload

    def summary_line(self) -> str:
        return (
            f"cost={self.total_cost} roc_auc={self.roc_auc:.4f} "
            f"pr_auc={self.pr_auc:.4f} recall={self.recall:.4f} "
            f"precision={self.precision:.4f} f1={self.f1:.4f}"
        )


class Evaluator:
    """Computes :class:`EvaluationResult` from labels and predicted scores."""

    def __init__(
        self,
        *,
        cost_false_positive: int = COST_FALSE_POSITIVE,
        cost_false_negative: int = COST_FALSE_NEGATIVE,
    ) -> None:
        self.cost_false_positive = cost_false_positive
        self.cost_false_negative = cost_false_negative

    def total_cost(self, false_positives: int, false_negatives: int) -> int:
        return (
            false_positives * self.cost_false_positive + false_negatives * self.cost_false_negative
        )

    def evaluate(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        *,
        threshold: float = 0.5,
    ) -> EvaluationResult:
        """Evaluate predicted probabilities against binary labels."""
        truth = np.asarray(y_true).astype(int).ravel()
        score = np.asarray(y_score, dtype="float64").ravel()
        if truth.shape != score.shape:
            raise ValueError(f"y_true and y_score must align: {truth.shape} vs {score.shape}")
        if truth.size == 0:
            raise ValueError("Cannot evaluate an empty prediction set")

        predicted = (score >= threshold).astype(int)
        matrix = confusion_matrix(truth, predicted, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        counts = ConfusionCounts(
            true_negatives=tn, false_positives=fp, false_negatives=fn, true_positives=tp
        )

        # ROC/PR AUC are undefined with a single class present; report 0.0
        # rather than raising, so a degenerate fold does not kill a run.
        both_classes_present = len(np.unique(truth)) > 1
        roc = float(roc_auc_score(truth, score)) if both_classes_present else 0.0
        pr = float(average_precision_score(truth, score)) if both_classes_present else 0.0

        cost = self.total_cost(fp, fn)
        return EvaluationResult(
            roc_auc=roc,
            pr_auc=pr,
            precision=float(precision_score(truth, predicted, zero_division=0)),
            recall=float(recall_score(truth, predicted, zero_division=0)),
            f1=float(f1_score(truth, predicted, zero_division=0)),
            accuracy=float(accuracy_score(truth, predicted)),
            balanced_accuracy=float(balanced_accuracy_score(truth, predicted)),
            confusion=counts,
            total_cost=cost,
            cost_per_1000_vehicles=round(cost / counts.total * 1000, 2),
            threshold=threshold,
            legacy_regression=self._legacy_regression(truth, score),
        )

    @staticmethod
    def _legacy_regression(truth: np.ndarray, score: np.ndarray) -> dict[str, float]:
        """The v1 metrics, computed on probability vs the 0/1 target.

        Reported for comparability with the notebook only. See the module
        docstring for why these should not drive any decision.
        """
        return {
            "r2": round(float(r2_score(truth, score)), 6),
            "rmse": round(float(root_mean_squared_error(truth, score)), 6),
            "mae": round(float(mean_absolute_error(truth, score)), 6),
        }

    def best_threshold(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        *,
        candidates: int = 99,
    ) -> tuple[float, int]:
        """Find the threshold minimising Scania cost.

        With a 50:1 FN:FP cost ratio the cost-optimal threshold is far below
        0.5, so leaving it at the default would throw away most of the model's
        value. Returns ``(threshold, cost)``.
        """
        truth = np.asarray(y_true).astype(int).ravel()
        score = np.asarray(y_score, dtype="float64").ravel()
        grid = np.linspace(0.01, 0.99, candidates)

        best = (0.5, self.total_cost(*self._counts(truth, score, 0.5)))
        for threshold in grid:
            fp, fn = self._counts(truth, score, float(threshold))
            cost = self.total_cost(fp, fn)
            if cost < best[1]:
                best = (round(float(threshold), 4), cost)
        return best

    @staticmethod
    def _counts(truth: np.ndarray, score: np.ndarray, threshold: float) -> tuple[int, int]:
        predicted = (score >= threshold).astype(int)
        false_positives = int(((predicted == 1) & (truth == 0)).sum())
        false_negatives = int(((predicted == 0) & (truth == 1)).sum())
        return false_positives, false_negatives
