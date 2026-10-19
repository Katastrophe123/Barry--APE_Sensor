"""Feature pipeline: imputation plus low-variance pruning.

Deliberately *not* scaled. The two candidate models are tree ensembles
(RandomForest, XGBoost), which are invariant to monotone rescaling, so a
``StandardScaler`` here would add a fitted artefact that changes nothing and
has to be versioned anyway. If a linear or distance-based model is ever added,
this is the one place that needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from app.logging_config import get_logger

logger = get_logger(__name__)


class FeaturePipelineError(Exception):
    """Raised when the pipeline is used before being fitted."""


@dataclass(frozen=True, slots=True)
class FeatureStats:
    """Per-feature training distribution, recorded for drift detection.

    Quantile edges are stored so the drift detector can bin an incoming batch
    against exactly the same boundaries the training data defined.
    """

    name: str
    mean: float
    std: float
    median: float
    minimum: float
    maximum: float
    missing_rate: float
    quantile_edges: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "missing_rate": self.missing_rate,
            "quantile_edges": list(self.quantile_edges),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureStats:
        return cls(
            name=str(data["name"]),
            mean=float(data["mean"]),
            std=float(data["std"]),
            median=float(data["median"]),
            minimum=float(data["minimum"]),
            maximum=float(data["maximum"]),
            missing_rate=float(data["missing_rate"]),
            quantile_edges=tuple(float(edge) for edge in data["quantile_edges"]),
        )


class FeaturePipeline:
    """Median-imputes, then drops near-constant columns.

    Near-constant pruning runs *after* imputation on purpose: a column that is
    95% missing and otherwise constant becomes genuinely constant once imputed,
    and only then is it safe to say it carries no signal.
    """

    DRIFT_BINS = 10

    def __init__(self, *, min_variance: float = 1e-12) -> None:
        self.min_variance = min_variance
        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.input_features_: list[str] = []
        self.feature_names_out_: list[str] = []
        self.dropped_low_variance_: list[str] = []
        self.feature_stats_: dict[str, FeatureStats] = {}
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise FeaturePipelineError("FeaturePipeline must be fitted before transform")

    @staticmethod
    def _quantile_edges(values: pd.Series, bins: int) -> tuple[float, ...]:
        """Monotone quantile edges, de-duplicated for low-cardinality columns."""
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.nanquantile(values.to_numpy(dtype="float64"), quantiles)
        unique = np.unique(edges[np.isfinite(edges)])
        if unique.size < 2:
            centre = float(unique[0]) if unique.size else 0.0
            return (centre - 0.5, centre + 0.5)
        return tuple(float(edge) for edge in unique)

    def _compute_stats(self, kept_columns: list[str], raw: pd.DataFrame) -> None:
        """Record training distributions from the *pre-imputation* values.

        Pre-imputation on purpose: the drift detector drops NaN from an
        incoming batch before binning it, so the reference distribution has to
        exclude imputed medians too. Comparing NaN-dropped batch values against
        median-inflated training quantiles would manufacture drift that is not
        there.
        """
        stats: dict[str, FeatureStats] = {}
        for name in kept_columns:
            original = raw[name]
            observed = original.dropna()
            # An all-missing column cannot happen here (it would be constant
            # after imputation and dropped), but guard anyway.
            reference = observed if not observed.empty else original.fillna(0.0)
            stats[name] = FeatureStats(
                name=name,
                mean=float(reference.mean()),
                std=float(reference.std(ddof=0)),
                median=float(reference.median()),
                minimum=float(reference.min()),
                maximum=float(reference.max()),
                missing_rate=float(original.isna().mean()),
                quantile_edges=self._quantile_edges(reference, self.DRIFT_BINS),
            )
        self.feature_stats_ = stats

    def fit(self, features: pd.DataFrame) -> FeaturePipeline:
        if features.empty or features.shape[1] == 0:
            raise FeaturePipelineError("Cannot fit a feature pipeline on an empty frame")

        self.input_features_ = list(features.columns)
        imputed_values = self._imputer.fit_transform(features)
        imputed = pd.DataFrame(imputed_values, columns=self.input_features_, index=features.index)

        variances = imputed.var(ddof=0)
        self.dropped_low_variance_ = sorted(
            variances[variances <= self.min_variance].index.tolist()
        )
        kept = imputed.drop(columns=self.dropped_low_variance_)
        self.feature_names_out_ = list(kept.columns)

        if not self.feature_names_out_:
            raise FeaturePipelineError(
                "Every feature was dropped as near-constant; check the input data"
            )

        self._compute_stats(self.feature_names_out_, features)
        self._fitted = True

        logger.info(
            "feature_pipeline_fitted",
            input_features=len(self.input_features_),
            output_features=len(self.feature_names_out_),
            dropped_low_variance=len(self.dropped_low_variance_),
        )
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        aligned = features.reindex(columns=self.input_features_)
        imputed_values = self._imputer.transform(aligned)
        imputed = pd.DataFrame(imputed_values, columns=self.input_features_, index=features.index)
        return imputed[self.feature_names_out_]

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        self.fit(features)
        return self.transform(features)

    def stats_as_dict(self) -> dict[str, dict[str, Any]]:
        """Serialisable training statistics, stored in the model metadata."""
        self._require_fitted()
        return {name: stats.to_dict() for name, stats in self.feature_stats_.items()}

    def as_sklearn_pipeline(self) -> Pipeline:
        """Expose the fitted imputer as an sklearn ``Pipeline``.

        Only used where sklearn tooling (e.g. ``cross_val_score``) expects a
        single estimator; the service itself calls this class directly.
        """
        self._require_fitted()
        return Pipeline([("imputer", self._imputer)])
