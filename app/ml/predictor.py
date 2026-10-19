"""Inference wrapper around a loaded model artifact.

The same cleaner and feature pipeline that were fitted during training are
replayed here, so a prediction at serving time sees exactly the transformations
the model was trained on. This is the whole reason
:class:`~app.ml.cleaning.DataCleaner` is a fit/transform class rather than a
notebook cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.logging_config import get_logger
from app.ml.registry import LoadedModel
from app.ml.schema import TARGET_COLUMN

logger = get_logger(__name__)

#: Risk bands are defined *relative to the model's own tuned threshold*, not as
#: fixed probabilities, because the cost-optimal threshold for this problem sits
#: far below 0.5. Bands are therefore consistent with ``predicted_failure``:
#:   low    -> below threshold            (not predicted to fail)
#:   medium -> threshold .. 2x threshold  (predicted to fail)
#:   high   -> >= 2x threshold            (predicted to fail, act first)
HIGH_RISK_MULTIPLIER = 2.0


class PredictionError(Exception):
    """Raised when a request cannot be scored."""


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    """One scored record."""

    failure_probability: float
    predicted_failure: bool
    risk_band: str  # low | medium | high
    threshold: float
    model_version: str
    missing_features: tuple[str, ...] = ()
    unexpected_features: tuple[str, ...] = ()

    @property
    def is_high_risk(self) -> bool:
        return self.risk_band == "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_probability": self.failure_probability,
            "predicted_failure": self.predicted_failure,
            "risk_band": self.risk_band,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "missing_features": list(self.missing_features),
            "unexpected_features": list(self.unexpected_features),
        }


class Predictor:
    """Scores single records or whole frames with a loaded model."""

    def __init__(self, loaded: LoadedModel) -> None:
        self._loaded = loaded

    @property
    def metadata(self) -> Any:
        return self._loaded.metadata

    @property
    def version(self) -> str:
        return self._loaded.version

    @property
    def threshold(self) -> float:
        return self._loaded.threshold

    @property
    def accepted_features(self) -> list[str]:
        """Every column a caller may send.

        Includes columns that cleaning dropped at training time: sending them is
        harmless and they are simply ignored, so they must not be reported back
        as "unexpected".
        """
        metadata = self._loaded.metadata
        return list(
            metadata.accepted_feature_list or metadata.raw_feature_list or metadata.feature_list
        )

    @property
    def required_features(self) -> list[str]:
        """Columns that actually reach the model."""
        metadata = self._loaded.metadata
        return list(metadata.raw_feature_list or metadata.feature_list)

    def risk_band(self, probability: float) -> str:
        """Band a probability relative to the model's own threshold.

        ``low`` is exactly "not predicted to fail", so the band and
        ``predicted_failure`` can never disagree.
        """
        if probability < self.threshold:
            return "low"
        if probability >= self.threshold * HIGH_RISK_MULTIPLIER:
            return "high"
        return "medium"

    def prepare_for_drift(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply training-time cleaning only, leaving NaNs intact.

        The drift detector needs the *uncleaned-of-missingness* view so it can
        measure missing-rate drift, which is why this stops before imputation.
        """
        return self._loaded.cleaner.transform(frame)

    # --- scoring ----------------------------------------------------------
    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Return failure probabilities for every row, in input order."""
        if frame.empty:
            return np.empty(0, dtype="float64")
        cleaned = self._loaded.cleaner.transform(frame)
        transformed = self._loaded.pipeline.transform(cleaned)
        probabilities = self._loaded.model.predict_proba(transformed)[:, 1]
        return np.asarray(probabilities, dtype="float64")

    def predict_one(self, features: dict[str, float | None]) -> PredictionOutcome:
        """Score a single record supplied as a feature mapping."""
        required = set(self.required_features)
        accepted = set(self.accepted_features)
        provided = set(features)
        missing = tuple(sorted(required - provided))
        unexpected = tuple(sorted(provided - accepted))

        if not required & provided:
            raise PredictionError(
                "None of the supplied features are known to the current model "
                f"(version {self.version}). Expected names like "
                f"{self.required_features[:5]}."
            )

        frame = pd.DataFrame([features])
        probability = float(self.predict_frame(frame)[0])

        outcome = PredictionOutcome(
            failure_probability=round(probability, 6),
            predicted_failure=probability >= self.threshold,
            risk_band=self.risk_band(probability),
            threshold=self.threshold,
            model_version=self.version,
            missing_features=missing,
            unexpected_features=unexpected,
        )
        if missing:
            logger.info(
                "prediction_with_missing_features",
                count=len(missing),
                sample=list(missing[:5]),
                model_version=self.version,
            )
        return outcome

    def score_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score a frame and return it with prediction columns appended.

        The target column, if present (a labelled backtest file), is preserved
        so the caller can compare predictions against truth.
        """
        probabilities = self.predict_frame(frame)
        result = frame.copy()
        result["failure_probability"] = np.round(probabilities, 6)
        result["predicted_failure"] = probabilities >= self.threshold
        result["risk_band"] = [self.risk_band(float(value)) for value in probabilities]
        result["model_version"] = self.version
        if TARGET_COLUMN in frame.columns:
            result = result.rename(columns={TARGET_COLUMN: "actual_class"})
        return result
