"""Holds the live model for the process and allows hot reloads.

The artifact is loaded once at startup rather than per request (unpickling a
RandomForest per call would dominate latency). A reload is explicit, so a newly
trained model becomes live without a restart - and a failed reload leaves the
previous model serving rather than taking the service down.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.logging_config import get_logger
from app.ml.drift import DriftDetector
from app.ml.predictor import Predictor
from app.ml.registry import ModelNotFoundError, ModelRegistry
from app.models.schemas import ModelSummary

logger = get_logger(__name__)


class ModelProvider:
    """Lazy, reload-able holder for the current :class:`Predictor`."""

    def __init__(
        self,
        artifacts_dir: Path | str,
        *,
        drift_kwargs: dict[str, float | int] | None = None,
    ) -> None:
        self.registry = ModelRegistry(artifacts_dir)
        self._predictor: Predictor | None = None
        self._lock = asyncio.Lock()
        self._drift_kwargs = drift_kwargs or {}

    # --- state ------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    @property
    def version(self) -> str | None:
        return self._predictor.version if self._predictor else None

    def predictor_or_none(self) -> Predictor | None:
        return self._predictor

    def require_predictor(self) -> Predictor:
        """Return the live predictor, or raise if nothing is trained."""
        if self._predictor is None:
            raise ModelNotFoundError(
                f"No model is loaded from {self.registry.artifacts_dir}. "
                "Run `python scripts/train.py` and restart, or POST /models/reload."
            )
        return self._predictor

    # --- loading ----------------------------------------------------------
    async def load(self) -> Predictor | None:
        """Load the current artifact. Returns ``None`` if none exists yet.

        Startup must not fail just because the service has not been trained
        yet - ``/health`` reports ``model_loaded: false`` and prediction
        endpoints return 503 until a model appears.
        """
        async with self._lock:
            try:
                loaded = await asyncio.to_thread(self.registry.load_current)
            except ModelNotFoundError as exc:
                logger.warning("model_not_available", reason=str(exc))
                return None
            except Exception as exc:  # noqa: BLE001 - a bad artifact must not crash boot
                logger.exception("model_load_failed", error=str(exc))
                return None

            self._predictor = Predictor(loaded)
            logger.info(
                "model_loaded",
                version=loaded.version,
                model_type=loaded.metadata.model_type,
                features=loaded.metadata.n_features,
                threshold=loaded.threshold,
            )
            return self._predictor

    async def reload(self) -> Predictor | None:
        """Re-read the pointer and swap the live model if it changed."""
        previous = self.version
        predictor = await self.load()
        if predictor is not None and previous != predictor.version:
            logger.info("model_swapped", previous=previous, current=predictor.version)
        return predictor

    # --- derived objects --------------------------------------------------
    def drift_detector(self) -> DriftDetector:
        """A detector seeded with the live model's training statistics."""
        predictor = self.require_predictor()
        return DriftDetector.from_metadata(
            predictor.metadata.feature_stats,
            model_version=predictor.version,
            **self._drift_kwargs,  # type: ignore[arg-type]
        )

    def summary(self) -> ModelSummary:
        """The payload for ``GET /models/current``."""
        predictor = self.require_predictor()
        metadata = predictor.metadata
        return ModelSummary(
            version=metadata.version,
            model_type=metadata.model_type,
            trained_at=metadata.trained_at,
            decision_threshold=metadata.decision_threshold,
            n_features=metadata.n_features,
            training_rows=metadata.training_rows,
            test_rows=metadata.test_rows,
            positive_rate=metadata.positive_rate,
            metrics=metadata.metrics,
            candidate_comparison=metadata.candidate_comparison,
            feature_list=metadata.feature_list,
            accepted_feature_list=metadata.accepted_feature_list,
            dataset_fingerprint=metadata.dataset_fingerprint,
            library_versions=metadata.library_versions,
            available_versions=self.registry.list_versions(),
        )
