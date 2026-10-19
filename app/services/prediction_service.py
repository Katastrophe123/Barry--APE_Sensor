"""Single-record prediction, with the audit row written as part of the call."""

from __future__ import annotations

from time import perf_counter

from app.db.models import Prediction
from app.db.repository import PredictionRepository
from app.logging_config import get_logger
from app.models.enums import PredictionSource, RiskBand
from app.models.schemas import PredictionRequest
from app.services.model_provider import ModelProvider

logger = get_logger(__name__)


class PredictionService:
    """Scores one record and records it for auditability."""

    def __init__(self, *, models: ModelProvider) -> None:
        self._models = models

    async def predict(
        self,
        request: PredictionRequest,
        repository: PredictionRepository,
        *,
        source: PredictionSource = PredictionSource.API,
    ) -> Prediction:
        """Score ``request`` and persist the result.

        The prediction is stored *with its model version and threshold*, so a
        later dispute about a prediction can be resolved without guessing which
        artifact produced it.
        """
        predictor = self._models.require_predictor()
        started = perf_counter()
        outcome = predictor.predict_one(request.features)
        latency_ms = max(0, int((perf_counter() - started) * 1000))

        prediction = await repository.record(
            features=dict(request.features),
            failure_probability=outcome.failure_probability,
            predicted_failure=outcome.predicted_failure,
            risk_band=RiskBand(outcome.risk_band),
            decision_threshold=outcome.threshold,
            model_version=outcome.model_version,
            vehicle_id=request.vehicle_id,
            source=source,
            missing_features=outcome.missing_features,
            unexpected_features=outcome.unexpected_features,
            request_metadata=request.metadata,
            latency_ms=latency_ms,
        )

        logger.info(
            "prediction_recorded",
            prediction_id=str(prediction.id),
            vehicle_id=request.vehicle_id,
            probability=outcome.failure_probability,
            risk_band=outcome.risk_band,
            model_version=outcome.model_version,
            latency_ms=latency_ms,
            missing_features=len(outcome.missing_features),
        )
        return prediction
