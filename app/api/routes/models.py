"""Model introspection and hot-reload endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import CurrentSubject, ModelProviderDep, PredictorDep
from app.api.errors import ServiceUnavailableError
from app.logging_config import get_logger
from app.models.schemas import ModelSummary

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "/current",
    response_model=ModelSummary,
    summary="The live model, its measured metrics and its feature list",
    responses={
        401: {"description": "Missing or invalid token"},
        503: {"description": "No trained model is loaded"},
    },
)
async def get_current_model(
    subject: CurrentSubject,
    predictor: PredictorDep,
    models: ModelProviderDep,
) -> ModelSummary:
    """Everything needed to interpret a prediction from this deployment.

    ``metrics`` and ``candidate_comparison`` are the values measured on the
    held-out split during the run that produced this artifact - not targets.
    """
    return models.summary()


@router.post(
    "/reload",
    response_model=ModelSummary,
    status_code=status.HTTP_200_OK,
    summary="Re-read the artifacts directory and swap in the newest model",
    responses={
        401: {"description": "Missing or invalid token"},
        503: {"description": "No model could be loaded"},
    },
)
async def reload_model(
    subject: CurrentSubject,
    models: ModelProviderDep,
) -> ModelSummary:
    """Pick up a newly trained model without restarting the process.

    On failure the previously loaded model keeps serving, and this returns 503 -
    a bad new artifact must not take the service down.
    """
    previous = models.version
    predictor = await models.reload()
    if predictor is None:
        raise ServiceUnavailableError(
            "No model could be loaded from the artifacts directory.",
            details={"previous_version": previous},
        )
    logger.info("model_reload_requested", actor=subject, version=predictor.version)
    return models.summary()
