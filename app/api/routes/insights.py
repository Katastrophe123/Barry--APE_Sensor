"""The optional LLM step: a plain-English maintenance note for an operator."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import (
    CurrentSubject,
    ModelProviderDep,
    NoteServiceDep,
    PredictionRepositoryDep,
)
from app.api.errors import NotFoundError
from app.logging_config import get_logger
from app.models.enums import RiskBand
from app.models.schemas import MaintenanceNoteRequest, MaintenanceNoteResponse
from app.services.insight_service import MaintenanceNoteService

logger = get_logger(__name__)

router = APIRouter(prefix="/insights", tags=["insights"])


@router.post(
    "/maintenance-note",
    response_model=MaintenanceNoteResponse,
    status_code=status.HTTP_200_OK,
    summary="Turn a prediction into a short maintenance note",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "The referenced prediction does not exist"},
        422: {"description": "Neither prediction_id nor failure_probability supplied"},
    },
)
async def maintenance_note(
    payload: MaintenanceNoteRequest,
    subject: CurrentSubject,
    notes: NoteServiceDep,
    predictions: PredictionRepositoryDep,
    models: ModelProviderDep,
) -> MaintenanceNoteResponse:
    """Generate an operator-facing note for a stored or supplied prediction.

    Never fails because of the LLM: if the provider is unavailable a
    deterministic template is used and ``generated_by`` says ``template``.
    """
    model_version: str | None = None
    stored = None
    if payload.prediction_id is not None:
        stored = await predictions.get(payload.prediction_id)
        if stored is None:
            raise NotFoundError(
                f"Prediction {payload.prediction_id} was not found",
                details={"prediction_id": str(payload.prediction_id)},
            )

    if stored is not None:
        probability = stored.failure_probability
        band = RiskBand(stored.risk_band)
        vehicle_id = payload.vehicle_id or stored.vehicle_id
        model_version = stored.model_version
        signals = payload.top_signals or MaintenanceNoteService.top_signals_from_features(
            dict(stored.features or {})
        )
    else:
        # Inline mode: band the supplied probability with the live model's
        # threshold when one is loaded, else fall back to a plain 0.5 split.
        probability = float(payload.failure_probability or 0.0)
        predictor = models.predictor_or_none()
        band = (
            RiskBand(predictor.risk_band(probability))
            if predictor is not None
            else (RiskBand.HIGH if probability >= 0.5 else RiskBand.LOW)
        )
        vehicle_id = payload.vehicle_id
        model_version = models.version
        signals = list(payload.top_signals)

    note = await notes.generate(
        failure_probability=probability,
        risk_band=band,
        vehicle_id=vehicle_id,
        top_signals=signals,
        model_version=model_version,
    )

    logger.info(
        "maintenance_note_served",
        actor=subject,
        prediction_id=str(payload.prediction_id) if payload.prediction_id else None,
        risk_band=band.value,
        generated_by=note.generated_by,
    )
    return MaintenanceNoteResponse(
        prediction_id=payload.prediction_id,
        vehicle_id=vehicle_id,
        failure_probability=round(probability, 6),
        risk_band=band,
        note=note.text,
        model_version=model_version,
        llm_provider=note.provider,
        llm_model=note.model,
        generated_by=note.generated_by,
    )
