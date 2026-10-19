"""Prediction endpoints: single record, batch upload, and history."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Path, Query, Request, UploadFile, status

from app.api.deps import (
    BatchServiceDep,
    CurrentSubject,
    PaginationDep,
    PredictionRepositoryDep,
    PredictionServiceDep,
    PredictorDep,
    SettingsDep,
)
from app.api.errors import BadRequestError, NotFoundError, PayloadTooLargeError
from app.logging_config import get_logger
from app.models.enums import JobStatus, PredictionSource, RiskBand
from app.models.schemas import (
    JobAccepted,
    Page,
    PredictionRequest,
    PredictionResponse,
)
from app.services.batch_service import BatchError

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"])
history_router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Score a single sensor record",
    responses={
        401: {"description": "Missing or invalid token"},
        422: {"description": "Invalid payload"},
        503: {"description": "No trained model is loaded"},
    },
)
async def predict_one(
    payload: PredictionRequest,
    subject: CurrentSubject,
    predictor: PredictorDep,
    service: PredictionServiceDep,
    repository: PredictionRepositoryDep,
) -> PredictionResponse:
    """Score one record and store it with the model version that produced it."""
    prediction = await service.predict(payload, repository, source=PredictionSource.API)
    return PredictionResponse.model_validate(prediction)


@router.post(
    "/batch",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a CSV for background scoring",
    responses={
        400: {"description": "The file could not be parsed or does not match the model"},
        401: {"description": "Missing or invalid token"},
        413: {"description": "The upload exceeds the size limit"},
        503: {"description": "No trained model is loaded"},
    },
)
async def predict_batch(
    request: Request,
    background: BackgroundTasks,
    subject: CurrentSubject,
    predictor: PredictorDep,
    service: BatchServiceDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="CSV with one sensor record per row")],
) -> JobAccepted:
    """Accept a CSV, register a job, and score it in the background.

    Returns 202 immediately: a 60k-row file takes long enough that holding the
    connection open would be the wrong design. Poll ``GET /jobs/{id}``.
    """
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"The upload is {len(content)} bytes; the limit is {settings.max_upload_bytes}.",
            details={"limit_bytes": settings.max_upload_bytes},
        )

    try:
        batch = service.parse_csv(content, filename=file.filename or "upload.csv")
    except BatchError as exc:
        raise BadRequestError(str(exc)) from exc

    job_id = await service.create_job(batch)
    background.add_task(service.run_job, job_id, batch)

    logger.info(
        "batch_accepted",
        job_id=str(job_id),
        rows=batch.row_count,
        filename=batch.filename,
        actor=subject,
    )
    return JobAccepted(
        job_id=job_id,
        status=JobStatus.QUEUED,
        total_rows=batch.row_count,
        model_version=predictor.version,
        poll_url=str(request.url_for("get_job", job_id=job_id)),
    )


@history_router.get(
    "",
    response_model=Page[PredictionResponse],
    summary="List stored predictions, newest first",
    responses={401: {"description": "Missing or invalid token"}},
)
async def list_predictions(
    subject: CurrentSubject,
    repository: PredictionRepositoryDep,
    pagination: PaginationDep,
    risk_band: Annotated[RiskBand | None, Query(description="Filter by risk band")] = None,
    model_version: Annotated[
        str | None, Query(max_length=48, description="Filter by model version")
    ] = None,
    vehicle_id: Annotated[str | None, Query(max_length=64, description="Filter by vehicle")] = None,
    source: Annotated[
        PredictionSource | None, Query(description="Filter by how it was submitted")
    ] = None,
    predicted_failure: Annotated[
        bool | None, Query(description="Filter to flagged (true) or clear (false)")
    ] = None,
) -> Page[PredictionResponse]:
    """The audit log, filterable by band, model version, vehicle and source."""
    rows, total = await repository.list_predictions(
        risk_band=risk_band,
        model_version=model_version,
        vehicle_id=vehicle_id,
        source=source,
        predicted_failure=predicted_failure,
        page=pagination.page,
        page_size=pagination.page_size,
    )
    total_pages = -(-total // pagination.page_size) if total else 0
    return Page[PredictionResponse](
        items=[PredictionResponse.model_validate(row) for row in rows],
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
        total_pages=total_pages,
    )


@history_router.get(
    "/{prediction_id}",
    response_model=PredictionResponse,
    summary="Fetch one stored prediction",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "Prediction not found"},
    },
)
async def get_prediction(
    subject: CurrentSubject,
    repository: PredictionRepositoryDep,
    prediction_id: Annotated[uuid.UUID, Path(description="Prediction identifier")],
) -> PredictionResponse:
    prediction = await repository.get(prediction_id)
    if prediction is None:
        raise NotFoundError(
            f"Prediction {prediction_id} was not found",
            details={"prediction_id": str(prediction_id)},
        )
    return PredictionResponse.model_validate(prediction)
