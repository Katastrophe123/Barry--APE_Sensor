"""Drift-check endpoints."""

from __future__ import annotations

import math
import uuid
from typing import Annotated

from fastapi import APIRouter, Path, Query, status

from app.api.deps import (
    CurrentSubject,
    DriftRepositoryDep,
    DriftServiceDep,
    PaginationDep,
    PredictionRepositoryDep,
    PredictorDep,
    SettingsDep,
)
from app.api.errors import NotFoundError
from app.models.enums import DriftOutcome
from app.models.schemas import DriftCheckResponse, Page

router = APIRouter(prefix="/drift", tags=["drift"])


@router.post(
    "/check",
    response_model=DriftCheckResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Compare recent prediction inputs with the training distribution",
    responses={
        401: {"description": "Missing or invalid token"},
        503: {"description": "No trained model is loaded"},
    },
)
async def run_drift_check(
    subject: CurrentSubject,
    predictor: PredictorDep,
    service: DriftServiceDep,
    settings: SettingsDep,
    predictions: PredictionRepositoryDep,
    drifts: DriftRepositoryDep,
    sample_size: Annotated[
        int | None,
        Query(ge=1, le=100_000, description="How many recent predictions to sample"),
    ] = None,
) -> DriftCheckResponse:
    """Run a drift check now and store the result.

    Also runs on a schedule; this endpoint exists so drift can be checked on
    demand after a suspected data-pipeline change.
    """
    check = await service.check_recent_predictions(
        predictions=predictions,
        drifts=drifts,
        sample_size=sample_size or settings.drift_sample_size,
    )
    return DriftCheckResponse.model_validate(check)


@router.get(
    "",
    response_model=Page[DriftCheckResponse],
    summary="List drift checks, newest first",
    responses={401: {"description": "Missing or invalid token"}},
)
async def list_drift_checks(
    subject: CurrentSubject,
    drifts: DriftRepositoryDep,
    pagination: PaginationDep,
    outcome: Annotated[DriftOutcome | None, Query(description="Filter by verdict")] = None,
) -> Page[DriftCheckResponse]:
    rows, total = await drifts.list_checks(
        outcome=outcome, page=pagination.page, page_size=pagination.page_size
    )
    return Page[DriftCheckResponse](
        items=[DriftCheckResponse.model_validate(row) for row in rows],
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
        total_pages=max(1, math.ceil(total / pagination.page_size)) if total else 0,
    )


@router.get(
    "/latest",
    response_model=DriftCheckResponse,
    summary="The most recent drift check",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "No drift check has been run yet"},
    },
)
async def latest_drift_check(
    subject: CurrentSubject,
    drifts: DriftRepositoryDep,
) -> DriftCheckResponse:
    check = await drifts.latest()
    if check is None:
        raise NotFoundError("No drift check has been recorded yet")
    return DriftCheckResponse.model_validate(check)


@router.get(
    "/{check_id}",
    response_model=DriftCheckResponse,
    summary="Fetch one drift check, including per-feature detail",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "Drift check not found"},
    },
)
async def get_drift_check(
    subject: CurrentSubject,
    drifts: DriftRepositoryDep,
    check_id: Annotated[uuid.UUID, Path(description="Drift check identifier")],
) -> DriftCheckResponse:
    check = await drifts.get(check_id)
    if check is None:
        raise NotFoundError(
            f"Drift check {check_id} was not found", details={"check_id": str(check_id)}
        )
    return DriftCheckResponse.model_validate(check)
