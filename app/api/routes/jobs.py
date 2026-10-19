"""Job status endpoints."""

from __future__ import annotations

import math
import uuid
from typing import Annotated

from fastapi import APIRouter, Path, Query

from app.api.deps import CurrentSubject, JobRepositoryDep, PaginationDep
from app.api.errors import NotFoundError
from app.models.enums import JobKind, JobStatus
from app.models.schemas import JobResponse, Page

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "",
    response_model=Page[JobResponse],
    summary="List jobs, newest first",
    responses={401: {"description": "Missing or invalid token"}},
)
async def list_jobs(
    subject: CurrentSubject,
    repository: JobRepositoryDep,
    pagination: PaginationDep,
    status_filter: Annotated[
        JobStatus | None, Query(alias="status", description="Filter by job status")
    ] = None,
    kind: Annotated[JobKind | None, Query(description="Filter by job kind")] = None,
) -> Page[JobResponse]:
    rows, total = await repository.list_jobs(
        status=status_filter,
        kind=kind,
        page=pagination.page,
        page_size=pagination.page_size,
    )
    return Page[JobResponse](
        items=[JobResponse.model_validate(row) for row in rows],
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
        total_pages=max(1, math.ceil(total / pagination.page_size)) if total else 0,
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    name="get_job",
    summary="Poll a batch or scheduled job",
    responses={
        401: {"description": "Missing or invalid token"},
        404: {"description": "Job not found"},
    },
)
async def get_job(
    subject: CurrentSubject,
    repository: JobRepositoryDep,
    job_id: Annotated[uuid.UUID, Path(description="Job identifier")],
) -> JobResponse:
    """Poll a job until ``status`` is ``succeeded`` or ``failed``."""
    job = await repository.get(job_id)
    if job is None:
        raise NotFoundError(f"Job {job_id} was not found", details={"job_id": str(job_id)})
    return JobResponse.model_validate(job)
