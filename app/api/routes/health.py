"""Unauthenticated liveness/readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from app import __version__
from app.api.deps import DatabaseDep, ModelProviderDep, SettingsDep
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service and dependency health",
    responses={503: {"description": "The database is unavailable"}},
)
async def health(
    request: Request,
    response: Response,
    settings: SettingsDep,
    database: DatabaseDep,
    models: ModelProviderDep,
) -> HealthResponse:
    """Report process health, database reachability and model availability.

    Returns 503 only when the **database** is unreachable. A missing model is
    reported as ``model_loaded: false`` with a 200, because an untrained
    deployment is still alive and able to accept a trained artifact.
    """
    db_ok = await database.healthcheck()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    scheduler = getattr(request.app.state, "scheduler", None)
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        app=settings.app_name,
        environment=settings.environment,
        version=__version__,
        database="up" if db_ok else "down",
        model_loaded=models.loaded,
        model_version=models.version,
        llm_provider=settings.llm_provider,
        scheduler_running=bool(scheduler is not None and scheduler.running),
    )
