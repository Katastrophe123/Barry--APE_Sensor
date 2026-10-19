"""FastAPI dependency providers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import ServiceUnavailableError, UnauthorizedError
from app.config import Settings, get_settings
from app.core.security import AuthError, decode_access_token
from app.db.repository import DriftRepository, JobRepository, PredictionRepository
from app.db.session import Database
from app.ml.predictor import Predictor
from app.ml.registry import ModelNotFoundError
from app.services.batch_service import BatchScoringService
from app.services.drift_service import DriftService
from app.services.insight_service import MaintenanceNoteService
from app.services.model_provider import ModelProvider
from app.services.prediction_service import PredictionService

bearer_scheme = HTTPBearer(auto_error=False, description="JWT from POST /auth/token")


def get_app_settings(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings or get_settings()


def _require_state(request: Request, name: str) -> object:
    value = getattr(request.app.state, name, None)
    if value is None:  # pragma: no cover - would mean a broken lifespan
        raise RuntimeError(f"{name} is not initialised on app state")
    return value


def get_database(request: Request) -> Database:
    return _require_state(request, "database")  # type: ignore[return-value]


def get_model_provider(request: Request) -> ModelProvider:
    return _require_state(request, "models")  # type: ignore[return-value]


def get_prediction_service(request: Request) -> PredictionService:
    return _require_state(request, "prediction_service")  # type: ignore[return-value]


def get_batch_service(request: Request) -> BatchScoringService:
    return _require_state(request, "batch_service")  # type: ignore[return-value]


def get_drift_service(request: Request) -> DriftService:
    return _require_state(request, "drift_service")  # type: ignore[return-value]


def get_note_service(request: Request) -> MaintenanceNoteService:
    return _require_state(request, "note_service")  # type: ignore[return-value]


async def get_session(
    database: Annotated[Database, Depends(get_database)],
) -> AsyncIterator[AsyncSession]:
    """Request-scoped session: commits on success, rolls back on error."""
    async with database.session() as session:
        yield session


def get_prediction_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PredictionRepository:
    return PredictionRepository(session)


def get_job_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobRepository:
    return JobRepository(session)


def get_drift_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DriftRepository:
    return DriftRepository(session)


def require_model(
    models: Annotated[ModelProvider, Depends(get_model_provider)],
) -> Predictor:
    """Guard for endpoints that cannot work without a trained model.

    Returns 503 rather than 500: "no model trained yet" is a deployment state,
    not a bug, and the message says how to fix it.
    """
    try:
        return models.require_predictor()
    except ModelNotFoundError as exc:
        raise ServiceUnavailableError(str(exc), details={"model_loaded": False}) from exc


async def get_current_subject(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> str:
    """Validate the bearer token and return its subject."""
    if credentials is None or not credentials.credentials:
        raise UnauthorizedError("Missing bearer token")
    try:
        payload = decode_access_token(credentials.credentials, settings)
    except AuthError as exc:
        raise UnauthorizedError(exc.message) from exc
    return str(payload["sub"])


class PaginationParams:
    """Shared ``page`` / ``page_size`` query parameters."""

    def __init__(
        self,
        page: Annotated[int, Query(ge=1, description="1-based page number")] = 1,
        page_size: Annotated[int, Query(ge=1, le=100, description="Items per page (max 100)")] = 20,
    ) -> None:
        self.page = page
        self.page_size = page_size


CurrentSubject = Annotated[str, Depends(get_current_subject)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
DatabaseDep = Annotated[Database, Depends(get_database)]
ModelProviderDep = Annotated[ModelProvider, Depends(get_model_provider)]
PredictorDep = Annotated[Predictor, Depends(require_model)]
PredictionRepositoryDep = Annotated[PredictionRepository, Depends(get_prediction_repository)]
JobRepositoryDep = Annotated[JobRepository, Depends(get_job_repository)]
DriftRepositoryDep = Annotated[DriftRepository, Depends(get_drift_repository)]
PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
BatchServiceDep = Annotated[BatchScoringService, Depends(get_batch_service)]
DriftServiceDep = Annotated[DriftService, Depends(get_drift_service)]
NoteServiceDep = Annotated[MaintenanceNoteService, Depends(get_note_service)]
PaginationDep = Annotated[PaginationParams, Depends(PaginationParams)]
