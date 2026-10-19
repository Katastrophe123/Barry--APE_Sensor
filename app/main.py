"""FastAPI application factory and lifespan wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import auth, drift, health, insights, jobs, models, predictions
from app.config import Settings, get_settings
from app.db.session import Database
from app.llm.base import LLMClient
from app.llm.factory import build_llm_client
from app.logging_config import configure_logging, get_logger
from app.scheduler import ScheduledJobs, build_scheduler
from app.services.batch_service import BatchScoringService
from app.services.drift_service import DriftService
from app.services.insight_service import MaintenanceNoteService
from app.services.model_provider import ModelProvider
from app.services.prediction_service import PredictionService

logger = get_logger(__name__)

DESCRIPTION = """
Inference service for heavy-truck **APS (air pressure system) failure
prediction**.

Single-record and CSV-batch scoring, with every prediction stored alongside the
**model version and decision threshold that produced it** - so a disputed
prediction can be audited rather than guessed at. A nightly job scores dropped
CSVs and compares incoming feature distributions with the training data,
flagging drift via PSI and missing-rate shifts.

Model selection uses the **official Scania cost metric** (10x false positive +
500x false negative), not accuracy: a missed failure costs 50 times a needless
workshop check.

All endpoints except `/health` and `/auth/token` require a bearer token from
`POST /auth/token`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build process-wide resources once and tear them down on shutdown."""
    settings: Settings = app.state.settings
    database = Database.from_settings(settings)
    llm: LLMClient = build_llm_client(settings)

    model_provider = ModelProvider(
        settings.artifacts_dir,
        drift_kwargs={
            "psi_significant": settings.drift_psi_significant,
            "psi_critical": settings.drift_psi_critical,
            "drift_share_threshold": settings.drift_share_threshold,
            "min_rows": settings.drift_min_rows,
        },
    )
    # A missing model must not stop the service booting; /health reports it and
    # prediction endpoints return 503 until an artifact appears.
    await model_provider.load()

    batch_service = BatchScoringService(
        models=model_provider,
        database=database,
        max_rows=settings.max_batch_rows,
        chunk_size=settings.batch_chunk_size,
    )
    drift_service = DriftService(models=model_provider, min_rows=settings.drift_min_rows)

    app.state.database = database
    app.state.llm = llm
    app.state.models = model_provider
    app.state.prediction_service = PredictionService(models=model_provider)
    app.state.batch_service = batch_service
    app.state.drift_service = drift_service
    app.state.note_service = MaintenanceNoteService(
        llm=llm,
        max_attempts=settings.llm_max_attempts,
        backoff_base_seconds=settings.llm_backoff_base_seconds,
    )

    scheduler = None
    if settings.scheduler_enabled:
        scheduled = ScheduledJobs(
            settings=settings,
            database=database,
            models=model_provider,
            batch_service=batch_service,
            drift_service=drift_service,
        )
        scheduler = build_scheduler(scheduled, settings)
        scheduler.start()
    app.state.scheduler = scheduler

    logger.info(
        "application_started",
        app=settings.app_name,
        version=__version__,
        environment=settings.environment,
        model_loaded=model_provider.loaded,
        model_version=model_provider.version,
        llm_provider=settings.llm_provider,
        scheduler_enabled=settings.scheduler_enabled,
    )
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)
        await llm.aclose()
        await database.dispose()
        logger.info("application_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app. Tests call this with overridden settings."""
    resolved = settings or get_settings()
    configure_logging(level=resolved.log_level, json_output=resolved.log_json)

    app = FastAPI(
        title="APS Failure Prediction Service",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = resolved

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(predictions.router)
    app.include_router(predictions.history_router)
    app.include_router(jobs.router)
    app.include_router(models.router)
    app.include_router(drift.router)
    app.include_router(insights.router)

    return app


app = create_app()
