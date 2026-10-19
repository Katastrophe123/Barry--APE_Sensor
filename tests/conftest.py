"""Shared fixtures.

The whole suite runs offline: SQLite in-memory for persistence, a small
synthetic dataset, and a stubbed LLM. No API key, no PostgreSQL, no network.

The trained model is built **once per session** and reused: fitting a
RandomForest plus an XGBoost per test would dominate the runtime, and the
artifact is read-only as far as the tests are concerned.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.core.security import create_access_token
from app.db.base import Base
from app.db.repository import DriftRepository, JobRepository, PredictionRepository
from app.db.session import Database
from app.llm.providers import StubLLMClient
from app.main import create_app
from app.ml.predictor import Predictor
from app.ml.registry import ModelMetadata, ModelRegistry
from app.ml.sampledata import generate_aps_like_frame
from app.ml.training import ModelTrainer, _random_forest
from app.services.batch_service import BatchScoringService
from app.services.drift_service import DriftService
from app.services.insight_service import MaintenanceNoteService
from app.services.model_provider import ModelProvider
from app.services.prediction_service import PredictionService

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_USERNAME = "operator"
TEST_PASSWORD = "test-password"

#: Small enough to train fast, large enough that the stratified split keeps
#: positives on both sides.
TRAIN_ROWS = 2_000


# --------------------------------------------------------------------------- #
# Session-scoped model artifact
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def training_frame() -> pd.DataFrame:
    """A deterministic synthetic training frame."""
    return generate_aps_like_frame(n_rows=TRAIN_ROWS, seed=7)


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory: pytest.TempPathFactory, training_frame: pd.DataFrame) -> Path:
    """Train once and save the artifact for the whole session.

    RandomForest only: the point of the tests is the service around the model,
    and training XGBoost as well would double the suite's runtime for no extra
    coverage. ``tests/test_training.py`` covers multi-candidate selection with
    stub estimators.
    """
    directory = tmp_path_factory.mktemp("artifacts")
    result = ModelTrainer(model_factories={"random_forest": _random_forest}).train(training_frame)
    ModelRegistry(directory).save(result, dataset_fingerprint="test-fixture")
    return directory


@pytest.fixture(scope="session")
def registry(artifacts_dir: Path) -> ModelRegistry:
    return ModelRegistry(artifacts_dir)


@pytest.fixture(scope="session")
def model_metadata(registry: ModelRegistry) -> ModelMetadata:
    version = registry.current_version()
    assert version is not None
    return registry.load_metadata(version)


@pytest.fixture(scope="session")
def predictor(registry: ModelRegistry) -> Predictor:
    return Predictor(registry.load_current())


@pytest.fixture
def sample_features(training_frame: pd.DataFrame) -> dict[str, float | None]:
    """One record from the training frame, in API shape."""
    row = training_frame.drop(columns=["class"]).iloc[0].to_dict()
    return {key: (None if value == "na" else float(value)) for key, value in row.items()}


@pytest.fixture
def sample_csv(training_frame: pd.DataFrame) -> bytes:
    """A small unlabelled CSV upload."""
    return training_frame.drop(columns=["class"]).head(120).to_csv(index=False).encode()


# --------------------------------------------------------------------------- #
# Settings / database
# --------------------------------------------------------------------------- #
@pytest.fixture
def settings(artifacts_dir: Path, tmp_path: Path) -> Settings:
    """Deterministic settings: no backoff, stub LLM, scheduler off."""
    return Settings(
        environment="test",
        log_level="WARNING",
        log_json=False,
        database_url=TEST_DATABASE_URL,
        jwt_secret="test-secret-value-for-unit-tests-only",
        jwt_expire_minutes=30,
        auth_username=TEST_USERNAME,
        auth_password=TEST_PASSWORD,
        artifacts_dir=artifacts_dir,
        llm_provider="stub",
        llm_max_attempts=2,
        llm_backoff_base_seconds=0.0,
        scheduler_enabled=False,
        scheduled_input_dir=tmp_path / "incoming",
        scheduled_output_dir=tmp_path / "scored",
        batch_chunk_size=50,
        # The real floor: tests must not rely on a laxer threshold than prod.
        drift_min_rows=200,
    )


@pytest.fixture
async def database(settings: Settings) -> AsyncIterator[Database]:
    """In-memory database with the schema created from the ORM metadata."""
    db = Database(TEST_DATABASE_URL, echo=False)
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield db
    finally:
        await db.dispose()


@pytest.fixture
async def session(database: Database) -> AsyncIterator[Any]:
    async with database.session() as sess:
        yield sess


@pytest.fixture
async def predictions(session: Any) -> PredictionRepository:
    return PredictionRepository(session)


@pytest.fixture
async def jobs(session: Any) -> JobRepository:
    return JobRepository(session)


@pytest.fixture
async def drifts(session: Any) -> DriftRepository:
    return DriftRepository(session)


# --------------------------------------------------------------------------- #
# Services
# --------------------------------------------------------------------------- #
@pytest.fixture
def sleep_calls() -> list[float]:
    return []


@pytest.fixture
def fake_sleep(sleep_calls: list[float]) -> Any:
    async def _sleep(delay: float) -> None:
        sleep_calls.append(delay)

    return _sleep


@pytest.fixture
async def models(settings: Settings) -> ModelProvider:
    """A loaded model provider pointing at the session artifact."""
    provider = ModelProvider(
        settings.artifacts_dir,
        drift_kwargs={
            "psi_significant": settings.drift_psi_significant,
            "psi_critical": settings.drift_psi_critical,
            "drift_share_threshold": settings.drift_share_threshold,
            "min_rows": settings.drift_min_rows,
        },
    )
    await provider.load()
    return provider


@pytest.fixture
def empty_models(tmp_path: Path) -> ModelProvider:
    """A provider with no artifact, for the 503 paths."""
    return ModelProvider(tmp_path / "no-artifacts")


@pytest.fixture
def prediction_service(models: ModelProvider) -> PredictionService:
    return PredictionService(models=models)


@pytest.fixture
def batch_service(
    models: ModelProvider, database: Database, settings: Settings
) -> BatchScoringService:
    return BatchScoringService(
        models=models,
        database=database,
        max_rows=settings.max_batch_rows,
        chunk_size=settings.batch_chunk_size,
    )


@pytest.fixture
def drift_service(models: ModelProvider, settings: Settings) -> DriftService:
    return DriftService(models=models, min_rows=settings.drift_min_rows)


@pytest.fixture
def note_service(fake_sleep: Any) -> MaintenanceNoteService:
    return MaintenanceNoteService(
        llm=StubLLMClient(), max_attempts=2, backoff_base_seconds=0.0, sleep=fake_sleep
    )


# --------------------------------------------------------------------------- #
# App / client
# --------------------------------------------------------------------------- #
@pytest.fixture
def app(
    settings: Settings,
    database: Database,
    models: ModelProvider,
    batch_service: BatchScoringService,
    drift_service: DriftService,
    note_service: MaintenanceNoteService,
) -> Iterator[FastAPI]:
    """App instance wired to the test database and the session artifact.

    ``create_app`` is used so routing, middleware and error handlers are the
    real ones; only the lifespan-managed resources are substituted.
    """
    application = create_app(settings)
    application.state.database = database
    application.state.llm = StubLLMClient()
    application.state.models = models
    application.state.prediction_service = PredictionService(models=models)
    application.state.batch_service = batch_service
    application.state.drift_service = drift_service
    application.state.note_service = note_service
    application.state.scheduler = None
    yield application


@pytest.fixture
def app_without_model(app: FastAPI, empty_models: ModelProvider) -> FastAPI:
    """The same app, but with no trained model loaded."""
    app.state.models = empty_models
    app.state.prediction_service = PredictionService(models=empty_models)
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http


@pytest.fixture
def auth_headers(settings: Settings) -> dict[str, str]:
    token, _ = create_access_token(TEST_USERNAME, settings)
    return {"Authorization": f"Bearer {token}"}
