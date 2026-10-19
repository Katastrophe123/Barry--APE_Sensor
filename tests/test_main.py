"""Tests for application assembly and the lifespan-managed resources."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.db.session import Database
from app.llm.base import LLMPermanentError
from app.llm.providers import StubLLMClient
from app.main import create_app
from app.services.model_provider import ModelProvider


class TestCreateApp:
    def test_every_route_is_registered(self, settings: Settings) -> None:
        paths = set(create_app(settings).openapi()["paths"])

        assert {
            "/health",
            "/auth/token",
            "/predict",
            "/predict/batch",
            "/predictions",
            "/jobs",
            "/models/current",
            "/models/reload",
            "/drift/check",
            "/drift/latest",
            "/insights/maintenance-note",
        } <= paths

    def test_settings_are_stored_on_app_state(self, settings: Settings) -> None:
        app = create_app(settings)
        assert app.state.settings is settings


class TestLifespan:
    async def test_lifespan_wires_every_resource(self, settings: Settings) -> None:
        """Runs the real lifespan against SQLite and the session artifact."""
        app = create_app(settings)

        async with app.router.lifespan_context(app):
            assert isinstance(app.state.database, Database)
            assert isinstance(app.state.llm, StubLLMClient)
            assert isinstance(app.state.models, ModelProvider)
            assert app.state.models.loaded is True
            assert app.state.prediction_service is not None
            assert app.state.batch_service is not None
            assert app.state.drift_service is not None
            assert app.state.note_service is not None
            # Disabled by default, so no surprise background writes.
            assert app.state.scheduler is None
            assert await app.state.database.healthcheck() is True

    async def test_booting_without_a_model_is_survivable(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """A fresh deployment has no artifact yet; it must still start."""
        untrained = settings.model_copy(update={"artifacts_dir": tmp_path / "empty"})
        app = create_app(untrained)

        async with app.router.lifespan_context(app):
            assert app.state.models.loaded is False
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as http:
                response = await http.get("/health")

        assert response.status_code == 200
        assert response.json()["model_loaded"] is False

    async def test_a_networked_llm_without_a_key_fails_fast(self, settings: Settings) -> None:
        broken = settings.model_copy(update={"llm_provider": "openai", "llm_api_key": None})
        app = create_app(broken)

        with pytest.raises(LLMPermanentError, match="LLM_API_KEY"):
            async with app.router.lifespan_context(app):
                pass  # pragma: no cover - the context must not open

    async def test_the_scheduler_starts_when_enabled(self, settings: Settings) -> None:
        enabled = settings.model_copy(
            update={
                "scheduler_enabled": True,
                "scheduled_scoring_cron": "0 2 * * *",
                "scheduled_drift_cron": "30 2 * * *",
            }
        )
        app = create_app(enabled)

        async with app.router.lifespan_context(app):
            scheduler = app.state.scheduler
            assert scheduler is not None
            assert scheduler.running is True
            assert {job.id for job in scheduler.get_jobs()} == {
                "scheduled_scoring",
                "scheduled_drift",
            }
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as http:
                assert (await http.get("/health")).json()["scheduler_running"] is True

        # Shutdown must stop it, or the process will not exit.
        assert scheduler.running is False

    async def test_health_reports_the_model_through_the_real_lifespan(
        self, settings: Settings
    ) -> None:
        app = create_app(settings)

        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as http,
        ):
            response = await http.get("/health")

        body = response.json()
        assert body["model_loaded"] is True
        assert body["model_version"]
        assert body["llm_provider"] == "stub"
