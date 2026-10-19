"""HTTP-level tests: auth, status codes, error envelopes, filters, 503 paths."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.errors import REQUEST_ID_HEADER
from app.config import Settings
from app.db.session import Database
from app.ml.sampledata import generate_aps_like_frame
from app.models.enums import JobStatus
from tests.conftest import TEST_PASSWORD, TEST_USERNAME


async def poll_job(
    client: AsyncClient, headers: dict[str, str], job_id: str, *, attempts: int = 200
) -> dict[str, Any]:
    """Poll until the job reaches a terminal state."""
    for _ in range(attempts):
        payload = (await client.get(f"/jobs/{job_id}", headers=headers)).json()
        if payload["status"] in {JobStatus.SUCCEEDED.value, JobStatus.FAILED.value}:
            return payload
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


class TestHealth:
    async def test_health_needs_no_token(self, client: AsyncClient) -> None:
        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["database"] == "up"
        assert body["model_loaded"] is True
        assert body["model_version"]
        assert body["scheduler_running"] is False

    async def test_health_reports_degraded_when_the_database_is_down(self, app: FastAPI) -> None:
        class DeadDatabase:
            async def healthcheck(self) -> bool:
                return False

        app.state.database = DeadDatabase()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as http:
            response = await http.get("/health")

        assert response.status_code == 503
        assert response.json()["database"] == "down"

    async def test_a_missing_model_is_reported_but_still_healthy(
        self, app_without_model: FastAPI
    ) -> None:
        """An untrained deployment is alive; only the DB being down is a 503."""
        async with AsyncClient(
            transport=ASGITransport(app=app_without_model), base_url="http://testserver"
        ) as http:
            response = await http.get("/health")

        assert response.status_code == 200
        assert response.json()["model_loaded"] is False
        assert response.json()["model_version"] is None

    async def test_response_carries_a_request_id(self, client: AsyncClient) -> None:
        assert (await client.get("/health")).headers[REQUEST_ID_HEADER]

    async def test_inbound_request_id_is_echoed(self, client: AsyncClient) -> None:
        response = await client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc"})
        assert response.headers[REQUEST_ID_HEADER] == "trace-abc"


class TestAuth:
    async def test_valid_credentials_return_a_token(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/token", json={"username": TEST_USERNAME, "password": TEST_PASSWORD}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == 1800

    async def test_bad_credentials_return_401(self, client: AsyncClient) -> None:
        response = await client.post(
            "/auth/token", json={"username": TEST_USERNAME, "password": "wrong"}
        )

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"
        assert response.headers["WWW-Authenticate"] == "Bearer"

    async def test_missing_fields_return_422(self, client: AsyncClient) -> None:
        response = await client.post("/auth/token", json={"username": TEST_USERNAME})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_an_issued_token_opens_a_protected_route(self, client: AsyncClient) -> None:
        token = (
            await client.post(
                "/auth/token",
                json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
            )
        ).json()["access_token"]

        response = await client.get("/predictions", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200


class TestAuthEnforcement:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("post", "/predict"),
            ("get", "/predictions"),
            ("get", "/jobs"),
            ("get", "/models/current"),
            ("post", "/models/reload"),
            ("post", "/drift/check"),
            ("get", "/drift"),
            ("post", "/insights/maintenance-note"),
        ],
    )
    async def test_protected_routes_reject_missing_tokens(
        self, client: AsyncClient, method: str, path: str
    ) -> None:
        kwargs = {"json": {}} if method == "post" else {}
        response = await getattr(client, method)(path, **kwargs)

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    async def test_rejects_a_malformed_token(self, client: AsyncClient) -> None:
        response = await client.get("/predictions", headers={"Authorization": "Bearer junk"})
        assert response.status_code == 401

    async def test_rejects_a_non_bearer_scheme(self, client: AsyncClient) -> None:
        response = await client.get("/predictions", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert response.status_code == 401


class TestPredict:
    async def test_returns_201_with_the_model_version(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        sample_features: dict[str, float | None],
    ) -> None:
        response = await client.post(
            "/predict",
            json={"vehicle_id": "TRK-7", "features": sample_features},
            headers=auth_headers,
        )

        assert response.status_code == 201
        body = response.json()
        assert body["vehicle_id"] == "TRK-7"
        assert 0.0 <= body["failure_probability"] <= 1.0
        assert body["risk_band"] in {"low", "medium", "high"}
        assert body["model_version"]
        assert body["decision_threshold"] > 0
        assert body["source"] == "api"
        assert uuid.UUID(body["id"])

    async def test_the_prediction_is_retrievable(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        sample_features: dict[str, float | None],
    ) -> None:
        created = await client.post(
            "/predict", json={"features": sample_features}, headers=auth_headers
        )
        prediction_id = created.json()["id"]

        fetched = await client.get(f"/predictions/{prediction_id}", headers=auth_headers)

        assert fetched.status_code == 200
        assert fetched.json()["id"] == prediction_id

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"features": {}},
            {"features": {"aa_000": "not a number"}},
            {"features": {"": 1.0}},
            {"features": {"aa_000": 1.0}, "vehicle_id": "x" * 100},
        ],
    )
    async def test_invalid_payloads_return_422(
        self, client: AsyncClient, auth_headers: dict[str, str], payload: dict[str, Any]
    ) -> None:
        response = await client.post("/predict", json=payload, headers=auth_headers)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_too_many_features_is_rejected(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            "/predict",
            json={"features": {f"f{index}": 1.0 for index in range(500)}},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_null_feature_values_are_accepted(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        sample_features: dict[str, float | None],
    ) -> None:
        nulled = dict.fromkeys(sample_features)
        response = await client.post("/predict", json={"features": nulled}, headers=auth_headers)
        assert response.status_code == 201

    async def test_unknown_features_are_reported_back(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        sample_features: dict[str, float | None],
    ) -> None:
        response = await client.post(
            "/predict",
            json={"features": {**sample_features, "not_a_sensor": 1.0}},
            headers=auth_headers,
        )
        assert response.json()["unexpected_features"] == ["not_a_sensor"]

    async def test_without_a_model_it_returns_503(
        self,
        app_without_model: FastAPI,
        auth_headers: dict[str, str],
        sample_features: dict[str, float | None],
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app_without_model), base_url="http://testserver"
        ) as http:
            response = await http.post(
                "/predict", json={"features": sample_features}, headers=auth_headers
            )

        assert response.status_code == 503
        body = response.json()
        assert body["error"]["code"] == "service_unavailable"
        assert "scripts/train.py" in body["error"]["message"]


class TestPredictionHistory:
    @pytest.fixture
    async def seeded(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        training_frame: pd.DataFrame,
    ) -> None:
        frame = training_frame.drop(columns=["class"]).head(6)
        for index, (_, row) in enumerate(frame.iterrows()):
            features = {
                key: (None if value == "na" else float(value))
                for key, value in row.to_dict().items()
            }
            response = await client.post(
                "/predict",
                json={"vehicle_id": f"TRK-{index}", "features": features},
                headers=auth_headers,
            )
            assert response.status_code == 201

    async def test_empty_history(self, client: AsyncClient, auth_headers: dict[str, str]) -> None:
        body = (await client.get("/predictions", headers=auth_headers)).json()
        assert body == {"items": [], "page": 1, "page_size": 20, "total": 0, "total_pages": 0}

    async def test_lists_newest_first(
        self, client: AsyncClient, auth_headers: dict[str, str], seeded: None
    ) -> None:
        body = (await client.get("/predictions", headers=auth_headers)).json()
        assert body["total"] == 6
        timestamps = [item["created_at"] for item in body["items"]]
        assert timestamps == sorted(timestamps, reverse=True)

    async def test_filters_by_vehicle(
        self, client: AsyncClient, auth_headers: dict[str, str], seeded: None
    ) -> None:
        body = (await client.get("/predictions?vehicle_id=TRK-3", headers=auth_headers)).json()
        assert body["total"] == 1
        assert body["items"][0]["vehicle_id"] == "TRK-3"

    async def test_filters_by_source(
        self, client: AsyncClient, auth_headers: dict[str, str], seeded: None
    ) -> None:
        api_total = (await client.get("/predictions?source=api", headers=auth_headers)).json()[
            "total"
        ]
        batch_total = (await client.get("/predictions?source=batch", headers=auth_headers)).json()[
            "total"
        ]
        assert api_total == 6
        assert batch_total == 0

    async def test_filters_by_risk_band(
        self, client: AsyncClient, auth_headers: dict[str, str], seeded: None
    ) -> None:
        bands = ["low", "medium", "high"]
        totals = [
            (await client.get(f"/predictions?risk_band={band}", headers=auth_headers)).json()[
                "total"
            ]
            for band in bands
        ]
        assert sum(totals) == 6

    async def test_filters_by_predicted_failure(
        self, client: AsyncClient, auth_headers: dict[str, str], seeded: None
    ) -> None:
        flagged = (
            await client.get("/predictions?predicted_failure=true", headers=auth_headers)
        ).json()["total"]
        clear = (
            await client.get("/predictions?predicted_failure=false", headers=auth_headers)
        ).json()["total"]
        assert flagged + clear == 6

    async def test_pagination_metadata(
        self, client: AsyncClient, auth_headers: dict[str, str], seeded: None
    ) -> None:
        body = (await client.get("/predictions?page=2&page_size=4", headers=auth_headers)).json()
        assert body["page"] == 2
        assert body["total"] == 6
        assert body["total_pages"] == 2
        assert len(body["items"]) == 2

    @pytest.mark.parametrize(
        "query", ["page=0", "page_size=0", "page_size=500", "risk_band=nonsense", "source=nope"]
    )
    async def test_invalid_query_parameters_return_422(
        self, client: AsyncClient, auth_headers: dict[str, str], query: str
    ) -> None:
        response = await client.get(f"/predictions?{query}", headers=auth_headers)
        assert response.status_code == 422

    async def test_unknown_id_returns_404(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        missing = uuid.uuid4()
        response = await client.get(f"/predictions/{missing}", headers=auth_headers)

        assert response.status_code == 404
        assert response.json()["error"]["details"]["prediction_id"] == str(missing)

    async def test_non_uuid_id_returns_422(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.get("/predictions/not-a-uuid", headers=auth_headers)
        assert response.status_code == 422


class TestBatchEndpoint:
    async def test_accepts_a_csv_and_scores_it_in_the_background(
        self, client: AsyncClient, auth_headers: dict[str, str], sample_csv: bytes
    ) -> None:
        response = await client.post(
            "/predict/batch",
            files={"file": ("batch.csv", sample_csv, "text/csv")},
            headers=auth_headers,
        )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == JobStatus.QUEUED.value
        assert body["total_rows"] == 120
        assert body["model_version"]
        assert "/jobs/" in body["poll_url"]

        job = await poll_job(client, auth_headers, body["job_id"])
        assert job["status"] == JobStatus.SUCCEEDED.value
        assert job["scored_rows"] == 120
        assert job["failed_rows"] == 0
        assert job["duration_ms"] is not None

    async def test_batch_predictions_are_queryable(
        self, client: AsyncClient, auth_headers: dict[str, str], sample_csv: bytes
    ) -> None:
        accepted = await client.post(
            "/predict/batch",
            files={"file": ("batch.csv", sample_csv, "text/csv")},
            headers=auth_headers,
        )
        await poll_job(client, auth_headers, accepted.json()["job_id"])

        body = (
            await client.get("/predictions?source=batch&page_size=1", headers=auth_headers)
        ).json()
        assert body["total"] == 120
        assert body["items"][0]["source"] == "batch"

    async def test_a_csv_with_vehicle_ids_keeps_them(
        self, client: AsyncClient, auth_headers: dict[str, str], training_frame: pd.DataFrame
    ) -> None:
        frame = training_frame.drop(columns=["class"]).head(20).copy()
        frame.insert(0, "vehicle_id", [f"TRK-{index}" for index in range(len(frame))])

        accepted = await client.post(
            "/predict/batch",
            files={"file": ("ids.csv", frame.to_csv(index=False).encode(), "text/csv")},
            headers=auth_headers,
        )
        await poll_job(client, auth_headers, accepted.json()["job_id"])

        body = (await client.get("/predictions?vehicle_id=TRK-5", headers=auth_headers)).json()
        assert body["total"] == 1

    @pytest.mark.parametrize(
        ("content", "fragment"),
        [
            (b"   ", "empty"),
            (b"aa_000,ab_000\n", "no data rows"),
            (b"alpha,beta\n1,2\n", "match the current model"),
        ],
    )
    async def test_bad_uploads_return_400(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        content: bytes,
        fragment: str,
    ) -> None:
        response = await client.post(
            "/predict/batch",
            files={"file": ("bad.csv", content, "text/csv")},
            headers=auth_headers,
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "bad_request"
        assert fragment in response.json()["error"]["message"]

    async def test_an_oversized_upload_returns_413(
        self, app: FastAPI, settings: Settings, auth_headers: dict[str, str]
    ) -> None:
        app.state.settings = settings.model_copy(update={"max_upload_bytes": 10})
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as http:
            response = await http.post(
                "/predict/batch",
                files={"file": ("big.csv", b"aa_000\n" + b"1\n" * 100, "text/csv")},
                headers=auth_headers,
            )

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

    async def test_missing_file_returns_422(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post("/predict/batch", headers=auth_headers)
        assert response.status_code == 422

    async def test_without_a_model_it_returns_503(
        self, app_without_model: FastAPI, auth_headers: dict[str, str], sample_csv: bytes
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app_without_model), base_url="http://testserver"
        ) as http:
            response = await http.post(
                "/predict/batch",
                files={"file": ("batch.csv", sample_csv, "text/csv")},
                headers=auth_headers,
            )
        assert response.status_code == 503


class TestJobsEndpoint:
    async def test_unknown_job_returns_404(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        missing = uuid.uuid4()
        response = await client.get(f"/jobs/{missing}", headers=auth_headers)

        assert response.status_code == 404
        assert response.json()["error"]["details"]["job_id"] == str(missing)

    async def test_empty_job_list(self, client: AsyncClient, auth_headers: dict[str, str]) -> None:
        body = (await client.get("/jobs", headers=auth_headers)).json()
        assert body["total"] == 0
        assert body["total_pages"] == 0

    async def test_jobs_are_filterable(
        self, client: AsyncClient, auth_headers: dict[str, str], sample_csv: bytes
    ) -> None:
        accepted = await client.post(
            "/predict/batch",
            files={"file": ("batch.csv", sample_csv, "text/csv")},
            headers=auth_headers,
        )
        await poll_job(client, auth_headers, accepted.json()["job_id"])

        by_kind = (await client.get("/jobs?kind=batch_upload", headers=auth_headers)).json()
        by_status = (await client.get("/jobs?status=succeeded", headers=auth_headers)).json()
        other_status = (await client.get("/jobs?status=failed", headers=auth_headers)).json()

        assert by_kind["total"] == 1
        assert by_status["total"] == 1
        assert other_status["total"] == 0

    @pytest.mark.parametrize("query", ["status=nonsense", "kind=nope", "page=0"])
    async def test_invalid_filters_return_422(
        self, client: AsyncClient, auth_headers: dict[str, str], query: str
    ) -> None:
        response = await client.get(f"/jobs?{query}", headers=auth_headers)
        assert response.status_code == 422


class TestModelsEndpoint:
    async def test_current_model_reports_measured_metrics(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        body = (await client.get("/models/current", headers=auth_headers)).json()

        assert body["model_type"] == "random_forest"
        assert body["n_features"] > 0
        assert body["training_rows"] > 0
        assert 0.0 < body["decision_threshold"] <= 1.0
        assert "roc_auc" in body["metrics"]
        assert "confusion" in body["metrics"]
        # The v1 metrics are present but namespaced, not headline.
        assert "legacy_regression" in body["metrics"]
        assert body["feature_list"]
        assert body["available_versions"]

    async def test_reload_keeps_serving(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        before = (await client.get("/models/current", headers=auth_headers)).json()
        response = await client.post("/models/reload", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["version"] == before["version"]

    async def test_current_model_without_an_artifact_returns_503(
        self, app_without_model: FastAPI, auth_headers: dict[str, str]
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app_without_model), base_url="http://testserver"
        ) as http:
            response = await http.get("/models/current", headers=auth_headers)
        assert response.status_code == 503

    async def test_reload_without_an_artifact_returns_503(
        self, app_without_model: FastAPI, auth_headers: dict[str, str]
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app_without_model), base_url="http://testserver"
        ) as http:
            response = await http.post("/models/reload", headers=auth_headers)

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "service_unavailable"


class TestDriftEndpoint:
    async def test_a_check_with_no_traffic_is_inconclusive(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post("/drift/check", headers=auth_headers)

        assert response.status_code == 201
        body = response.json()
        assert body["outcome"] == "inconclusive"
        assert body["rows_analysed"] == 0

    async def test_checks_are_listed_and_fetchable(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        created = (await client.post("/drift/check", headers=auth_headers)).json()

        listed = (await client.get("/drift", headers=auth_headers)).json()
        fetched = await client.get(f"/drift/{created['id']}", headers=auth_headers)
        latest = await client.get("/drift/latest", headers=auth_headers)

        assert listed["total"] == 1
        assert fetched.status_code == 200
        assert latest.json()["id"] == created["id"]

    async def test_latest_without_any_check_returns_404(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.get("/drift/latest", headers=auth_headers)
        assert response.status_code == 404

    async def test_unknown_check_returns_404(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.get(f"/drift/{uuid.uuid4()}", headers=auth_headers)
        assert response.status_code == 404

    async def test_checks_filter_by_outcome(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        await client.post("/drift/check", headers=auth_headers)

        inconclusive = (
            await client.get("/drift?outcome=inconclusive", headers=auth_headers)
        ).json()
        clean = (await client.get("/drift?outcome=clean", headers=auth_headers)).json()

        assert inconclusive["total"] == 1
        assert clean["total"] == 0

    async def test_drifted_traffic_is_flagged_through_the_api(
        self, client: AsyncClient, auth_headers: dict[str, str], sample_csv: bytes
    ) -> None:
        """End to end: score shifted data, then ask for a drift check."""
        from app.ml.sampledata import shift_frame_for_drift

        frame = shift_frame_for_drift(generate_aps_like_frame(n_rows=250, seed=4242)).drop(
            columns=["class"]
        )
        accepted = await client.post(
            "/predict/batch",
            files={"file": ("drifted.csv", frame.to_csv(index=False).encode(), "text/csv")},
            headers=auth_headers,
        )
        await poll_job(client, auth_headers, accepted.json()["job_id"])

        body = (await client.post("/drift/check", headers=auth_headers)).json()

        assert body["outcome"] == "drift_detected"
        assert body["critical_features"]
        assert body["details"]["top_features"]

    async def test_without_a_model_it_returns_503(
        self, app_without_model: FastAPI, auth_headers: dict[str, str]
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app_without_model), base_url="http://testserver"
        ) as http:
            response = await http.post("/drift/check", headers=auth_headers)
        assert response.status_code == 503


class TestInsightsEndpoint:
    async def test_note_for_a_stored_prediction(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
        sample_features: dict[str, float | None],
    ) -> None:
        created = await client.post(
            "/predict",
            json={"vehicle_id": "TRK-2", "features": sample_features},
            headers=auth_headers,
        )
        prediction_id = created.json()["id"]

        response = await client.post(
            "/insights/maintenance-note",
            json={"prediction_id": prediction_id},
            headers=auth_headers,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["prediction_id"] == prediction_id
        assert body["vehicle_id"] == "TRK-2"
        assert body["generated_by"] in {"llm", "template"}
        assert len(body["note"]) > 20
        assert body["risk_band"] == created.json()["risk_band"]

    async def test_the_note_matches_the_risk_band(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """A low-risk vehicle must not be told to book a workshop visit."""
        response = await client.post(
            "/insights/maintenance-note",
            json={"failure_probability": 0.001, "vehicle_id": "TRK-X"},
            headers=auth_headers,
        )

        body = response.json()
        assert body["risk_band"] == "low"
        assert "no" in body["note"].lower()

    async def test_inline_mode_needs_no_stored_prediction(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            "/insights/maintenance-note",
            json={
                "failure_probability": 0.92,
                "vehicle_id": "TRK-9",
                "top_signals": ["aa_000 = 98765"],
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.json()["risk_band"] == "high"

    async def test_unknown_prediction_returns_404(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            "/insights/maintenance-note",
            json={"prediction_id": str(uuid.uuid4())},
            headers=auth_headers,
        )
        assert response.status_code == 404

    async def test_neither_source_supplied_returns_422(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post("/insights/maintenance-note", json={}, headers=auth_headers)
        assert response.status_code == 422

    async def test_a_failing_llm_still_returns_a_note(
        self,
        app: FastAPI,
        auth_headers: dict[str, str],
        fake_sleep: Any,
    ) -> None:
        """The LLM is peripheral: its outage must not fail the request."""
        from app.services.insight_service import MaintenanceNoteService
        from tests.test_services import FailingLLM

        app.state.note_service = MaintenanceNoteService(
            llm=FailingLLM(), max_attempts=1, sleep=fake_sleep
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as http:
            response = await http.post(
                "/insights/maintenance-note",
                json={"failure_probability": 0.95, "vehicle_id": "TRK-4"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["generated_by"] == "template"
        assert "TRK-4" in body["note"]

    async def test_works_without_a_loaded_model(
        self, app_without_model: FastAPI, auth_headers: dict[str, str]
    ) -> None:
        """Inline mode does not need a model, so it must not 503."""
        async with AsyncClient(
            transport=ASGITransport(app=app_without_model), base_url="http://testserver"
        ) as http:
            response = await http.post(
                "/insights/maintenance-note",
                json={"failure_probability": 0.9},
                headers=auth_headers,
            )

        assert response.status_code == 200
        assert response.json()["risk_band"] == "high"


class TestUnhandledErrors:
    async def test_an_unexpected_failure_returns_a_sanitised_500(
        self,
        app: FastAPI,
        auth_headers: dict[str, str],
        sample_features: dict[str, float | None],
    ) -> None:
        class BoomService:
            async def predict(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("secret internal detail")

        app.state.prediction_service = BoomService()
        # ServerErrorMiddleware re-raises after responding, so opt out.
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://testserver") as http:
            response = await http.post(
                "/predict", json={"features": sample_features}, headers=auth_headers
            )

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"
        assert "secret internal detail" not in response.text


class TestOpenAPI:
    async def test_schema_covers_every_route(self, client: AsyncClient) -> None:
        paths = set((await client.get("/openapi.json")).json()["paths"])

        assert {
            "/health",
            "/auth/token",
            "/predict",
            "/predict/batch",
            "/predictions",
            "/predictions/{prediction_id}",
            "/jobs",
            "/jobs/{job_id}",
            "/models/current",
            "/models/reload",
            "/drift/check",
            "/insights/maintenance-note",
        } <= paths

    async def test_protected_routes_declare_security(self, client: AsyncClient) -> None:
        schema = (await client.get("/openapi.json")).json()
        assert "security" in schema["paths"]["/predict"]["post"]
        assert "security" not in schema["paths"]["/health"]["get"]


class TestDatabaseIsolation:
    async def test_each_test_gets_a_fresh_database(
        self, client: AsyncClient, auth_headers: dict[str, str], database: Database
    ) -> None:
        """Guards against cross-test leakage through the in-memory database."""
        body = (await client.get("/predictions", headers=auth_headers)).json()
        assert body["total"] == 0
