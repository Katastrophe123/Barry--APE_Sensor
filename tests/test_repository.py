"""Repository tests against a real (in-memory) database.

These cover the auditability requirement: a prediction must be findable later,
with the model version that produced it.
"""

from __future__ import annotations

import uuid

import pytest

from app.db.repository import DriftRepository, JobRepository, PredictionRepository
from app.models.enums import (
    DriftOutcome,
    JobKind,
    JobStatus,
    PredictionSource,
    RiskBand,
)


async def record(
    repository: PredictionRepository,
    *,
    probability: float = 0.5,
    band: RiskBand = RiskBand.MEDIUM,
    version: str = "v1",
    vehicle_id: str | None = None,
    source: PredictionSource = PredictionSource.API,
    predicted: bool = True,
    **kwargs: object,
) -> object:
    return await repository.record(
        features={"aa_000": 1.0},
        failure_probability=probability,
        predicted_failure=predicted,
        risk_band=band,
        decision_threshold=0.04,
        model_version=version,
        vehicle_id=vehicle_id,
        source=source,
        **kwargs,  # type: ignore[arg-type]
    )


class TestPredictionRepository:
    async def test_a_prediction_round_trips(self, predictions: PredictionRepository) -> None:
        created = await record(
            predictions, probability=0.81, band=RiskBand.HIGH, vehicle_id="TRK-1"
        )
        fetched = await predictions.get(created.id)  # type: ignore[attr-defined]

        assert fetched is not None
        assert fetched.failure_probability == pytest.approx(0.81)
        assert fetched.risk_band == RiskBand.HIGH.value
        assert fetched.vehicle_id == "TRK-1"
        assert fetched.decision_threshold == pytest.approx(0.04)
        assert fetched.model_version == "v1"
        assert isinstance(fetched.id, uuid.UUID)

    async def test_unknown_id_returns_none(self, predictions: PredictionRepository) -> None:
        assert await predictions.get(uuid.uuid4()) is None

    async def test_optional_metadata_is_stored(self, predictions: PredictionRepository) -> None:
        created = await record(
            predictions,
            missing_features=["ab_000"],
            unexpected_features=["mystery"],
            request_metadata={"depot": "Malmo"},
            latency_ms=12,
        )
        fetched = await predictions.get(created.id)  # type: ignore[attr-defined]

        assert fetched is not None
        assert fetched.missing_features == ["ab_000"]
        assert fetched.unexpected_features == ["mystery"]
        assert fetched.request_metadata == {"depot": "Malmo"}
        assert fetched.latency_ms == 12

    async def test_bulk_insert_counts_rows(self, predictions: PredictionRepository) -> None:
        from app.db.models import Prediction

        rows = [
            Prediction(
                features={},
                failure_probability=0.1,
                predicted_failure=False,
                risk_band=RiskBand.LOW.value,
                decision_threshold=0.04,
                model_version="v1",
                source=PredictionSource.BATCH.value,
                missing_features=[],
                unexpected_features=[],
                request_metadata={},
            )
            for _ in range(5)
        ]
        assert await predictions.record_many(rows) == 5

    async def test_bulk_insert_of_nothing_is_a_noop(
        self, predictions: PredictionRepository
    ) -> None:
        assert await predictions.record_many([]) == 0

    async def test_filters_combine(self, predictions: PredictionRepository) -> None:
        await record(predictions, band=RiskBand.HIGH, version="v1", vehicle_id="A")
        await record(predictions, band=RiskBand.LOW, version="v1", vehicle_id="B")
        await record(predictions, band=RiskBand.HIGH, version="v2", vehicle_id="A")

        _, high = await predictions.list_predictions(risk_band=RiskBand.HIGH)
        _, v1 = await predictions.list_predictions(model_version="v1")
        _, high_v1 = await predictions.list_predictions(risk_band=RiskBand.HIGH, model_version="v1")
        _, vehicle_a = await predictions.list_predictions(vehicle_id="A")

        assert (high, v1, high_v1, vehicle_a) == (2, 2, 1, 2)

    async def test_filters_by_source_and_flag(self, predictions: PredictionRepository) -> None:
        await record(predictions, source=PredictionSource.API, predicted=True)
        await record(predictions, source=PredictionSource.BATCH, predicted=False)

        _, api = await predictions.list_predictions(source=PredictionSource.API)
        _, flagged = await predictions.list_predictions(predicted_failure=True)
        _, clear = await predictions.list_predictions(predicted_failure=False)

        assert (api, flagged, clear) == (1, 1, 1)

    async def test_pagination_splits_the_result_set(
        self, predictions: PredictionRepository
    ) -> None:
        for index in range(5):
            await record(predictions, vehicle_id=f"TRK-{index}")

        first, total = await predictions.list_predictions(page=1, page_size=2)
        second, _ = await predictions.list_predictions(page=2, page_size=2)
        third, _ = await predictions.list_predictions(page=3, page_size=2)

        assert total == 5
        assert (len(first), len(second), len(third)) == (2, 2, 1)
        ids = {row.id for row in first} | {row.id for row in second} | {row.id for row in third}
        assert len(ids) == 5

    async def test_page_beyond_the_end_is_empty(self, predictions: PredictionRepository) -> None:
        await record(predictions)
        rows, total = await predictions.list_predictions(page=99)
        assert rows == []
        assert total == 1

    async def test_results_are_newest_first(self, predictions: PredictionRepository) -> None:
        for index in range(3):
            await record(predictions, vehicle_id=f"TRK-{index}")
        rows, _ = await predictions.list_predictions()
        timestamps = [row.created_at for row in rows]
        assert timestamps == sorted(timestamps, reverse=True)

    async def test_recent_features_feeds_the_drift_checker(
        self, predictions: PredictionRepository
    ) -> None:
        for _ in range(3):
            await record(predictions)
        payloads = await predictions.recent_features(limit=2)

        assert len(payloads) == 2
        assert payloads[0] == {"aa_000": 1.0}

    async def test_recent_features_on_an_empty_table(
        self, predictions: PredictionRepository
    ) -> None:
        assert await predictions.recent_features() == []

    async def test_count_by_risk_band(self, predictions: PredictionRepository) -> None:
        await record(predictions, band=RiskBand.HIGH)
        await record(predictions, band=RiskBand.HIGH)
        await record(predictions, band=RiskBand.LOW)

        assert await predictions.count_by_risk_band() == {"high": 2, "low": 1}


class TestJobRepository:
    async def test_a_new_job_starts_queued(self, jobs: JobRepository) -> None:
        job = await jobs.create(
            kind=JobKind.BATCH_UPLOAD,
            source_filename="a.csv",
            model_version="v1",
            total_rows=10,
        )

        assert job.status == JobStatus.QUEUED.value
        assert job.started_at is None
        assert job.finished_at is None
        assert job.total_rows == 10

    async def test_unknown_job_returns_none(self, jobs: JobRepository) -> None:
        assert await jobs.get(uuid.uuid4()) is None

    async def test_the_lifecycle_sets_timestamps(self, jobs: JobRepository) -> None:
        job = await jobs.create(kind=JobKind.BATCH_UPLOAD)

        await jobs.mark_running(job)
        assert job.status == JobStatus.RUNNING.value
        assert job.started_at is not None

        await jobs.mark_succeeded(
            job, scored_rows=9, failed_rows=1, high_risk_rows=2, duration_ms=50
        )
        assert job.status == JobStatus.SUCCEEDED.value
        assert job.finished_at is not None
        assert (job.scored_rows, job.failed_rows, job.high_risk_rows) == (9, 1, 2)

    async def test_failure_records_the_error(self, jobs: JobRepository) -> None:
        job = await jobs.create(kind=JobKind.DRIFT_CHECK)
        await jobs.mark_failed(job, error="boom", duration_ms=5)

        assert job.status == JobStatus.FAILED.value
        assert job.error == "boom"
        assert job.finished_at is not None

    async def test_long_errors_are_truncated(self, jobs: JobRepository) -> None:
        job = await jobs.create(kind=JobKind.DRIFT_CHECK)
        await jobs.mark_failed(job, error="x" * 9000)
        assert job.error is not None
        assert len(job.error) == 4000

    async def test_filters_and_pagination(self, jobs: JobRepository) -> None:
        for kind in (JobKind.BATCH_UPLOAD, JobKind.DRIFT_CHECK, JobKind.BATCH_UPLOAD):
            await jobs.create(kind=kind)
        done = await jobs.create(kind=JobKind.BATCH_UPLOAD)
        await jobs.mark_succeeded(done, scored_rows=1)

        _, uploads = await jobs.list_jobs(kind=JobKind.BATCH_UPLOAD)
        _, succeeded = await jobs.list_jobs(status=JobStatus.SUCCEEDED)
        page, total = await jobs.list_jobs(page=1, page_size=2)

        assert uploads == 3
        assert succeeded == 1
        assert total == 4
        assert len(page) == 2

    async def test_jobs_are_newest_first(self, jobs: JobRepository) -> None:
        for _ in range(3):
            await jobs.create(kind=JobKind.BATCH_UPLOAD)
        rows, _ = await jobs.list_jobs()
        timestamps = [row.created_at for row in rows]
        assert timestamps == sorted(timestamps, reverse=True)


class TestDriftRepository:
    async def test_a_check_round_trips(self, drifts: DriftRepository) -> None:
        created = await drifts.record(
            model_version="v1",
            outcome=DriftOutcome.DRIFT_DETECTED,
            rows_analysed=500,
            features_analysed=35,
            drift_share=0.2,
            max_psi=3.1,
            drifted_features=["aa_000"],
            critical_features=["aa_000"],
            details={"top_features": []},
            note="",
        )
        fetched = await drifts.get(created.id)

        assert fetched is not None
        assert fetched.outcome == DriftOutcome.DRIFT_DETECTED.value
        assert fetched.drifted_features == ["aa_000"]
        assert fetched.critical_features == ["aa_000"]
        assert fetched.max_psi == pytest.approx(3.1)

    async def test_unknown_check_returns_none(self, drifts: DriftRepository) -> None:
        assert await drifts.get(uuid.uuid4()) is None

    async def test_latest_returns_the_most_recent(self, drifts: DriftRepository) -> None:
        for index in range(3):
            await drifts.record(
                model_version=f"v{index}",
                outcome=DriftOutcome.CLEAN,
                rows_analysed=500,
                features_analysed=10,
                drift_share=0.0,
                max_psi=0.0,
            )
        latest = await drifts.latest()
        assert latest is not None
        assert latest.model_version == "v2"

    async def test_latest_can_be_scoped_to_a_model_version(self, drifts: DriftRepository) -> None:
        for version in ("v1", "v2"):
            await drifts.record(
                model_version=version,
                outcome=DriftOutcome.CLEAN,
                rows_analysed=500,
                features_analysed=10,
                drift_share=0.0,
                max_psi=0.0,
            )
        scoped = await drifts.latest(model_version="v1")
        assert scoped is not None
        assert scoped.model_version == "v1"

    async def test_latest_on_an_empty_table(self, drifts: DriftRepository) -> None:
        assert await drifts.latest() is None

    async def test_filters_by_outcome(self, drifts: DriftRepository) -> None:
        for outcome in (
            DriftOutcome.CLEAN,
            DriftOutcome.DRIFT_DETECTED,
            DriftOutcome.CLEAN,
        ):
            await drifts.record(
                model_version="v1",
                outcome=outcome,
                rows_analysed=500,
                features_analysed=10,
                drift_share=0.0,
                max_psi=0.0,
            )

        _, clean = await drifts.list_checks(outcome=DriftOutcome.CLEAN)
        _, drifted = await drifts.list_checks(outcome=DriftOutcome.DRIFT_DETECTED)
        _, total = await drifts.list_checks()

        assert (clean, drifted, total) == (2, 1, 3)
