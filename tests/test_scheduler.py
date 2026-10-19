"""Tests for the scheduled automation."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.config import Settings
from app.db.repository import JobRepository, PredictionRepository
from app.db.session import Database
from app.ml.sampledata import generate_aps_like_frame, shift_frame_for_drift
from app.models.enums import DriftOutcome, JobKind, JobStatus, PredictionSource
from app.scheduler import ScheduledJobs, build_scheduler, parse_cron
from app.services.batch_service import BatchScoringService
from app.services.drift_service import DriftService
from app.services.model_provider import ModelProvider


@pytest.fixture
def scheduled(
    settings: Settings,
    database: Database,
    models: ModelProvider,
    batch_service: BatchScoringService,
    drift_service: DriftService,
) -> ScheduledJobs:
    settings.scheduled_input_dir.mkdir(parents=True, exist_ok=True)
    return ScheduledJobs(
        settings=settings,
        database=database,
        models=models,
        batch_service=batch_service,
        drift_service=drift_service,
    )


@pytest.fixture
def scheduled_without_model(
    settings: Settings,
    database: Database,
    empty_models: ModelProvider,
    batch_service: BatchScoringService,
    drift_service: DriftService,
) -> ScheduledJobs:
    return ScheduledJobs(
        settings=settings,
        database=database,
        models=empty_models,
        batch_service=batch_service,
        drift_service=drift_service,
    )


class TestParseCron:
    def test_parses_a_five_field_expression(self) -> None:
        trigger = parse_cron("0 2 * * *")
        assert str(trigger)

    def test_parses_a_step_expression(self) -> None:
        assert str(parse_cron("*/5 * * * *"))

    @pytest.mark.parametrize("expression", ["0 2 * *", "", "0 2 * * * *", "every night"])
    def test_rejects_anything_else(self, expression: str) -> None:
        with pytest.raises(ValueError, match="5-field cron"):
            parse_cron(expression)


class TestBuildScheduler:
    def test_registers_both_jobs(self, scheduled: ScheduledJobs, settings: Settings) -> None:
        scheduler = build_scheduler(scheduled, settings)
        ids = {job.id for job in scheduler.get_jobs()}
        assert ids == {"scheduled_scoring", "scheduled_drift"}

    def test_jobs_do_not_overlap_themselves(
        self, scheduled: ScheduledJobs, settings: Settings
    ) -> None:
        """max_instances=1 stops a long run colliding with the next tick."""
        scheduler = build_scheduler(scheduled, settings)
        assert all(job.max_instances == 1 for job in scheduler.get_jobs())

    def test_an_invalid_cron_is_rejected_at_build_time(
        self, scheduled: ScheduledJobs, settings: Settings
    ) -> None:
        broken = settings.model_copy(update={"scheduled_scoring_cron": "nonsense"})
        with pytest.raises(ValueError, match="5-field cron"):
            build_scheduler(scheduled, broken)


class TestPendingFiles:
    def test_empty_directory_has_nothing_pending(self, scheduled: ScheduledJobs) -> None:
        assert scheduled.pending_files() == []

    def test_a_missing_directory_is_not_an_error(
        self, settings: Settings, scheduled: ScheduledJobs
    ) -> None:
        import shutil

        shutil.rmtree(settings.scheduled_input_dir)
        assert scheduled.pending_files() == []

    def test_only_csv_files_are_picked_up(
        self, settings: Settings, scheduled: ScheduledJobs, sample_csv: bytes
    ) -> None:
        (settings.scheduled_input_dir / "a.csv").write_bytes(sample_csv)
        (settings.scheduled_input_dir / "notes.txt").write_text("ignore me")

        pending = scheduled.pending_files()
        assert [path.name for path in pending] == ["a.csv"]

    def test_files_are_processed_in_a_stable_order(
        self, settings: Settings, scheduled: ScheduledJobs, sample_csv: bytes
    ) -> None:
        for name in ("c.csv", "a.csv", "b.csv"):
            (settings.scheduled_input_dir / name).write_bytes(sample_csv)
        assert [path.name for path in scheduled.pending_files()] == [
            "a.csv",
            "b.csv",
            "c.csv",
        ]


class TestScorePendingFiles:
    async def test_an_idle_directory_creates_no_jobs(
        self, scheduled: ScheduledJobs, database: Database
    ) -> None:
        assert await scheduled.score_pending_files() == []

        async with database.session() as session:
            _, total = await JobRepository(session).list_jobs()
            assert total == 0

    async def test_a_dropped_file_is_scored_and_archived(
        self,
        settings: Settings,
        scheduled: ScheduledJobs,
        database: Database,
        sample_csv: bytes,
    ) -> None:
        path = settings.scheduled_input_dir / "nightly.csv"
        path.write_bytes(sample_csv)

        job_ids = await scheduled.score_pending_files()

        assert len(job_ids) == 1
        # Archived, so the next tick does not rescore it.
        assert not path.exists()
        assert (settings.scheduled_output_dir / "nightly.csv").exists()

        async with database.session() as session:
            job = await JobRepository(session).get(uuid.UUID(job_ids[0]))
            assert job is not None
            assert job.status == JobStatus.SUCCEEDED.value
            assert job.scored_rows == 120
            assert job.source_filename == "nightly.csv"

            _, total = await PredictionRepository(session).list_predictions(
                source=PredictionSource.BATCH, page_size=1
            )
            assert total == 120

    async def test_running_twice_does_not_rescore(
        self,
        settings: Settings,
        scheduled: ScheduledJobs,
        database: Database,
        sample_csv: bytes,
    ) -> None:
        """Archiving is what makes the schedule idempotent."""
        (settings.scheduled_input_dir / "once.csv").write_bytes(sample_csv)

        first = await scheduled.score_pending_files()
        second = await scheduled.score_pending_files()

        assert len(first) == 1
        assert second == []

        async with database.session() as session:
            _, total = await PredictionRepository(session).list_predictions(page_size=1)
            assert total == 120

    async def test_multiple_files_are_all_processed(
        self,
        settings: Settings,
        scheduled: ScheduledJobs,
        database: Database,
        sample_csv: bytes,
    ) -> None:
        for name in ("a.csv", "b.csv"):
            (settings.scheduled_input_dir / name).write_bytes(sample_csv)

        job_ids = await scheduled.score_pending_files()

        assert len(job_ids) == 2
        async with database.session() as session:
            _, total = await PredictionRepository(session).list_predictions(page_size=1)
            assert total == 240

    async def test_a_bad_file_is_recorded_as_failed_and_others_still_run(
        self,
        settings: Settings,
        scheduled: ScheduledJobs,
        database: Database,
        sample_csv: bytes,
    ) -> None:
        (settings.scheduled_input_dir / "a_broken.csv").write_bytes(b"alpha,beta\n1,2\n")
        (settings.scheduled_input_dir / "b_good.csv").write_bytes(sample_csv)

        job_ids = await scheduled.score_pending_files()

        assert len(job_ids) == 1  # only the good file produced a scoring job
        async with database.session() as session:
            jobs, total = await JobRepository(session).list_jobs()
            statuses = {job.status for job in jobs}
            assert total == 2
            assert statuses == {JobStatus.SUCCEEDED.value, JobStatus.FAILED.value}

            failed = next(job for job in jobs if job.status == JobStatus.FAILED.value)
            assert failed.kind == JobKind.SCHEDULED_SCORING.value
            assert failed.source_filename == "a_broken.csv"
            assert failed.error

    async def test_a_bad_file_is_left_in_place_for_inspection(
        self, settings: Settings, scheduled: ScheduledJobs
    ) -> None:
        path = settings.scheduled_input_dir / "broken.csv"
        path.write_bytes(b"alpha,beta\n1,2\n")

        await scheduled.score_pending_files()

        assert path.exists()

    async def test_without_a_model_nothing_runs(
        self,
        settings: Settings,
        scheduled_without_model: ScheduledJobs,
        sample_csv: bytes,
    ) -> None:
        settings.scheduled_input_dir.mkdir(parents=True, exist_ok=True)
        (settings.scheduled_input_dir / "a.csv").write_bytes(sample_csv)

        assert await scheduled_without_model.score_pending_files() == []
        # The file is untouched, so it will be picked up once a model exists.
        assert (settings.scheduled_input_dir / "a.csv").exists()


class TestScheduledDriftCheck:
    async def test_records_a_job_and_a_check(
        self, scheduled: ScheduledJobs, database: Database
    ) -> None:
        job_id = await scheduled.run_drift_check()

        assert job_id is not None
        async with database.session() as session:
            job = await JobRepository(session).get(uuid.UUID(job_id))
            assert job is not None
            assert job.kind == JobKind.DRIFT_CHECK.value
            assert job.status == JobStatus.SUCCEEDED.value
            assert job.summary["outcome"] == DriftOutcome.INCONCLUSIVE.value

    async def test_detects_drift_in_recent_traffic(
        self,
        settings: Settings,
        scheduled: ScheduledJobs,
        database: Database,
    ) -> None:
        drifted = shift_frame_for_drift(generate_aps_like_frame(n_rows=250, seed=777)).drop(
            columns=["class"]
        )
        (settings.scheduled_input_dir / "drifted.csv").write_bytes(
            drifted.to_csv(index=False).encode()
        )
        await scheduled.score_pending_files()

        job_id = await scheduled.run_drift_check()

        assert job_id is not None
        async with database.session() as session:
            job = await JobRepository(session).get(uuid.UUID(job_id))
            assert job is not None
            assert job.summary["outcome"] == DriftOutcome.DRIFT_DETECTED.value
            assert job.summary["critical_features"]

    async def test_clean_traffic_is_not_flagged(
        self,
        settings: Settings,
        scheduled: ScheduledJobs,
        database: Database,
        training_frame: pd.DataFrame,
    ) -> None:
        clean = generate_aps_like_frame(n_rows=250, seed=888).drop(columns=["class"])
        (settings.scheduled_input_dir / "clean.csv").write_bytes(clean.to_csv(index=False).encode())
        await scheduled.score_pending_files()

        job_id = await scheduled.run_drift_check()

        assert job_id is not None
        async with database.session() as session:
            job = await JobRepository(session).get(uuid.UUID(job_id))
            assert job is not None
            assert job.summary["outcome"] == DriftOutcome.CLEAN.value

    async def test_without_a_model_it_is_skipped(
        self, scheduled_without_model: ScheduledJobs, database: Database
    ) -> None:
        assert await scheduled_without_model.run_drift_check() is None

        async with database.session() as session:
            _, total = await JobRepository(session).list_jobs()
            assert total == 0

    async def test_a_failure_is_recorded_as_a_failed_job(
        self,
        scheduled: ScheduledJobs,
        database: Database,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def boom(**kwargs: Any) -> None:
            raise RuntimeError("drift exploded")

        monkeypatch.setattr(
            scheduled._drift,
            "check_recent_predictions",
            boom,  # noqa: SLF001
        )

        assert await scheduled.run_drift_check() is None

        async with database.session() as session:
            jobs, total = await JobRepository(session).list_jobs()
            assert total == 1
            assert jobs[0].status == JobStatus.FAILED.value
            assert "drift exploded" in (jobs[0].error or "")


class TestArchiving:
    def test_the_output_directory_is_created_on_demand(
        self, settings: Settings, scheduled: ScheduledJobs, sample_csv: bytes
    ) -> None:
        assert not settings.scheduled_output_dir.exists()
        path = settings.scheduled_input_dir / "x.csv"
        path.write_bytes(sample_csv)

        scheduled._archive(path)  # noqa: SLF001

        assert (settings.scheduled_output_dir / "x.csv").exists()

    def test_archiving_moves_rather_than_copies(
        self, settings: Settings, scheduled: ScheduledJobs, sample_csv: bytes
    ) -> None:
        path = settings.scheduled_input_dir / "y.csv"
        path.write_bytes(sample_csv)

        scheduled._archive(path)  # noqa: SLF001

        assert not path.exists()


class TestDirectoryDefaults:
    def test_input_and_output_directories_are_configurable(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        assert settings.scheduled_input_dir != settings.scheduled_output_dir
        assert settings.scheduled_input_dir.is_relative_to(tmp_path)
