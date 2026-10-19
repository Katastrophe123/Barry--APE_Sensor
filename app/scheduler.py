"""Scheduled automation: nightly batch scoring and a drift check.

APScheduler rather than a cron container, because the jobs need the loaded
model and a database session - the same objects the API already holds. One
process, one model in memory.

Both jobs are **idempotent and self-contained**: scoring moves each file it
processes out of the input directory, and the drift check records a row whatever
it finds. Neither raises; a failed run is recorded as a failed job so it shows
up in ``GET /jobs``.

Disabled by default (``SCHEDULER_ENABLED=false``) so tests and local runs do not
get surprise background writes. The compose file turns it on for a dedicated
worker container.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from time import perf_counter

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import Settings
from app.db.repository import DriftRepository, JobRepository, PredictionRepository
from app.db.session import Database
from app.logging_config import bind_log_context, get_logger, unbind_log_context
from app.models.enums import JobKind
from app.services.batch_service import BatchScoringService
from app.services.drift_service import DriftService
from app.services.model_provider import ModelProvider

logger = get_logger(__name__)


def parse_cron(expression: str) -> CronTrigger:
    """Build a trigger from a 5-field cron expression."""
    fields = expression.split()
    if len(fields) != 5:
        raise ValueError(
            f"Expected a 5-field cron expression (m h dom mon dow), got {expression!r}"
        )
    minute, hour, day, month, day_of_week = fields
    return CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=day_of_week)


class ScheduledJobs:
    """The job bodies, separated from the scheduler so they are testable."""

    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        models: ModelProvider,
        batch_service: BatchScoringService,
        drift_service: DriftService,
    ) -> None:
        self._settings = settings
        self._database = database
        self._models = models
        self._batch = batch_service
        self._drift = drift_service

    # --- nightly scoring --------------------------------------------------
    def pending_files(self) -> list[Path]:
        """CSV files waiting in the watched input directory."""
        directory = self._settings.scheduled_input_dir
        if not directory.exists():
            return []
        return sorted(path for path in directory.glob("*.csv") if path.is_file())

    async def score_pending_files(self) -> list[str]:
        """Score every CSV in the input directory; return the job ids created."""
        if not self._models.loaded:
            logger.warning("scheduled_scoring_skipped", reason="no model loaded")
            return []

        files = self.pending_files()
        if not files:
            logger.info("scheduled_scoring_idle", directory=str(self._settings.scheduled_input_dir))
            return []

        job_ids: list[str] = []
        for path in files:
            bind_log_context(source_file=path.name)
            try:
                batch = self._batch.parse_csv(path.read_bytes(), filename=path.name)
                job_id = await self._batch.create_job(batch)
                await self._batch.run_job(job_id, batch)
                job_ids.append(str(job_id))
                self._archive(path)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
                logger.exception("scheduled_scoring_file_failed", error=str(exc))
                await self._record_failed_job(path.name, exc)
            finally:
                unbind_log_context("source_file")

        logger.info("scheduled_scoring_completed", files=len(files), jobs=len(job_ids))
        return job_ids

    def _archive(self, path: Path) -> None:
        """Move a processed file out of the input directory (idempotency)."""
        destination_dir = self._settings.scheduled_output_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination_dir / path.name))

    async def _record_failed_job(self, filename: str, exc: BaseException) -> None:
        async with self._database.session() as session:
            jobs = JobRepository(session)
            job = await jobs.create(
                kind=JobKind.SCHEDULED_SCORING,
                source_filename=filename,
                model_version=self._models.version,
            )
            await jobs.mark_failed(job, error=f"{type(exc).__name__}: {exc}")

    # --- nightly drift check ---------------------------------------------
    async def run_drift_check(self) -> str | None:
        """Compare recent prediction inputs with the training distribution."""
        if not self._models.loaded:
            logger.warning("scheduled_drift_skipped", reason="no model loaded")
            return None

        started = perf_counter()
        async with self._database.session() as session:
            jobs = JobRepository(session)
            job = await jobs.create(kind=JobKind.DRIFT_CHECK, model_version=self._models.version)
            job_id = job.id
            await jobs.mark_running(job)

        try:
            async with self._database.session() as session:
                check = await self._drift.check_recent_predictions(
                    predictions=PredictionRepository(session),
                    drifts=DriftRepository(session),
                    sample_size=self._settings.drift_sample_size,
                    job_id=job_id,
                )
                summary = {
                    "drift_check_id": str(check.id),
                    "outcome": check.outcome,
                    "drift_share": check.drift_share,
                    "max_psi": check.max_psi,
                    "critical_features": check.critical_features,
                }
                rows = check.rows_analysed
        except Exception as exc:  # noqa: BLE001 - recorded as a failed job
            logger.exception("scheduled_drift_failed", error=str(exc))
            async with self._database.session() as session:
                jobs = JobRepository(session)
                failed_job = await jobs.get(job_id)
                if failed_job is not None:
                    await jobs.mark_failed(
                        failed_job,
                        error=f"{type(exc).__name__}: {exc}",
                        duration_ms=int((perf_counter() - started) * 1000),
                    )
            return None

        async with self._database.session() as session:
            jobs = JobRepository(session)
            finished_job = await jobs.get(job_id)
            if finished_job is not None:
                await jobs.mark_succeeded(
                    finished_job,
                    scored_rows=rows,
                    duration_ms=int((perf_counter() - started) * 1000),
                    summary=summary,
                )
        return str(job_id)


def build_scheduler(jobs: ScheduledJobs, settings: Settings) -> AsyncIOScheduler:
    """Wire the job bodies onto cron triggers."""
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        jobs.score_pending_files,
        trigger=parse_cron(settings.scheduled_scoring_cron),
        id="scheduled_scoring",
        name="Score CSVs waiting in the input directory",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
        replace_existing=True,
    )
    scheduler.add_job(
        jobs.run_drift_check,
        trigger=parse_cron(settings.scheduled_drift_cron),
        id="scheduled_drift",
        name="Compare recent predictions with the training distribution",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
        replace_existing=True,
    )
    logger.info(
        "scheduler_configured",
        scoring_cron=settings.scheduled_scoring_cron,
        drift_cron=settings.scheduled_drift_cron,
        input_dir=str(settings.scheduled_input_dir),
    )
    return scheduler
