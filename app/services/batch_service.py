"""Batch CSV scoring, executed as a background job.

``POST /predict/batch`` returns 202 immediately with a job id; the work happens
in a FastAPI background task that owns its own database session, because the
request-scoped session is closed as soon as the 202 is sent.

Rows are written in chunks rather than one transaction per row (one round trip
per truck would dominate the runtime) and rows that fail to score are counted
rather than aborting the whole job - a single malformed row in a 60k-row file
should not throw the other 59,999 away.
"""

from __future__ import annotations

import io
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from app.db.models import Prediction
from app.db.repository import JobRepository, PredictionRepository
from app.db.session import Database
from app.logging_config import bind_log_context, get_logger, unbind_log_context
from app.ml.schema import TARGET_COLUMN
from app.models.enums import JobKind, JobStatus, PredictionSource, RiskBand
from app.services.model_provider import ModelProvider

logger = get_logger(__name__)


class BatchError(Exception):
    """Raised when an uploaded batch cannot be accepted."""


@dataclass(frozen=True, slots=True)
class ParsedBatch:
    """A validated upload, ready to score."""

    frame: pd.DataFrame
    filename: str
    vehicle_ids: list[str | None]

    @property
    def row_count(self) -> int:
        return len(self.frame)


class BatchScoringService:
    """Parses, validates and scores uploaded CSV batches."""

    #: Column used as the vehicle identifier if present in the upload.
    VEHICLE_ID_COLUMNS = ("vehicle_id", "truck_id", "id")

    def __init__(
        self,
        *,
        models: ModelProvider,
        database: Database,
        max_rows: int = 200_000,
        chunk_size: int = 5_000,
    ) -> None:
        self._models = models
        self._database = database
        self.max_rows = max_rows
        self.chunk_size = chunk_size

    # --- parsing ----------------------------------------------------------
    def parse_csv(self, content: bytes, *, filename: str = "upload.csv") -> ParsedBatch:
        """Read and sanity-check an uploaded CSV."""
        if not content.strip():
            raise BatchError("The uploaded file is empty")
        try:
            frame = pd.read_csv(io.BytesIO(content), dtype=str, low_memory=False)
        except Exception as exc:  # noqa: BLE001 - any parse failure is a 400
            raise BatchError(f"Could not parse the file as CSV: {exc}") from exc

        if frame.empty:
            raise BatchError("The uploaded CSV has a header but no data rows")
        if len(frame) > self.max_rows:
            raise BatchError(f"The upload has {len(frame)} rows; the limit is {self.max_rows}")

        vehicle_ids: list[str | None] = [None] * len(frame)
        for column in self.VEHICLE_ID_COLUMNS:
            if column in frame.columns:
                vehicle_ids = [
                    None if pd.isna(value) else str(value)[:64] for value in frame[column].tolist()
                ]
                frame = frame.drop(columns=[column])
                break

        predictor = self._models.require_predictor()
        overlap = set(frame.columns) & set(predictor.required_features)
        if not overlap:
            raise BatchError(
                "None of the CSV columns match the current model's features. "
                f"Expected names like {predictor.required_features[:5]}."
            )

        return ParsedBatch(frame=frame, filename=filename, vehicle_ids=vehicle_ids)

    # --- scoring ----------------------------------------------------------
    async def create_job(self, batch: ParsedBatch) -> uuid.UUID:
        """Register a queued job and return its id (so the route can 202)."""
        predictor = self._models.require_predictor()
        async with self._database.session() as session:
            job = await JobRepository(session).create(
                kind=JobKind.BATCH_UPLOAD,
                source_filename=batch.filename,
                model_version=predictor.version,
                total_rows=batch.row_count,
            )
            return job.id

    async def run_job(self, job_id: uuid.UUID, batch: ParsedBatch) -> None:
        """Score the batch and update the job row. Never raises."""
        bind_log_context(job_id=str(job_id))
        started = perf_counter()
        try:
            await self._run(job_id, batch, started)
        except Exception as exc:  # noqa: BLE001 - a job failure is recorded, not raised
            logger.exception("batch_job_failed", error=str(exc))
            async with self._database.session() as session:
                jobs = JobRepository(session)
                job = await jobs.get(job_id)
                if job is not None:
                    await jobs.mark_failed(
                        job,
                        error=f"{type(exc).__name__}: {exc}",
                        duration_ms=self._elapsed_ms(started),
                    )
        finally:
            unbind_log_context("job_id")

    async def _run(self, job_id: uuid.UUID, batch: ParsedBatch, started: float) -> None:
        predictor = self._models.require_predictor()

        async with self._database.session() as session:
            jobs = JobRepository(session)
            job = await jobs.get(job_id)
            if job is None:
                logger.error("batch_job_missing", job_id=str(job_id))
                return
            await jobs.mark_running(job)

        scored = failed = high_risk = 0
        band_counts: dict[str, int] = {}
        probability_sum = 0.0

        for offset in range(0, batch.row_count, self.chunk_size):
            chunk = batch.frame.iloc[offset : offset + self.chunk_size]
            ids = batch.vehicle_ids[offset : offset + self.chunk_size]
            try:
                probabilities = predictor.predict_frame(chunk)
            except Exception as exc:  # noqa: BLE001 - skip the chunk, keep the job
                failed += len(chunk)
                logger.warning("batch_chunk_failed", offset=offset, rows=len(chunk), error=str(exc))
                continue

            rows: list[Prediction] = []
            for position, probability in enumerate(probabilities):
                value = float(probability)
                band = predictor.risk_band(value)
                band_counts[band] = band_counts.get(band, 0) + 1
                probability_sum += value
                if band == RiskBand.HIGH.value:
                    high_risk += 1
                record = chunk.iloc[position].to_dict()
                rows.append(
                    Prediction(
                        vehicle_id=ids[position] if position < len(ids) else None,
                        features=self._coerce_features(record),
                        request_metadata={"source_filename": batch.filename},
                        missing_features=[],
                        unexpected_features=[],
                        failure_probability=round(value, 6),
                        predicted_failure=value >= predictor.threshold,
                        risk_band=band,
                        decision_threshold=predictor.threshold,
                        model_version=predictor.version,
                        source=PredictionSource.BATCH.value,
                        job_id=job_id,
                    )
                )

            async with self._database.session() as session:
                await PredictionRepository(session).record_many(rows)
            scored += len(rows)
            logger.info("batch_chunk_scored", offset=offset, rows=len(rows), total=scored)

        duration_ms = self._elapsed_ms(started)
        summary: dict[str, Any] = {
            "risk_bands": band_counts,
            "mean_probability": round(probability_sum / scored, 6) if scored else 0.0,
            "rows_per_second": round(scored / (duration_ms / 1000), 1) if duration_ms else None,
            "had_labels": TARGET_COLUMN in batch.frame.columns,
        }

        async with self._database.session() as session:
            jobs = JobRepository(session)
            job = await jobs.get(job_id)
            if job is not None:
                await jobs.mark_succeeded(
                    job,
                    scored_rows=scored,
                    failed_rows=failed,
                    high_risk_rows=high_risk,
                    duration_ms=duration_ms,
                    summary=summary,
                )

        logger.info(
            "batch_job_completed",
            status=JobStatus.SUCCEEDED.value,
            scored=scored,
            failed=failed,
            high_risk=high_risk,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _coerce_features(record: Mapping[Any, Any]) -> dict[str, float | None]:
        """Store the row as JSON-safe numerics, with ``na`` becoming null."""
        coerced: dict[str, float | None] = {}
        for key, value in record.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                coerced[str(key)] = None
            else:
                coerced[str(key)] = None if pd.isna(number) else number
        return coerced

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int((perf_counter() - started) * 1000))
