"""Data-access layer for predictions, jobs and drift checks."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utcnow
from app.db.models import DriftCheck, Job, Prediction
from app.models.enums import (
    DriftOutcome,
    JobKind,
    JobStatus,
    PredictionSource,
    RiskBand,
)

SelectT = TypeVar("SelectT", bound=Select[Any])

#: Only the worst offenders are stored per drift check; the full per-feature
#: detail for 170 columns would bloat every row for little benefit.
DRIFT_DETAIL_FEATURES = 20


class PredictionRepository:
    """Writes and queries the prediction audit log."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        features: dict[str, Any],
        failure_probability: float,
        predicted_failure: bool,
        risk_band: RiskBand,
        decision_threshold: float,
        model_version: str,
        vehicle_id: str | None = None,
        source: PredictionSource = PredictionSource.API,
        job_id: uuid.UUID | None = None,
        missing_features: Sequence[str] = (),
        unexpected_features: Sequence[str] = (),
        request_metadata: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> Prediction:
        prediction = Prediction(
            vehicle_id=vehicle_id,
            features=features,
            request_metadata=request_metadata or {},
            missing_features=list(missing_features),
            unexpected_features=list(unexpected_features),
            failure_probability=failure_probability,
            predicted_failure=predicted_failure,
            risk_band=risk_band.value,
            decision_threshold=decision_threshold,
            model_version=model_version,
            source=source.value,
            job_id=job_id,
            latency_ms=latency_ms,
        )
        self._session.add(prediction)
        await self._session.flush()
        return prediction

    async def record_many(self, predictions: Sequence[Prediction]) -> int:
        if not predictions:
            return 0
        self._session.add_all(list(predictions))
        await self._session.flush()
        return len(predictions)

    async def get(self, prediction_id: uuid.UUID) -> Prediction | None:
        stmt = select(Prediction).where(Prediction.id == prediction_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    @staticmethod
    def _apply_filters(
        stmt: SelectT,
        *,
        risk_band: RiskBand | None,
        model_version: str | None,
        vehicle_id: str | None,
        source: PredictionSource | None,
        predicted_failure: bool | None,
    ) -> SelectT:
        if risk_band is not None:
            stmt = stmt.where(Prediction.risk_band == risk_band.value)
        if model_version is not None:
            stmt = stmt.where(Prediction.model_version == model_version)
        if vehicle_id is not None:
            stmt = stmt.where(Prediction.vehicle_id == vehicle_id)
        if source is not None:
            stmt = stmt.where(Prediction.source == source.value)
        if predicted_failure is not None:
            stmt = stmt.where(Prediction.predicted_failure.is_(predicted_failure))
        return stmt

    async def list_predictions(
        self,
        *,
        risk_band: RiskBand | None = None,
        model_version: str | None = None,
        vehicle_id: str | None = None,
        source: PredictionSource | None = None,
        predicted_failure: bool | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[Sequence[Prediction], int]:
        """Return ``(page_of_predictions, total)`` newest-first."""
        filters: dict[str, Any] = {
            "risk_band": risk_band,
            "model_version": model_version,
            "vehicle_id": vehicle_id,
            "source": source,
            "predicted_failure": predicted_failure,
        }
        count_stmt = self._apply_filters(
            select(Prediction).with_only_columns(func.count(Prediction.id)).order_by(None),
            **filters,
        )
        total = (await self._session.execute(count_stmt)).scalar_one()

        page_stmt = self._apply_filters(select(Prediction), **filters)
        page_stmt = (
            page_stmt.order_by(Prediction.created_at.desc(), Prediction.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        rows = (await self._session.execute(page_stmt)).scalars().all()
        return rows, int(total)

    async def recent_features(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        """Feature payloads of recent predictions, for the drift checker."""
        stmt = select(Prediction.features).order_by(Prediction.created_at.desc()).limit(limit)
        rows = (await self._session.execute(stmt)).scalars().all()
        return [dict(row or {}) for row in rows]

    async def count_by_risk_band(self) -> dict[str, int]:
        stmt = select(Prediction.risk_band, func.count(Prediction.id)).group_by(
            Prediction.risk_band
        )
        rows = await self._session.execute(stmt)
        return {band: int(count) for band, count in rows.all()}


class JobRepository:
    """Lifecycle management for batch and scheduled jobs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        kind: JobKind,
        source_filename: str | None = None,
        model_version: str | None = None,
        total_rows: int = 0,
    ) -> Job:
        job = Job(
            kind=kind.value,
            status=JobStatus.QUEUED.value,
            source_filename=source_filename,
            model_version=model_version,
            total_rows=total_rows,
        )
        self._session.add(job)
        await self._session.flush()
        return job

    async def get(self, job_id: uuid.UUID) -> Job | None:
        stmt = select(Job).where(Job.id == job_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def mark_running(self, job: Job) -> Job:
        job.status = JobStatus.RUNNING.value
        job.started_at = utcnow()
        await self._session.flush()
        return job

    async def mark_succeeded(
        self,
        job: Job,
        *,
        scored_rows: int,
        failed_rows: int = 0,
        high_risk_rows: int = 0,
        duration_ms: int | None = None,
        summary: dict[str, Any] | None = None,
    ) -> Job:
        job.status = JobStatus.SUCCEEDED.value
        job.scored_rows = scored_rows
        job.failed_rows = failed_rows
        job.high_risk_rows = high_risk_rows
        job.duration_ms = duration_ms
        job.summary = summary or {}
        job.finished_at = utcnow()
        await self._session.flush()
        return job

    async def mark_failed(self, job: Job, *, error: str, duration_ms: int | None = None) -> Job:
        job.status = JobStatus.FAILED.value
        job.error = error[:4000]
        job.duration_ms = duration_ms
        job.finished_at = utcnow()
        await self._session.flush()
        return job

    async def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        kind: JobKind | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[Sequence[Job], int]:
        def apply(stmt: SelectT) -> SelectT:
            if status is not None:
                stmt = stmt.where(Job.status == status.value)
            if kind is not None:
                stmt = stmt.where(Job.kind == kind.value)
            return stmt

        count_stmt = apply(select(Job).with_only_columns(func.count(Job.id)).order_by(None))
        total = (await self._session.execute(count_stmt)).scalar_one()

        page_stmt = apply(select(Job))
        page_stmt = (
            page_stmt.order_by(Job.created_at.desc(), Job.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        rows = (await self._session.execute(page_stmt)).scalars().all()
        return rows, int(total)


class DriftRepository:
    """Stores drift-check outcomes so drift over time is queryable."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        model_version: str,
        outcome: DriftOutcome,
        rows_analysed: int,
        features_analysed: int,
        drift_share: float,
        max_psi: float,
        drifted_features: Sequence[str] = (),
        critical_features: Sequence[str] = (),
        details: dict[str, Any] | None = None,
        note: str = "",
        job_id: uuid.UUID | None = None,
    ) -> DriftCheck:
        check = DriftCheck(
            model_version=model_version,
            outcome=outcome.value,
            rows_analysed=rows_analysed,
            features_analysed=features_analysed,
            drift_share=drift_share,
            max_psi=max_psi,
            drifted_features=list(drifted_features),
            critical_features=list(critical_features),
            details=details or {},
            note=note,
            job_id=job_id,
        )
        self._session.add(check)
        await self._session.flush()
        return check

    async def get(self, check_id: uuid.UUID) -> DriftCheck | None:
        stmt = select(DriftCheck).where(DriftCheck.id == check_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def latest(self, *, model_version: str | None = None) -> DriftCheck | None:
        stmt = select(DriftCheck).order_by(DriftCheck.created_at.desc()).limit(1)
        if model_version is not None:
            stmt = stmt.where(DriftCheck.model_version == model_version)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list_checks(
        self,
        *,
        outcome: DriftOutcome | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[Sequence[DriftCheck], int]:
        def apply(stmt: SelectT) -> SelectT:
            if outcome is not None:
                stmt = stmt.where(DriftCheck.outcome == outcome.value)
            return stmt

        count_stmt = apply(
            select(DriftCheck).with_only_columns(func.count(DriftCheck.id)).order_by(None)
        )
        total = (await self._session.execute(count_stmt)).scalar_one()

        page_stmt = apply(select(DriftCheck))
        page_stmt = (
            page_stmt.order_by(DriftCheck.created_at.desc(), DriftCheck.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        rows = (await self._session.execute(page_stmt)).scalars().all()
        return rows, int(total)
