"""SQLAlchemy 2.0 ORM models for prediction auditability.

Every prediction the service makes is stored with the **model version that
produced it**, which is the point: when a truck fails that the model called
low-risk, you need to know what was asked, what came back, and which artifact
answered.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampType, utcnow
from app.models.enums import JobStatus, PredictionSource


class Prediction(Base):
    """One scored record: the request, the result, and the model version."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    vehicle_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # --- request ----------------------------------------------------------
    features: Mapped[dict[str, Any]] = mapped_column(Base.json_type, nullable=False, default=dict)
    request_metadata: Mapped[dict[str, Any]] = mapped_column(
        Base.json_type, nullable=False, default=dict
    )
    missing_features: Mapped[list[str]] = mapped_column(
        Base.json_type, nullable=False, default=list
    )
    unexpected_features: Mapped[list[str]] = mapped_column(
        Base.json_type, nullable=False, default=list
    )

    # --- result -----------------------------------------------------------
    failure_probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_failure: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    risk_band: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(48), nullable=False, index=True)

    # --- provenance -------------------------------------------------------
    source: Mapped[str] = mapped_column(
        String(16), nullable=False, default=PredictionSource.API, index=True
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TimestampType, nullable=False, default=utcnow, index=True
    )

    job: Mapped[Job | None] = relationship(back_populates="predictions")

    __table_args__ = (
        # "Recent high-risk predictions" and "this model's recent output" are
        # the two queries an operator and an ML engineer actually run.
        Index("ix_predictions_risk_band_created_at", "risk_band", "created_at"),
        Index("ix_predictions_model_version_created_at", "model_version", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Prediction id={self.id} p={self.failure_probability:.4f} band={self.risk_band}>"


class Job(Base):
    """A batch-scoring or drift-check run."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(12), nullable=False, default=JobStatus.QUEUED, index=True
    )
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(48), nullable=True, index=True)

    total_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scored_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    high_risk_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary: Mapped[dict[str, Any]] = mapped_column(Base.json_type, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        TimestampType, nullable=False, default=utcnow, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(TimestampType, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(TimestampType, nullable=True)

    predictions: Mapped[list[Prediction]] = relationship(back_populates="job", passive_deletes=True)

    __table_args__ = (Index("ix_jobs_status_created_at", "status", "created_at"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Job id={self.id} kind={self.kind} status={self.status}>"


class DriftCheck(Base):
    """The outcome of one drift comparison against the training distribution."""

    __tablename__ = "drift_checks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    model_version: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    outcome: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    rows_analysed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    features_analysed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    drift_share: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_psi: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    drifted_features: Mapped[list[str]] = mapped_column(
        Base.json_type, nullable=False, default=list
    )
    critical_features: Mapped[list[str]] = mapped_column(
        Base.json_type, nullable=False, default=list
    )
    #: Per-feature PSI/KS detail, trimmed to the worst offenders.
    details: Mapped[dict[str, Any]] = mapped_column(Base.json_type, nullable=False, default=dict)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")

    job_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TimestampType, nullable=False, default=utcnow, index=True
    )

    __table_args__ = (Index("ix_drift_checks_outcome_created_at", "outcome", "created_at"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DriftCheck id={self.id} outcome={self.outcome} max_psi={self.max_psi}>"
