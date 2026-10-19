"""create predictions, jobs and drift_checks

Three tables, each matching a thing the API exposes:

* ``predictions`` - the audit log. Every row records the model version and
  decision threshold that produced it, which is what makes a disputed
  prediction answerable months later.
* ``jobs`` - batch uploads, scheduled scoring runs and drift checks.
* ``drift_checks`` - one row per comparison against the training distribution.

Indexes follow the queries the service issues: ``(risk_band, created_at)`` for
"recent high-risk vehicles", ``(model_version, created_at)`` for "what has this
artifact been predicting", and ``(status, created_at)`` for job polling.

Revision ID: bb3323396af5
Revises:
Create Date: 2026-10-11 12:28:05.853941+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "bb3323396af5"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("source_filename", sa.String(length=255), nullable=True),
        sa.Column("model_version", sa.String(length=48), nullable=True),
        sa.Column("total_rows", sa.Integer(), nullable=False),
        sa.Column("scored_rows", sa.Integer(), nullable=False),
        sa.Column("failed_rows", sa.Integer(), nullable=False),
        sa.Column("high_risk_rows", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column(
            "summary",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_jobs_created_at"), "jobs", ["created_at"], unique=False)
    op.create_index(op.f("ix_jobs_kind"), "jobs", ["kind"], unique=False)
    op.create_index(op.f("ix_jobs_model_version"), "jobs", ["model_version"], unique=False)
    op.create_index(op.f("ix_jobs_status"), "jobs", ["status"], unique=False)
    op.create_index("ix_jobs_status_created_at", "jobs", ["status", "created_at"], unique=False)
    op.create_table(
        "drift_checks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("model_version", sa.String(length=48), nullable=False),
        sa.Column("outcome", sa.String(length=20), nullable=False),
        sa.Column("rows_analysed", sa.Integer(), nullable=False),
        sa.Column("features_analysed", sa.Integer(), nullable=False),
        sa.Column("drift_share", sa.Float(), nullable=False),
        sa.Column("max_psi", sa.Float(), nullable=False),
        sa.Column(
            "drifted_features",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "critical_features",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "details",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_drift_checks_created_at"), "drift_checks", ["created_at"], unique=False
    )
    op.create_index(
        op.f("ix_drift_checks_model_version"), "drift_checks", ["model_version"], unique=False
    )
    op.create_index(op.f("ix_drift_checks_outcome"), "drift_checks", ["outcome"], unique=False)
    op.create_index(
        "ix_drift_checks_outcome_created_at",
        "drift_checks",
        ["outcome", "created_at"],
        unique=False,
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("vehicle_id", sa.String(length=64), nullable=True),
        sa.Column(
            "features",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "request_metadata",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "missing_features",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "unexpected_features",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("failure_probability", sa.Float(), nullable=False),
        sa.Column("predicted_failure", sa.Boolean(), nullable=False),
        sa.Column("risk_band", sa.String(length=8), nullable=False),
        sa.Column("decision_threshold", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=48), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_predictions_created_at"), "predictions", ["created_at"], unique=False)
    op.create_index(op.f("ix_predictions_job_id"), "predictions", ["job_id"], unique=False)
    op.create_index(
        op.f("ix_predictions_model_version"), "predictions", ["model_version"], unique=False
    )
    op.create_index(
        "ix_predictions_model_version_created_at",
        "predictions",
        ["model_version", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_predictions_predicted_failure"), "predictions", ["predicted_failure"], unique=False
    )
    op.create_index(op.f("ix_predictions_risk_band"), "predictions", ["risk_band"], unique=False)
    op.create_index(
        "ix_predictions_risk_band_created_at",
        "predictions",
        ["risk_band", "created_at"],
        unique=False,
    )
    op.create_index(op.f("ix_predictions_source"), "predictions", ["source"], unique=False)
    op.create_index(op.f("ix_predictions_vehicle_id"), "predictions", ["vehicle_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_predictions_vehicle_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_source"), table_name="predictions")
    op.drop_index("ix_predictions_risk_band_created_at", table_name="predictions")
    op.drop_index(op.f("ix_predictions_risk_band"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_predicted_failure"), table_name="predictions")
    op.drop_index("ix_predictions_model_version_created_at", table_name="predictions")
    op.drop_index(op.f("ix_predictions_model_version"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_job_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_created_at"), table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_drift_checks_outcome_created_at", table_name="drift_checks")
    op.drop_index(op.f("ix_drift_checks_outcome"), table_name="drift_checks")
    op.drop_index(op.f("ix_drift_checks_model_version"), table_name="drift_checks")
    op.drop_index(op.f("ix_drift_checks_created_at"), table_name="drift_checks")
    op.drop_table("drift_checks")
    op.drop_index("ix_jobs_status_created_at", table_name="jobs")
    op.drop_index(op.f("ix_jobs_status"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_model_version"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_kind"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_created_at"), table_name="jobs")
    op.drop_table("jobs")
