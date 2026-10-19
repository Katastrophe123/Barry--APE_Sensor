"""Domain enumerations shared by the ML layer, the database and the API."""

from __future__ import annotations

from enum import StrEnum


class RiskBand(StrEnum):
    """Operator-facing severity, defined relative to the model's threshold."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionSource(StrEnum):
    """How a prediction reached the service."""

    API = "api"
    BATCH = "batch"
    SCHEDULED = "scheduled"


class JobStatus(StrEnum):
    """Lifecycle of a batch-scoring job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobKind(StrEnum):
    BATCH_UPLOAD = "batch_upload"
    SCHEDULED_SCORING = "scheduled_scoring"
    DRIFT_CHECK = "drift_check"


class DriftSeverity(StrEnum):
    STABLE = "stable"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"


class DriftOutcome(StrEnum):
    """Verdict recorded for a drift check."""

    CLEAN = "clean"
    DRIFT_DETECTED = "drift_detected"
    INCONCLUSIVE = "inconclusive"
