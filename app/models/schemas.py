"""Pydantic v2 request/response models for the HTTP layer."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import DriftOutcome, JobKind, JobStatus, PredictionSource, RiskBand

T = TypeVar("T")

MAX_FEATURES_PER_RECORD = 400


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
class TokenRequest(BaseModel):
    username: Annotated[str, Field(min_length=1, max_length=120)]
    password: Annotated[str, Field(min_length=1, max_length=256)]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Seconds until the token expires")


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
class PredictionRequest(BaseModel):
    """One sensor record to score.

    Features arrive as a mapping rather than ~170 declared fields: the real
    dataset's schema is wide and anonymised, and the authoritative list is
    whatever the loaded model recorded in its metadata. Validation here covers
    shape and types; the *names* are checked against the live model in
    :class:`~app.ml.predictor.Predictor`, which is the only component that
    knows them.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vehicle_id": "TRK-00421",
                "features": {
                    "aa_000": 76698.0,
                    "ab_000": 0.0,
                    "ac_000": 2130706438.0,
                    "ag_003": 0.0,
                    "ah_000": 1240520.0,
                },
            }
        }
    )

    vehicle_id: Annotated[str | None, Field(default=None, max_length=64)] = None
    features: dict[str, float | None] = Field(
        description="Sensor name -> value. null means the sensor reported 'na'."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("features")
    @classmethod
    def _validate_features(cls, value: dict[str, float | None]) -> dict[str, float | None]:
        if not value:
            raise ValueError("features must not be empty")
        if len(value) > MAX_FEATURES_PER_RECORD:
            raise ValueError(f"at most {MAX_FEATURES_PER_RECORD} features may be supplied")
        for name, item in value.items():
            if not name or len(name) > 64:
                raise ValueError(f"invalid feature name: {name!r}")
            if item is not None and not isinstance(item, (int, float)):
                raise ValueError(f"feature {name!r} must be numeric or null")
        return value


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    vehicle_id: str | None = None
    failure_probability: float
    predicted_failure: bool
    risk_band: RiskBand
    decision_threshold: float
    model_version: str
    source: PredictionSource
    missing_features: list[str] = Field(default_factory=list)
    unexpected_features: list[str] = Field(default_factory=list)
    latency_ms: int | None = None
    created_at: datetime


# --------------------------------------------------------------------------- #
# Batch jobs
# --------------------------------------------------------------------------- #
class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    kind: JobKind
    status: JobStatus
    source_filename: str | None = None
    model_version: str | None = None
    total_rows: int
    scored_rows: int
    failed_rows: int
    high_risk_rows: int
    error: str | None = None
    duration_ms: int | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}


class JobAccepted(BaseModel):
    """202 response for an accepted batch upload."""

    model_config = ConfigDict(protected_namespaces=())

    job_id: uuid.UUID
    status: JobStatus
    total_rows: int
    model_version: str
    poll_url: str


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ModelSummary(BaseModel):
    """What ``GET /models/current`` reports. All metrics are measured values."""

    model_config = ConfigDict(protected_namespaces=())

    version: str
    model_type: str
    trained_at: datetime
    decision_threshold: float
    n_features: int
    training_rows: int
    test_rows: int
    positive_rate: float
    metrics: dict[str, Any]
    candidate_comparison: list[dict[str, Any]] = Field(default_factory=list)
    feature_list: list[str]
    accepted_feature_list: list[str] = Field(default_factory=list)
    dataset_fingerprint: str = ""
    library_versions: dict[str, str] = Field(default_factory=dict)
    available_versions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
class DriftCheckResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    model_version: str
    outcome: DriftOutcome
    rows_analysed: int
    features_analysed: int
    drift_share: float
    max_psi: float
    drifted_features: list[str] = Field(default_factory=list)
    critical_features: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    note: str = ""
    created_at: datetime


# --------------------------------------------------------------------------- #
# LLM maintenance note
# --------------------------------------------------------------------------- #
class MaintenanceNoteRequest(BaseModel):
    """Ask for a plain-English note about an existing prediction."""

    prediction_id: uuid.UUID | None = None
    #: Alternative to ``prediction_id``: describe the prediction inline.
    failure_probability: Annotated[float | None, Field(default=None, ge=0.0, le=1.0)] = None
    vehicle_id: Annotated[str | None, Field(default=None, max_length=64)] = None
    top_signals: list[str] = Field(
        default_factory=list, max_length=10, description="Notable sensor readings"
    )

    @model_validator(mode="after")
    def _require_one_source(self) -> MaintenanceNoteRequest:
        if self.prediction_id is None and self.failure_probability is None:
            raise ValueError("supply either prediction_id or failure_probability")
        return self


class MaintenanceNoteResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction_id: uuid.UUID | None = None
    vehicle_id: str | None = None
    failure_probability: float
    risk_band: RiskBand
    note: str
    model_version: str | None = None
    llm_provider: str
    llm_model: str
    generated_by: str = Field(
        description="'llm' when the model wrote it, 'template' when the fallback did"
    )


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #
class Page(BaseModel, Generic[T]):
    items: list[T]
    page: int
    page_size: int
    total: int
    total_pages: int


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    app: str
    environment: str
    version: str
    database: str
    model_loaded: bool
    model_version: str | None = None
    llm_provider: str
    scheduler_running: bool
