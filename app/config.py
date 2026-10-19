"""Application settings, loaded from the environment (12-factor style)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["stub", "openai", "huggingface"]
Environment = Literal["local", "test", "production"]


class Settings(BaseSettings):
    """Runtime configuration. See ``.env.example`` for the full list."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    app_name: str = "aps-failure-prediction-service"
    environment: Environment = "local"
    log_level: str = "INFO"
    log_json: bool = False

    # --- Database ---------------------------------------------------------
    database_url: str = "postgresql+asyncpg://aps:aps@localhost:5432/aps"
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # --- Auth -------------------------------------------------------------
    jwt_secret: str = "insecure-local-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 120
    auth_username: str = "operator"
    auth_password: str = "operator"

    # --- Model artifacts --------------------------------------------------
    artifacts_dir: Path = Path("artifacts")
    #: Training data used by scripts/train.py and the scheduled drift baseline.
    training_data_path: Path = Path("data/sample/aps_sample_train.csv")

    # --- Batch scoring ----------------------------------------------------
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    max_batch_rows: int = Field(default=200_000, ge=1)
    #: Rows per database flush while scoring a batch. Lower values mean more
    #: round trips; the tests use a small value to exercise chunk boundaries.
    batch_chunk_size: int = Field(default=5_000, ge=1)

    # --- Drift ------------------------------------------------------------
    drift_psi_significant: float = Field(default=0.25, gt=0.0)
    drift_psi_critical: float = Field(default=1.0, gt=0.0)
    drift_share_threshold: float = Field(default=0.1, gt=0.0, le=1.0)
    #: Minimum batch size for a *flaggable* drift verdict. Below this, PSI is
    #: dominated by sampling noise; see app/ml/drift.py for the measurements.
    drift_min_rows: int = Field(default=200, ge=1)
    #: How many recent predictions the scheduled drift check samples.
    drift_sample_size: int = Field(default=1_000, ge=1)

    # --- Scheduler --------------------------------------------------------
    scheduler_enabled: bool = False
    #: Cron-style schedule for the batch-scoring job.
    scheduled_scoring_cron: str = "0 2 * * *"
    scheduled_drift_cron: str = "30 2 * * *"
    #: Directory the scheduled scoring job watches for new CSVs.
    scheduled_input_dir: Path = Path("data/incoming")
    scheduled_output_dir: Path = Path("data/scored")

    # --- LLM --------------------------------------------------------------
    llm_provider: LLMProvider = "stub"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0
    llm_max_attempts: int = Field(default=2, ge=1, le=10)
    llm_backoff_base_seconds: float = Field(default=0.5, ge=0.0)

    # --- Pagination -------------------------------------------------------
    default_page_size: int = Field(default=20, ge=1, le=200)
    max_page_size: int = Field(default=100, ge=1, le=500)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so the env is parsed once per process."""
    return Settings()
