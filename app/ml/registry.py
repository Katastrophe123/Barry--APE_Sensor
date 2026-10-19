"""Model artifact registry.

v1 produced a model that lived in a notebook's memory. Here every training run
writes two files plus a pointer:

    artifacts/
      model-<version>.joblib      # cleaner + feature pipeline + model + threshold
      metadata-<version>.json     # model type, measured metrics, date, feature list
      current.json                # {"version": "..."} - which model is live

Keeping the metadata as separate JSON means ``GET /models/current`` and the
drift checker can read it without unpickling the model, and a human can read it
without Python at all.
"""

from __future__ import annotations

import json
import platform
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from app.logging_config import get_logger
from app.ml.cleaning import DataCleaner
from app.ml.features import FeaturePipeline
from app.ml.training import TrainingResult

logger = get_logger(__name__)

CURRENT_POINTER = "current.json"


class ModelNotFoundError(Exception):
    """Raised when no trained model is available to load."""


class ModelMetadata(BaseModel):
    """Everything needed to audit a prediction after the fact.

    Every metric here was measured on the held-out split during the run that
    produced the artifact. Nothing is a target or an estimate.
    """

    model_config = ConfigDict(protected_namespaces=())

    version: str
    model_type: str = Field(description="Winning candidate, e.g. 'xgboost'")
    trained_at: datetime
    decision_threshold: float = Field(ge=0.0, le=1.0)

    # --- measured performance --------------------------------------------
    metrics: dict[str, Any] = Field(description="Held-out metrics for the winning model")
    candidate_comparison: list[dict[str, Any]] = Field(
        default_factory=list, description="Every candidate's measured metrics, cheapest first"
    )

    # --- data lineage -----------------------------------------------------
    feature_list: list[str] = Field(description="Exact feature order the model expects")
    raw_feature_list: list[str] = Field(
        default_factory=list,
        description="Columns that survived cleaning and feed the feature pipeline",
    )
    accepted_feature_list: list[str] = Field(
        default_factory=list,
        description=(
            "Every column the API accepts, including ones cleaning dropped. "
            "Callers may send these; only feature_list reaches the model."
        ),
    )
    training_rows: int
    test_rows: int
    positive_rate: float
    cleaning_report: dict[str, Any] = Field(default_factory=dict)
    feature_stats: dict[str, Any] = Field(
        default_factory=dict, description="Training distributions, used for drift detection"
    )
    dataset_fingerprint: str = Field(default="", description="Hash of the training frame")

    # --- environment ------------------------------------------------------
    library_versions: dict[str, str] = Field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_list)


@dataclass(slots=True)
class LoadedModel:
    """A deserialised artifact, ready to score."""

    metadata: ModelMetadata
    cleaner: DataCleaner
    pipeline: FeaturePipeline
    model: Any

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def threshold(self) -> float:
        return self.metadata.decision_threshold


def library_versions() -> dict[str, str]:
    """Record the environment an artifact was produced in.

    XGBoost is imported lazily: its wheel needs a native OpenMP runtime, so a
    hard import here would stop the registry working on a host that only ever
    serves a RandomForest artifact.
    """
    versions = {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except Exception:  # noqa: BLE001 - absence is information, not an error
        versions["xgboost"] = "unavailable"
    return versions


class ModelRegistry:
    """Saves and loads model artifacts from a directory."""

    def __init__(self, artifacts_dir: Path | str) -> None:
        self.artifacts_dir = Path(artifacts_dir)

    # --- paths ------------------------------------------------------------
    def model_path(self, version: str) -> Path:
        return self.artifacts_dir / f"model-{version}.joblib"

    def metadata_path(self, version: str) -> Path:
        return self.artifacts_dir / f"metadata-{version}.json"

    @property
    def pointer_path(self) -> Path:
        return self.artifacts_dir / CURRENT_POINTER

    # --- write ------------------------------------------------------------
    @staticmethod
    def new_version() -> str:
        """Sortable timestamp plus a short random suffix (two runs per second)."""
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:6]}"

    def save(
        self,
        result: TrainingResult,
        *,
        dataset_fingerprint: str = "",
        version: str | None = None,
        set_current: bool = True,
    ) -> ModelMetadata:
        """Persist a training result and optionally make it the live model."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        resolved = version or self.new_version()

        metadata = ModelMetadata(
            version=resolved,
            model_type=result.best_name,
            trained_at=datetime.now(tz=UTC),
            decision_threshold=result.best_threshold,
            metrics=result.best.evaluation.to_dict(),
            candidate_comparison=result.comparison(),
            feature_list=list(result.pipeline.feature_names_out_),
            raw_feature_list=list(result.cleaner.feature_names_),
            accepted_feature_list=list(result.cleaner.input_feature_names_),
            training_rows=result.training_rows,
            test_rows=result.test_rows,
            positive_rate=round(result.positive_rate, 6),
            cleaning_report=result.cleaner.report_.to_dict(),
            feature_stats=result.pipeline.stats_as_dict(),
            dataset_fingerprint=dataset_fingerprint,
            library_versions=library_versions(),
        )

        joblib.dump(
            {
                "cleaner": result.cleaner,
                "pipeline": result.pipeline,
                "model": result.best_model,
                "threshold": result.best_threshold,
                "version": resolved,
            },
            self.model_path(resolved),
            compress=3,
        )
        self.metadata_path(resolved).write_text(
            metadata.model_dump_json(indent=2), encoding="utf-8"
        )
        if set_current:
            self.set_current(resolved)

        logger.info(
            "model_saved",
            version=resolved,
            model_type=metadata.model_type,
            features=metadata.n_features,
            threshold=metadata.decision_threshold,
            path=str(self.model_path(resolved)),
        )
        return metadata

    def set_current(self, version: str) -> None:
        if not self.metadata_path(version).exists():
            raise ModelNotFoundError(f"No metadata for version {version}")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.pointer_path.write_text(json.dumps({"version": version}, indent=2), encoding="utf-8")

    # --- read -------------------------------------------------------------
    def list_versions(self) -> list[str]:
        """All saved versions, newest first (the version prefix sorts by time)."""
        if not self.artifacts_dir.exists():
            return []
        versions = [
            path.name.removeprefix("metadata-").removesuffix(".json")
            for path in self.artifacts_dir.glob("metadata-*.json")
        ]
        return sorted(versions, reverse=True)

    def current_version(self) -> str | None:
        """The pointed-at version, falling back to the newest on disk."""
        if self.pointer_path.exists():
            try:
                payload = json.loads(self.pointer_path.read_text(encoding="utf-8"))
                version = str(payload["version"])
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("current_pointer_unreadable", path=str(self.pointer_path))
            else:
                if self.metadata_path(version).exists():
                    return version
                logger.warning("current_pointer_dangling", version=version)
        versions = self.list_versions()
        return versions[0] if versions else None

    def load_metadata(self, version: str) -> ModelMetadata:
        path = self.metadata_path(version)
        if not path.exists():
            raise ModelNotFoundError(f"No metadata for version {version}")
        return ModelMetadata.model_validate_json(path.read_text(encoding="utf-8"))

    def load(self, version: str) -> LoadedModel:
        """Deserialise a specific version."""
        metadata = self.load_metadata(version)
        path = self.model_path(version)
        if not path.exists():
            raise ModelNotFoundError(f"No model artifact for version {version}")
        bundle = joblib.load(path)
        return LoadedModel(
            metadata=metadata,
            cleaner=bundle["cleaner"],
            pipeline=bundle["pipeline"],
            model=bundle["model"],
        )

    def load_current(self) -> LoadedModel:
        """Deserialise the live model, raising if nothing is trained yet."""
        version = self.current_version()
        if version is None:
            raise ModelNotFoundError(
                f"No trained model in {self.artifacts_dir}. Run `python scripts/train.py` first."
            )
        return self.load(version)
