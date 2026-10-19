"""Data cleaning, extracted from the v1 notebook into a fit/transform class.

The notebook cleaned the training frame inline, which meant inference had no way
to reproduce the same decisions. :class:`DataCleaner` *learns* what to drop
during ``fit`` and replays exactly that in ``transform``, so a column dropped at
training time is dropped identically at scoring time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.logging_config import get_logger
from app.ml.schema import (
    MISSING_TOKENS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    TARGET_COLUMN,
)

logger = get_logger(__name__)


class DataCleaningError(Exception):
    """Raised when a frame cannot be cleaned (e.g. the target is missing)."""


@dataclass(slots=True)
class CleaningReport:
    """What ``fit`` decided, kept for logging, tests and model metadata."""

    input_rows: int = 0
    input_columns: int = 0
    input_features: list[str] = field(default_factory=list)
    dropped_high_missing: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    duplicate_rows_removed: int = 0
    kept_features: list[str] = field(default_factory=list)
    missing_fraction: dict[str, float] = field(default_factory=dict)
    positive_rows: int = 0
    negative_rows: int = 0

    @property
    def positive_rate(self) -> float:
        total = self.positive_rows + self.negative_rows
        return self.positive_rows / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "input_columns": self.input_columns,
            "dropped_high_missing": self.dropped_high_missing,
            "dropped_constant": self.dropped_constant,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "kept_feature_count": len(self.kept_features),
            "positive_rows": self.positive_rows,
            "negative_rows": self.negative_rows,
            "positive_rate": round(self.positive_rate, 6),
        }


class DataCleaner:
    """Coerces the raw APS CSV into a numeric frame plus a 0/1 target.

    Steps, in order:

    1. Replace the dataset's ``na``-style missing tokens with ``NaN``.
    2. Coerce every feature column to numeric (unparseable values become
       ``NaN`` rather than killing the run).
    3. Drop columns whose missing fraction exceeds ``max_missing_fraction``.
    4. Drop columns that are constant (zero variance carries no signal).
    5. Drop duplicate rows (training only - scoring must not silently drop
       rows the caller sent).
    """

    def __init__(
        self,
        *,
        max_missing_fraction: float = 0.7,
        drop_constant: bool = True,
        drop_duplicates: bool = True,
    ) -> None:
        if not 0.0 < max_missing_fraction <= 1.0:
            raise ValueError("max_missing_fraction must be in (0, 1]")
        self.max_missing_fraction = max_missing_fraction
        self.drop_constant = drop_constant
        self.drop_duplicates = drop_duplicates

        #: Every feature column seen during fit, including those dropped. The
        #: API accepts all of these; only ``feature_names_`` reach the model.
        self.input_feature_names_: list[str] = []
        self.feature_names_: list[str] = []
        self.report_ = CleaningReport()
        self._fitted = False

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _to_numeric(frame: pd.DataFrame) -> pd.DataFrame:
        """Replace missing tokens then coerce everything to float."""
        cleaned = frame.replace(list(MISSING_TOKENS), np.nan)
        return cleaned.apply(pd.to_numeric, errors="coerce").astype("float64")

    @staticmethod
    def encode_target(series: pd.Series) -> pd.Series:
        """Map ``pos``/``neg`` (or already-numeric labels) onto 1/0."""
        if pd.api.types.is_numeric_dtype(series):
            encoded = series.astype("float64")
        else:
            normalised = series.astype(str).str.strip().str.lower()
            mapping = {POSITIVE_LABEL: 1.0, NEGATIVE_LABEL: 0.0, "1": 1.0, "0": 0.0}
            encoded = normalised.map(mapping)

        if encoded.isna().any():
            unknown = sorted(set(series[encoded.isna()].astype(str).unique()))[:5]
            raise DataCleaningError(f"Unrecognised target labels: {unknown}")
        return encoded.astype("int64")

    @property
    def fitted(self) -> bool:
        return self._fitted

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise DataCleaningError("DataCleaner must be fitted before transform")

    # --- API --------------------------------------------------------------
    def fit(self, frame: pd.DataFrame) -> DataCleaner:
        """Learn which columns to keep from a labelled training frame."""
        if TARGET_COLUMN not in frame.columns:
            raise DataCleaningError(f"Training frame is missing the '{TARGET_COLUMN}' column")

        report = CleaningReport(input_rows=len(frame), input_columns=frame.shape[1])
        working = frame.copy()

        if self.drop_duplicates:
            before = len(working)
            working = working.drop_duplicates()
            report.duplicate_rows_removed = before - len(working)

        target = self.encode_target(working[TARGET_COLUMN])
        features = self._to_numeric(working.drop(columns=[TARGET_COLUMN]))
        self.input_feature_names_ = list(features.columns)

        missing_fraction = features.isna().mean()
        report.missing_fraction = {
            str(name): round(float(value), 6) for name, value in missing_fraction.items()
        }
        report.dropped_high_missing = sorted(
            missing_fraction[missing_fraction > self.max_missing_fraction].index.tolist()
        )
        kept = features.drop(columns=report.dropped_high_missing)

        if self.drop_constant and not kept.empty:
            # nunique(dropna=True) is the point, not an inefficiency: it makes
            # an all-NaN column count as constant, which `.std() == 0` would not.
            constant = [
                name
                for name in kept.columns
                if kept[name].nunique(dropna=True) <= 1  # noqa: PD101
            ]
            report.dropped_constant = sorted(constant)
            kept = kept.drop(columns=constant)

        report.kept_features = list(kept.columns)
        report.input_features = self.input_feature_names_
        report.positive_rows = int((target == 1).sum())
        report.negative_rows = int((target == 0).sum())

        self.feature_names_ = report.kept_features
        self.report_ = report
        self._fitted = True

        logger.info(
            "data_cleaner_fitted",
            input_rows=report.input_rows,
            kept_features=len(report.kept_features),
            dropped_high_missing=len(report.dropped_high_missing),
            dropped_constant=len(report.dropped_constant),
            duplicates_removed=report.duplicate_rows_removed,
            positive_rate=round(report.positive_rate, 6),
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned column selection and numeric coercion.

        Missing learned columns are added as all-``NaN`` so the imputer handles
        them; unknown extra columns are ignored. Rows are never dropped.
        """
        self._require_fitted()
        working = frame.drop(columns=[TARGET_COLUMN], errors="ignore")
        numeric = self._to_numeric(working)

        absent = [name for name in self.feature_names_ if name not in numeric.columns]
        if absent:
            logger.warning("cleaner_missing_columns", columns=absent[:10], count=len(absent))
            for name in absent:
                numeric[name] = np.nan

        return numeric[self.feature_names_]

    def fit_transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Fit on a labelled frame and return ``(features, target)``."""
        self.fit(frame)
        working = frame.drop_duplicates() if self.drop_duplicates else frame
        target = self.encode_target(working[TARGET_COLUMN])
        features = self.transform(working)
        return features.reset_index(drop=True), target.reset_index(drop=True)
