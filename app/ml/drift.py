"""Data-drift detection against the training distribution.

Two complementary signals per feature:

**PSI (Population Stability Index)** bins the incoming batch using the *exact
quantile edges recorded at training time* and compares bin shares. Industry
convention: < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant.

**KS statistic** (two-sample Kolmogorov-Smirnov) against a normal approximation
of the training feature, which catches shape changes PSI's coarse bins miss.

Also tracked: **missing-rate drift**, because in this dataset a sensor going
offline shows up as a jump in ``na`` rather than a shift in values - and that is
exactly the failure mode that would silently degrade predictions.

A batch is flagged when **either**:

* the *share* of significantly-drifted features crosses ``drift_share_threshold``
  (default 10%) - broad, diffuse drift; or
* any single feature's PSI reaches ``psi_critical`` (default 1.0) - one sensor
  has moved so far that it is actionable on its own.

The share rule alone would miss "four critical sensors were recalibrated"; the
per-feature rule alone would cry wolf on one noisy column. Both together is the
behaviour the tests pin down.

Why ``min_rows`` defaults to 200
--------------------------------
PSI over 10 bins is badly behaved on small samples. Measured on this code,
drawing two samples from the *same* normal distribution gives:

===========  ===========  ========  ========
rows         median PSI   p90 PSI   max PSI
===========  ===========  ========  ========
30           0.347        1.503     2.702
60           0.149        0.263     1.440
100          0.089        0.155     0.283
200          0.042        0.075     0.123
800          0.010        0.018     0.034
===========  ===========  ========  ========

At 30 rows the *median* PSI of identical distributions already exceeds the 0.25
"significant" threshold, so any verdict from a batch that small is noise. At 200
rows the 90th percentile is 0.075 - comfortably clear. Smaller batches are still
measured and returned, but are reported as inconclusive rather than flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.logging_config import get_logger
from app.ml.features import FeatureStats

logger = get_logger(__name__)

PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25
#: A single feature at or above this PSI flags the whole batch on its own.
PSI_CRITICAL = 1.0
#: Guards against log(0) when a bin is empty on one side.
_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """Drift measurement for one feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    training_mean: float
    batch_mean: float
    training_missing_rate: float
    batch_missing_rate: float
    missing_rate_delta: float
    severity: str  # stable | moderate | significant

    @property
    def drifted(self) -> bool:
        return self.severity == "significant"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": self.psi,
            "ks_statistic": self.ks_statistic,
            "ks_p_value": self.ks_p_value,
            "training_mean": self.training_mean,
            "batch_mean": self.batch_mean,
            "training_missing_rate": self.training_missing_rate,
            "batch_missing_rate": self.batch_missing_rate,
            "missing_rate_delta": self.missing_rate_delta,
            "severity": self.severity,
        }


@dataclass(slots=True)
class DriftReport:
    """Batch-level drift verdict."""

    rows_analysed: int
    features_analysed: int
    drifted_features: list[str] = field(default_factory=list)
    moderate_features: list[str] = field(default_factory=list)
    #: Features whose PSI alone is enough to flag the batch.
    critical_features: list[str] = field(default_factory=list)
    per_feature: list[FeatureDrift] = field(default_factory=list)
    drift_share: float = 0.0
    drift_detected: bool = False
    max_psi: float = 0.0
    model_version: str | None = None
    note: str = ""

    def top(self, limit: int = 10) -> list[FeatureDrift]:
        """Worst-drifting features first, for logs and API responses."""
        return sorted(self.per_feature, key=lambda item: item.psi, reverse=True)[:limit]

    def to_dict(self, *, include_features: int = 10) -> dict[str, Any]:
        return {
            "rows_analysed": self.rows_analysed,
            "features_analysed": self.features_analysed,
            "drift_detected": self.drift_detected,
            "drift_share": self.drift_share,
            "max_psi": self.max_psi,
            "drifted_features": self.drifted_features,
            "moderate_features": self.moderate_features,
            "critical_features": self.critical_features,
            "model_version": self.model_version,
            "note": self.note,
            "top_features": [item.to_dict() for item in self.top(include_features)],
        }


class DriftDetector:
    """Compares a batch against recorded training statistics."""

    def __init__(
        self,
        *,
        training_stats: dict[str, FeatureStats],
        psi_significant: float = PSI_SIGNIFICANT,
        psi_moderate: float = PSI_MODERATE,
        psi_critical: float = PSI_CRITICAL,
        drift_share_threshold: float = 0.1,
        min_rows: int = 200,
        model_version: str | None = None,
    ) -> None:
        self.training_stats = training_stats
        self.psi_significant = psi_significant
        self.psi_moderate = psi_moderate
        self.psi_critical = psi_critical
        self.drift_share_threshold = drift_share_threshold
        self.min_rows = min_rows
        self.model_version = model_version

    @classmethod
    def from_metadata(
        cls, feature_stats: dict[str, Any], *, model_version: str | None = None, **kwargs: Any
    ) -> DriftDetector:
        """Build a detector from the ``feature_stats`` block of model metadata."""
        stats_map = {
            name: FeatureStats.from_dict(payload) for name, payload in feature_stats.items()
        }
        return cls(training_stats=stats_map, model_version=model_version, **kwargs)

    # --- metrics ----------------------------------------------------------
    @staticmethod
    def population_stability_index(
        batch: np.ndarray, edges: tuple[float, ...], training_shares: np.ndarray | None = None
    ) -> float:
        """PSI of ``batch`` against the training bin shares.

        When ``training_shares`` is omitted the edges are quantile boundaries,
        so training shares are uniform by construction.
        """
        bins = np.asarray(edges, dtype="float64")
        if bins.size < 2:
            return 0.0
        # Open the outer edges so out-of-range values are counted, not dropped.
        bounded = bins.copy()
        bounded[0], bounded[-1] = -np.inf, np.inf

        batch_counts, _ = np.histogram(batch, bins=bounded)
        batch_shares = batch_counts / max(batch_counts.sum(), 1)

        if training_shares is None:
            expected = np.full(batch_shares.shape, 1.0 / batch_shares.size)
        else:
            expected = np.asarray(training_shares, dtype="float64")

        batch_safe = np.clip(batch_shares, _EPSILON, None)
        expected_safe = np.clip(expected, _EPSILON, None)
        return float(np.sum((batch_safe - expected_safe) * np.log(batch_safe / expected_safe)))

    @staticmethod
    def ks_against_training(batch: np.ndarray, stats_entry: FeatureStats) -> tuple[float, float]:
        """KS test against a normal approximation of the training feature.

        The raw training sample is not kept (it would bloat every artifact), so
        the recorded mean/std stand in for it. That makes this a shape *hint*
        rather than a strict test, which is why PSI drives the verdict.
        """
        if batch.size == 0:
            return 0.0, 1.0
        scale = stats_entry.std if stats_entry.std > 0 else 1.0
        # A frozen distribution's `.cdf`, not ("norm", args=...): scipy 1.18
        # stopped forwarding `args` to the named-distribution CDF.
        reference = stats.norm(loc=stats_entry.mean, scale=scale)
        result = stats.kstest(batch, reference.cdf)
        return float(result.statistic), float(result.pvalue)

    def _severity(self, psi: float, missing_delta: float) -> str:
        if psi >= self.psi_significant or abs(missing_delta) >= 0.2:
            return "significant"
        if psi >= self.psi_moderate or abs(missing_delta) >= 0.1:
            return "moderate"
        return "stable"

    # --- API --------------------------------------------------------------
    def analyse(self, batch: pd.DataFrame) -> DriftReport:
        """Measure drift for every feature present in both batch and training."""
        shared = [name for name in self.training_stats if name in batch.columns]
        report = DriftReport(
            rows_analysed=len(batch),
            features_analysed=len(shared),
            model_version=self.model_version,
        )

        if not shared:
            report.note = "No overlapping features between the batch and the training schema."
            logger.warning("drift_no_shared_features", batch_columns=len(batch.columns))
            return report

        if len(batch) < self.min_rows:
            report.note = (
                f"Only {len(batch)} row(s); at least {self.min_rows} are needed for a "
                "meaningful drift verdict. Metrics are reported but not flagged."
            )

        measurements: list[FeatureDrift] = []
        for name in shared:
            stats_entry = self.training_stats[name]
            column = pd.to_numeric(batch[name], errors="coerce")
            batch_missing_rate = float(column.isna().mean())
            values = column.dropna().to_numpy(dtype="float64")

            psi = self.population_stability_index(values, stats_entry.quantile_edges)
            ks_statistic, ks_p = self.ks_against_training(values, stats_entry)
            missing_delta = round(batch_missing_rate - stats_entry.missing_rate, 6)

            measurements.append(
                FeatureDrift(
                    feature=name,
                    psi=round(psi, 6),
                    ks_statistic=round(ks_statistic, 6),
                    ks_p_value=round(ks_p, 6),
                    training_mean=round(stats_entry.mean, 6),
                    batch_mean=round(float(values.mean()), 6) if values.size else float("nan"),
                    training_missing_rate=round(stats_entry.missing_rate, 6),
                    batch_missing_rate=round(batch_missing_rate, 6),
                    missing_rate_delta=missing_delta,
                    severity=self._severity(psi, missing_delta),
                )
            )

        report.per_feature = measurements
        report.drifted_features = sorted(
            item.feature for item in measurements if item.severity == "significant"
        )
        report.moderate_features = sorted(
            item.feature for item in measurements if item.severity == "moderate"
        )
        report.max_psi = round(max(item.psi for item in measurements), 6)
        report.drift_share = round(len(report.drifted_features) / len(measurements), 6)
        report.critical_features = sorted(
            item.feature for item in measurements if item.psi >= self.psi_critical
        )
        # Either broad drift or one catastrophically-moved sensor flags the
        # batch. Too few rows means report-but-do-not-flag.
        report.drift_detected = len(batch) >= self.min_rows and (
            report.drift_share >= self.drift_share_threshold or bool(report.critical_features)
        )

        log = logger.warning if report.drift_detected else logger.info
        log(
            "drift_analysed",
            rows=report.rows_analysed,
            features=report.features_analysed,
            drift_detected=report.drift_detected,
            drift_share=report.drift_share,
            max_psi=report.max_psi,
            drifted=report.drifted_features[:10],
            critical=report.critical_features[:10],
            model_version=self.model_version,
        )
        return report
