"""Tests for drift detection.

The important property is **both directions**: unchanged data must not be
flagged (or the alert is noise), and genuinely shifted data must be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.drift import PSI_CRITICAL, DriftDetector, DriftReport
from app.ml.features import FeaturePipeline, FeatureStats
from app.ml.predictor import Predictor
from app.ml.registry import ModelMetadata
from app.ml.sampledata import generate_aps_like_frame, shift_frame_for_drift


def stats_for(values: list[float], *, name: str = "a", missing_rate: float = 0.0) -> FeatureStats:
    """Build reference statistics the way the pipeline would."""
    pipeline = FeaturePipeline()
    series = pd.Series(values, dtype="float64")
    return FeatureStats(
        name=name,
        mean=float(series.mean()),
        std=float(series.std(ddof=0)),
        median=float(series.median()),
        minimum=float(series.min()),
        maximum=float(series.max()),
        missing_rate=missing_rate,
        quantile_edges=pipeline._quantile_edges(series, 10),  # noqa: SLF001
    )


class TestPopulationStabilityIndex:
    def test_identical_distribution_has_near_zero_psi(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(0, 1, 5000)
        edges = tuple(np.quantile(values, np.linspace(0, 1, 11)))

        psi = DriftDetector.population_stability_index(values, edges)
        assert psi < 0.01

    def test_shifted_distribution_has_large_psi(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(0, 1, 5000)
        edges = tuple(np.quantile(values, np.linspace(0, 1, 11)))

        psi = DriftDetector.population_stability_index(values + 5.0, edges)
        assert psi > PSI_CRITICAL

    def test_psi_is_non_negative(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.normal(0, 1, 500)
        edges = tuple(np.quantile(values, np.linspace(0, 1, 11)))
        assert DriftDetector.population_stability_index(values * 2, edges) >= 0.0

    def test_degenerate_edges_return_zero(self) -> None:
        assert DriftDetector.population_stability_index(np.array([1.0, 2.0]), (1.0,)) == 0.0

    def test_out_of_range_values_are_counted_not_dropped(self) -> None:
        """Outer edges are opened to infinity, so extremes still register.

        If out-of-range values were dropped, an all-out-of-range batch would
        produce an empty histogram and a PSI of ~0 - i.e. "no drift" for data
        that is entirely outside the training range.
        """
        edges = tuple(float(edge) for edge in range(11))  # 10 bins over 0..10

        balanced = DriftDetector.population_stability_index(np.arange(0.5, 10.5, 1.0), edges)
        far_below = DriftDetector.population_stability_index(np.full(100, -1000.0), edges)

        assert balanced < 0.01
        assert far_below > PSI_CRITICAL

    def test_empty_batch_does_not_raise(self) -> None:
        assert DriftDetector.population_stability_index(np.array([]), (0.0, 1.0)) >= 0.0


class TestKsAgainstTraining:
    def test_matching_data_has_a_small_statistic(self) -> None:
        rng = np.random.default_rng(0)
        reference = stats_for(list(rng.normal(10, 2, 2000)))
        statistic, p_value = DriftDetector.ks_against_training(rng.normal(10, 2, 2000), reference)
        assert statistic < 0.1
        assert p_value > 0.01

    def test_shifted_data_has_a_large_statistic(self) -> None:
        rng = np.random.default_rng(0)
        reference = stats_for(list(rng.normal(10, 2, 2000)))
        statistic, p_value = DriftDetector.ks_against_training(rng.normal(40, 2, 2000), reference)
        assert statistic > 0.9
        assert p_value < 0.01

    def test_empty_batch_is_neutral(self) -> None:
        reference = stats_for([1.0, 2.0, 3.0])
        assert DriftDetector.ks_against_training(np.array([]), reference) == (0.0, 1.0)

    def test_zero_variance_reference_does_not_divide_by_zero(self) -> None:
        reference = stats_for([5.0, 5.0, 5.0])
        statistic, _ = DriftDetector.ks_against_training(np.array([5.0, 6.0]), reference)
        assert np.isfinite(statistic)


class TestAnalyse:
    @pytest.fixture
    def detector(self) -> DriftDetector:
        rng = np.random.default_rng(0)
        return DriftDetector(
            training_stats={
                f"f{index}": stats_for(list(rng.normal(10, 2, 2000)), name=f"f{index}")
                for index in range(10)
            },
            min_rows=200,
        )

    @pytest.fixture
    def wide_detector(self) -> DriftDetector:
        """20 features, so one drifted feature is 5% - below the share rule."""
        rng = np.random.default_rng(0)
        return DriftDetector(
            training_stats={
                f"f{index}": stats_for(list(rng.normal(10, 2, 2000)), name=f"f{index}")
                for index in range(20)
            },
            min_rows=200,
        )

    @staticmethod
    def batch(shift: float = 0.0, columns: int = 10, rows: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        return pd.DataFrame(
            {f"f{index}": rng.normal(10 + shift, 2, rows) for index in range(columns)}
        )

    def test_unchanged_data_is_not_flagged(self, detector: DriftDetector) -> None:
        report = detector.analyse(self.batch())
        assert report.drift_detected is False
        assert report.drifted_features == []
        assert report.max_psi < 0.1

    def test_shifted_data_is_flagged(self, detector: DriftDetector) -> None:
        report = detector.analyse(self.batch(shift=20.0))
        assert report.drift_detected is True
        assert len(report.drifted_features) == 10
        assert report.drift_share == 1.0

    def test_one_catastrophically_drifted_feature_flags_the_batch(
        self, wide_detector: DriftDetector
    ) -> None:
        """Below the share threshold, but a single critical PSI must still fire."""
        frame = self.batch(columns=20)
        frame["f0"] = frame["f0"] + 100.0

        report = wide_detector.analyse(frame)

        # 1 of 20 features = 5%, under the 10% share rule.
        assert report.drift_share < wide_detector.drift_share_threshold
        assert report.critical_features == ["f0"]
        assert report.drift_detected is True

    def test_one_moderately_drifted_feature_does_not_flag_the_batch(
        self, wide_detector: DriftDetector
    ) -> None:
        """The converse: a single mild mover must not raise an alert."""
        frame = self.batch(columns=20)
        frame["f0"] = frame["f0"] + 1.0

        report = wide_detector.analyse(frame)

        assert report.critical_features == []
        assert report.drift_detected is False

    def test_missing_rate_jump_is_drift(self, detector: DriftDetector) -> None:
        """A sensor going offline shows up as `na`, not as a value shift."""
        frame = self.batch()
        frame.loc[frame.index[:300], "f1"] = np.nan

        report = detector.analyse(frame)

        drift = next(item for item in report.per_feature if item.feature == "f1")
        assert drift.batch_missing_rate == pytest.approx(0.6)
        assert drift.severity == "significant"

    def test_small_batches_are_reported_but_not_flagged(self, detector: DriftDetector) -> None:
        report = detector.analyse(self.batch(shift=20.0, rows=50))
        assert report.drift_detected is False
        assert "at least 200" in report.note
        assert report.max_psi > 0  # still measured

    def test_no_shared_features_is_a_clear_message(self, detector: DriftDetector) -> None:
        report = detector.analyse(pd.DataFrame({"unrelated": [1.0] * 100}))
        assert report.features_analysed == 0
        assert "No overlapping features" in report.note
        assert report.drift_detected is False

    def test_non_numeric_values_are_coerced(self, detector: DriftDetector) -> None:
        frame = self.batch()
        frame["f2"] = frame["f2"].astype(str)
        frame.loc[frame.index[0], "f2"] = "na"

        report = detector.analyse(frame)
        assert report.features_analysed == 10

    def test_report_serialises_with_a_feature_cap(self, detector: DriftDetector) -> None:
        report = detector.analyse(self.batch(shift=20.0))
        payload = report.to_dict(include_features=3)
        assert len(payload["top_features"]) == 3
        assert payload["drift_detected"] is True
        assert "critical_features" in payload

    def test_top_features_are_worst_first(self, detector: DriftDetector) -> None:
        frame = self.batch()
        frame["f0"] = frame["f0"] + 100.0
        report = detector.analyse(frame)
        assert report.top(3)[0].feature == "f0"

    def test_from_metadata_rebuilds_the_detector(self, detector: DriftDetector) -> None:
        payload = {name: stats.to_dict() for name, stats in detector.training_stats.items()}
        rebuilt = DriftDetector.from_metadata(payload, model_version="v1", min_rows=200)

        assert rebuilt.model_version == "v1"
        assert rebuilt.analyse(self.batch()).drift_detected is False


class TestEndToEndDrift:
    """Against the real trained artifact, not synthetic reference stats."""

    def test_training_like_data_is_not_flagged(
        self, predictor: Predictor, model_metadata: ModelMetadata
    ) -> None:
        detector = DriftDetector.from_metadata(
            model_metadata.feature_stats, model_version=model_metadata.version
        )
        frame = generate_aps_like_frame(n_rows=1000, seed=123)

        report = detector.analyse(predictor.prepare_for_drift(frame))

        assert report.drift_detected is False
        assert report.features_analysed == len(model_metadata.feature_list)

    def test_deliberately_shifted_data_is_flagged(
        self, predictor: Predictor, model_metadata: ModelMetadata
    ) -> None:
        detector = DriftDetector.from_metadata(
            model_metadata.feature_stats, model_version=model_metadata.version
        )
        frame = shift_frame_for_drift(generate_aps_like_frame(n_rows=1000, seed=123))

        report = detector.analyse(predictor.prepare_for_drift(frame))

        assert report.drift_detected is True
        assert report.critical_features
        assert report.max_psi > PSI_CRITICAL


class TestDriftReportDefaults:
    def test_an_empty_report_is_inert(self) -> None:
        report = DriftReport(rows_analysed=0, features_analysed=0)
        assert report.drift_detected is False
        assert report.top() == []
        assert report.to_dict()["drifted_features"] == []
