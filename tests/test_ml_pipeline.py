"""Tests for the feature pipeline, evaluator, trainer and registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.evaluation import Evaluator
from app.ml.features import FeaturePipeline, FeaturePipelineError, FeatureStats
from app.ml.registry import ModelNotFoundError, ModelRegistry, library_versions
from app.ml.sampledata import generate_aps_like_frame
from app.ml.schema import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, DatasetSpec
from app.ml.training import (
    ModelTrainer,
    TrainingResult,
    available_model_factories,
    xgboost_available,
)


class TestFeaturePipeline:
    def test_requires_fitting_before_transform(self) -> None:
        with pytest.raises(FeaturePipelineError, match="must be fitted"):
            FeaturePipeline().transform(pd.DataFrame({"a": [1.0]}))

    def test_rejects_an_empty_frame(self) -> None:
        with pytest.raises(FeaturePipelineError, match="empty frame"):
            FeaturePipeline().fit(pd.DataFrame())

    def test_median_imputes_missing_values(self) -> None:
        pipeline = FeaturePipeline()
        out = pipeline.fit_transform(pd.DataFrame({"a": [1.0, np.nan, 3.0, 5.0]}))
        # median of [1, 3, 5] is 3
        assert out["a"].tolist() == [1.0, 3.0, 3.0, 5.0]

    def test_drops_near_constant_columns_after_imputation(self) -> None:
        pipeline = FeaturePipeline()
        pipeline.fit(pd.DataFrame({"varies": [1.0, 2.0, 3.0], "flat": [7.0, 7.0, 7.0]}))
        assert pipeline.dropped_low_variance_ == ["flat"]
        assert pipeline.feature_names_out_ == ["varies"]

    def test_raises_if_everything_is_constant(self) -> None:
        with pytest.raises(FeaturePipelineError, match="near-constant"):
            FeaturePipeline().fit(pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]}))

    def test_transform_aligns_and_reorders_columns(self) -> None:
        pipeline = FeaturePipeline()
        pipeline.fit(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        out = pipeline.transform(pd.DataFrame({"b": [9.0], "a": [8.0]}))
        assert list(out.columns) == ["a", "b"]
        assert out.iloc[0].tolist() == [8.0, 9.0]

    def test_transform_fills_absent_columns_with_the_training_median(self) -> None:
        pipeline = FeaturePipeline()
        pipeline.fit(pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [1.0, 2.0, 3.0]}))
        out = pipeline.transform(pd.DataFrame({"a": [10.0]}))
        assert out["b"].iloc[0] == pytest.approx(2.0)

    def test_output_has_no_missing_values(self) -> None:
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 4.0]})
        out = FeaturePipeline().fit_transform(frame)
        assert out.isna().sum().sum() == 0


class TestFeatureStats:
    @pytest.fixture
    def fitted(self) -> FeaturePipeline:
        pipeline = FeaturePipeline()
        pipeline.fit(
            pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
        )
        return pipeline

    def test_missing_rate_comes_from_the_raw_column(self, fitted: FeaturePipeline) -> None:
        assert fitted.feature_stats_["a"].missing_rate == pytest.approx(0.2)
        assert fitted.feature_stats_["b"].missing_rate == 0.0

    def test_stats_exclude_imputed_values(self, fitted: FeaturePipeline) -> None:
        """Mean must be of the observed values, not of the median-filled ones."""
        assert fitted.feature_stats_["a"].mean == pytest.approx(3.0)

    def test_quantile_edges_are_monotone(self, fitted: FeaturePipeline) -> None:
        edges = fitted.feature_stats_["b"].quantile_edges
        assert len(edges) >= 2
        assert list(edges) == sorted(edges)

    def test_low_cardinality_column_still_gets_two_edges(self) -> None:
        pipeline = FeaturePipeline()
        pipeline.fit(pd.DataFrame({"a": [1.0, 1.0, 1.0, 2.0], "b": [1.0, 2.0, 3.0, 4.0]}))
        assert len(pipeline.feature_stats_["a"].quantile_edges) >= 2

    def test_stats_roundtrip_through_json(self, fitted: FeaturePipeline) -> None:
        payload = fitted.stats_as_dict()
        restored = FeatureStats.from_dict(payload["a"])
        assert restored == fitted.feature_stats_["a"]


class TestEvaluator:
    def test_perfect_predictions_cost_nothing(self) -> None:
        result = Evaluator().evaluate(np.array([0, 0, 1, 1]), np.array([0.01, 0.02, 0.98, 0.99]))
        assert result.total_cost == 0
        assert result.recall == 1.0
        assert result.precision == 1.0
        assert result.confusion.false_negatives == 0

    def test_cost_uses_the_scania_weights(self) -> None:
        evaluator = Evaluator()
        # One false positive and one false negative.
        result = evaluator.evaluate(np.array([0, 1, 0, 1]), np.array([0.9, 0.1, 0.1, 0.9]))
        assert result.confusion.false_positives == 1
        assert result.confusion.false_negatives == 1
        assert result.total_cost == COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE

    def test_a_missed_failure_costs_fifty_times_a_false_alarm(self) -> None:
        evaluator = Evaluator()
        assert evaluator.total_cost(50, 0) == evaluator.total_cost(0, 1)

    def test_custom_costs_are_respected(self) -> None:
        evaluator = Evaluator(cost_false_positive=1, cost_false_negative=2)
        assert evaluator.total_cost(3, 4) == 3 + 8

    def test_single_class_truth_does_not_raise(self) -> None:
        """A degenerate fold must report 0.0 AUC, not kill the run."""
        result = Evaluator().evaluate(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
        assert result.roc_auc == 0.0
        assert result.pr_auc == 0.0

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Evaluator().evaluate(np.array([]), np.array([]))

    def test_mismatched_shapes_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="align"):
            Evaluator().evaluate(np.array([0, 1]), np.array([0.5]))

    def test_legacy_regression_metrics_are_reported_separately(self) -> None:
        result = Evaluator().evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
        assert set(result.legacy_regression) == {"r2", "rmse", "mae"}
        # They are namespaced, not mixed into the headline metrics.
        assert "r2" not in result.to_dict()["confusion"]

    def test_threshold_tuning_beats_the_default_on_imbalanced_data(self) -> None:
        """With a 50:1 cost ratio the cost-optimal threshold is far below 0.5."""
        evaluator = Evaluator()
        truth = np.array([0] * 95 + [1] * 5)
        # Positives score 0.3, negatives 0.1: nothing clears 0.5.
        scores = np.array([0.1] * 95 + [0.3] * 5)

        default_cost = evaluator.evaluate(truth, scores, threshold=0.5).total_cost
        threshold, tuned_cost = evaluator.best_threshold(truth, scores)

        assert default_cost == 5 * COST_FALSE_NEGATIVE
        assert tuned_cost < default_cost
        assert threshold < 0.5

    def test_threshold_tuning_returns_the_reported_cost(self) -> None:
        evaluator = Evaluator()
        truth = np.array([0] * 50 + [1] * 10)
        rng = np.random.default_rng(0)
        scores = np.concatenate([rng.uniform(0, 0.4, 50), rng.uniform(0.2, 1.0, 10)])

        threshold, cost = evaluator.best_threshold(truth, scores)
        assert evaluator.evaluate(truth, scores, threshold=threshold).total_cost == cost

    def test_summary_line_is_readable(self) -> None:
        result = Evaluator().evaluate(np.array([0, 1]), np.array([0.1, 0.9]))
        assert "cost=" in result.summary_line()
        assert "roc_auc=" in result.summary_line()


class TestTrainer:
    def test_rejects_an_invalid_test_size(self) -> None:
        with pytest.raises(ValueError, match="test_size"):
            ModelTrainer(test_size=1.5)

    def test_rejects_an_empty_candidate_set(self) -> None:
        with pytest.raises(ValueError, match="No model candidates"):
            ModelTrainer(model_factories={})

    def test_scale_pos_weight_is_the_imbalance_ratio(self) -> None:
        weight = ModelTrainer.scale_pos_weight(np.array([0] * 90 + [1] * 10))
        assert weight == pytest.approx(9.0)

    def test_scale_pos_weight_survives_zero_positives(self) -> None:
        assert ModelTrainer.scale_pos_weight(np.array([0, 0, 0])) == 1.0

    def test_single_class_data_is_rejected(self) -> None:
        frame = generate_aps_like_frame(n_rows=200, seed=1)
        frame["class"] = "neg"
        with pytest.raises(ValueError, match="single class"):
            ModelTrainer().train(frame)

    def test_available_candidates_track_the_xgboost_runtime(self) -> None:
        factories = available_model_factories()
        assert "random_forest" in factories
        assert ("xgboost" in factories) is xgboost_available()

    def test_selection_follows_cost_not_auc(self) -> None:
        """Both candidates have AUC 0.5, so only cost can decide between them.

        At a 10% failure rate, missing every failure (500 each) is dearer than
        flagging every vehicle (10 each), so ``catch_all`` must win.
        """
        from sklearn.dummy import DummyClassifier

        frame = generate_aps_like_frame(n_rows=600, seed=3, positive_rate=0.1)
        result = ModelTrainer(
            model_factories={
                "catch_all": lambda _: DummyClassifier(strategy="constant", constant=1),
                "miss_all": lambda _: DummyClassifier(strategy="constant", constant=0),
            },
            tune_threshold=False,
        ).train(frame)

        assert result.best_name == "catch_all"
        assert {item.name for item in result.candidates} == {"catch_all", "miss_all"}
        assert all(item.evaluation.roc_auc == 0.5 for item in result.candidates)

    def test_cost_metric_has_a_break_even_positive_rate(self) -> None:
        """Documents a real property of the Scania cost metric.

        "Flag everything" costs 10 per negative; "miss everything" costs 500
        per positive. Below a positive rate of 10/510 (~1.96%), doing nothing is
        genuinely cheaper than flagging everything - which is exactly why a
        model has to beat *both* trivial baselines, and why cost alone is not a
        sufficient acceptance criterion. See README "Limitations".
        """
        from sklearn.dummy import DummyClassifier

        factories = {
            "catch_all": lambda _: DummyClassifier(strategy="constant", constant=1),
            "miss_all": lambda _: DummyClassifier(strategy="constant", constant=0),
        }
        # 1.7% positives, the real dataset's rate: below break-even.
        sparse = ModelTrainer(model_factories=factories, tune_threshold=False).train(
            generate_aps_like_frame(n_rows=600, seed=3, positive_rate=0.017)
        )
        assert sparse.best_name == "miss_all"

        # 10% positives: above break-even.
        dense = ModelTrainer(model_factories=factories, tune_threshold=False).train(
            generate_aps_like_frame(n_rows=600, seed=3, positive_rate=0.1)
        )
        assert dense.best_name == "catch_all"

    def test_result_exposes_a_sorted_comparison(self) -> None:
        from sklearn.dummy import DummyClassifier

        frame = generate_aps_like_frame(n_rows=600, seed=4)
        result = ModelTrainer(
            model_factories={
                "a": lambda _: DummyClassifier(strategy="constant", constant=1),
                "b": lambda _: DummyClassifier(strategy="constant", constant=0),
            },
            tune_threshold=False,
        ).train(frame)

        costs = [item["tuned_cost"] for item in result.comparison()]
        assert costs == sorted(costs)
        assert result.best.name == result.best_name

    def test_training_records_split_sizes(self) -> None:
        from sklearn.dummy import DummyClassifier

        frame = generate_aps_like_frame(n_rows=1000, seed=5)
        result = ModelTrainer(
            model_factories={"d": lambda _: DummyClassifier(strategy="constant", constant=1)},
            test_size=0.25,
        ).train(frame)

        assert isinstance(result, TrainingResult)
        assert result.test_rows == pytest.approx(result.training_rows / 3, rel=0.1)
        assert 0.0 < result.positive_rate < 0.1


class TestRegistry:
    def test_versions_sort_newest_first(self, registry: ModelRegistry) -> None:
        versions = registry.list_versions()
        assert versions == sorted(versions, reverse=True)

    def test_current_version_matches_the_pointer(self, registry: ModelRegistry) -> None:
        assert registry.current_version() in registry.list_versions()

    def test_loading_an_unknown_version_raises(self, registry: ModelRegistry) -> None:
        with pytest.raises(ModelNotFoundError):
            registry.load("20000101T000000-deadbe")

    def test_empty_directory_has_no_current_version(self, tmp_path: Path) -> None:
        assert ModelRegistry(tmp_path / "nothing").current_version() is None

    def test_load_current_on_an_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotFoundError, match="scripts/train.py"):
            ModelRegistry(tmp_path / "nothing").load_current()

    def test_set_current_rejects_an_unknown_version(self, registry: ModelRegistry) -> None:
        with pytest.raises(ModelNotFoundError):
            registry.set_current("nope")

    def test_a_dangling_pointer_falls_back_to_the_newest(
        self, registry: ModelRegistry, tmp_path: Path
    ) -> None:
        import json
        import shutil

        copy_dir = tmp_path / "copy"
        shutil.copytree(registry.artifacts_dir, copy_dir)
        copied = ModelRegistry(copy_dir)
        copied.pointer_path.write_text(json.dumps({"version": "gone"}), encoding="utf-8")

        assert copied.current_version() == copied.list_versions()[0]

    def test_an_unreadable_pointer_falls_back_to_the_newest(
        self, registry: ModelRegistry, tmp_path: Path
    ) -> None:
        import shutil

        copy_dir = tmp_path / "copy2"
        shutil.copytree(registry.artifacts_dir, copy_dir)
        copied = ModelRegistry(copy_dir)
        copied.pointer_path.write_text("not json at all", encoding="utf-8")

        assert copied.current_version() == copied.list_versions()[0]

    def test_versions_are_unique_per_call(self) -> None:
        assert ModelRegistry.new_version() != ModelRegistry.new_version()

    def test_library_versions_are_recorded(self) -> None:
        versions = library_versions()
        assert "python" in versions
        assert "scikit-learn" in versions
        # Present either way: a real version, or the string "unavailable".
        assert "xgboost" in versions


class TestDatasetSpec:
    def test_sample_spec_includes_the_target(self) -> None:
        spec = DatasetSpec.sample()
        assert spec.target_column == "class"
        assert spec.n_features == len(spec.feature_names)
        assert spec.all_columns()[0] == "class"
