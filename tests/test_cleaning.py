"""Tests for the data cleaner - the piece the v1 notebook could not reuse."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.cleaning import DataCleaner, DataCleaningError
from app.ml.schema import TARGET_COLUMN


def frame(**columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns)


class TestEncodeTarget:
    def test_maps_pos_and_neg(self) -> None:
        encoded = DataCleaner.encode_target(pd.Series(["neg", "pos", "neg"]))
        assert encoded.tolist() == [0, 1, 0]

    def test_is_case_and_whitespace_insensitive(self) -> None:
        encoded = DataCleaner.encode_target(pd.Series([" POS ", "Neg"]))
        assert encoded.tolist() == [1, 0]

    def test_accepts_already_numeric_labels(self) -> None:
        encoded = DataCleaner.encode_target(pd.Series([0, 1, 1]))
        assert encoded.tolist() == [0, 1, 1]

    def test_accepts_numeric_strings(self) -> None:
        encoded = DataCleaner.encode_target(pd.Series(["0", "1"]))
        assert encoded.tolist() == [0, 1]

    def test_rejects_unknown_labels(self) -> None:
        with pytest.raises(DataCleaningError, match="Unrecognised target labels"):
            DataCleaner.encode_target(pd.Series(["neg", "maybe"]))


class TestFit:
    def test_requires_the_target_column(self) -> None:
        with pytest.raises(DataCleaningError, match=TARGET_COLUMN):
            DataCleaner().fit(frame(aa_000=["1", "2"]))

    def test_na_tokens_become_missing(self) -> None:
        cleaner = DataCleaner()
        features, _ = cleaner.fit_transform(
            frame(
                **{
                    TARGET_COLUMN: ["neg", "pos", "neg", "pos"],
                    "aa_000": ["1", "na", "3", "4"],
                }
            )
        )
        assert features["aa_000"].isna().sum() == 1

    def test_unparseable_values_become_missing_not_errors(self) -> None:
        # Enough distinct values that the column survives the constant check
        # once the bad value is coerced away.
        cleaner = DataCleaner()
        features, _ = cleaner.fit_transform(
            frame(
                **{
                    TARGET_COLUMN: ["neg", "pos", "neg", "pos"],
                    "aa_000": ["1", "oops", "3", "4"],
                }
            )
        )
        assert features["aa_000"].isna().sum() == 1
        assert features["aa_000"].notna().sum() == 3

    def test_drops_columns_above_the_missing_threshold(self) -> None:
        cleaner = DataCleaner(max_missing_fraction=0.5)
        cleaner.fit(
            frame(
                **{
                    TARGET_COLUMN: ["neg", "pos", "neg", "pos"],
                    "good": ["1", "2", "3", "4"],
                    "mostly_missing": ["1", "na", "na", "na"],
                }
            )
        )
        assert cleaner.report_.dropped_high_missing == ["mostly_missing"]
        assert cleaner.feature_names_ == ["good"]

    def test_drops_constant_columns(self) -> None:
        cleaner = DataCleaner()
        cleaner.fit(
            frame(
                **{
                    TARGET_COLUMN: ["neg", "pos", "neg"],
                    "varies": ["1", "2", "3"],
                    "constant": ["5", "5", "5"],
                }
            )
        )
        assert cleaner.report_.dropped_constant == ["constant"]

    def test_all_missing_column_counts_as_constant(self) -> None:
        cleaner = DataCleaner(max_missing_fraction=1.0)
        cleaner.fit(
            frame(
                **{
                    TARGET_COLUMN: ["neg", "pos"],
                    "varies": ["1", "2"],
                    "empty": ["na", "na"],
                }
            )
        )
        assert "empty" in cleaner.report_.dropped_constant

    def test_drops_duplicate_rows(self) -> None:
        cleaner = DataCleaner()
        features, target = cleaner.fit_transform(
            frame(
                **{
                    TARGET_COLUMN: ["neg", "neg", "pos"],
                    "aa_000": ["1", "1", "2"],
                }
            )
        )
        assert cleaner.report_.duplicate_rows_removed == 1
        assert len(features) == len(target) == 2

    def test_duplicate_dropping_can_be_disabled(self) -> None:
        cleaner = DataCleaner(drop_duplicates=False)
        features, _ = cleaner.fit_transform(
            frame(**{TARGET_COLUMN: ["neg", "neg"], "aa_000": ["1", "1"]})
        )
        assert len(features) == 2

    def test_records_the_full_input_feature_set(self) -> None:
        """Dropped columns are still *accepted* at the API boundary."""
        cleaner = DataCleaner()
        cleaner.fit(
            frame(
                **{
                    TARGET_COLUMN: ["neg", "pos", "neg"],
                    "kept": ["1", "2", "3"],
                    "constant": ["5", "5", "5"],
                }
            )
        )
        assert cleaner.input_feature_names_ == ["kept", "constant"]
        assert cleaner.feature_names_ == ["kept"]

    def test_report_tracks_the_class_balance(self) -> None:
        cleaner = DataCleaner()
        cleaner.fit(
            frame(**{TARGET_COLUMN: ["neg"] * 9 + ["pos"], "aa_000": [str(i) for i in range(10)]})
        )
        assert cleaner.report_.positive_rows == 1
        assert cleaner.report_.negative_rows == 9
        assert cleaner.report_.positive_rate == pytest.approx(0.1)

    def test_report_serialises(self) -> None:
        cleaner = DataCleaner()
        cleaner.fit(frame(**{TARGET_COLUMN: ["neg", "pos"], "aa_000": ["1", "2"]}))
        payload = cleaner.report_.to_dict()
        assert payload["kept_feature_count"] == 1
        assert payload["positive_rows"] == 1

    @pytest.mark.parametrize("value", [0.0, 1.5, -0.1])
    def test_rejects_an_invalid_missing_threshold(self, value: float) -> None:
        with pytest.raises(ValueError, match="max_missing_fraction"):
            DataCleaner(max_missing_fraction=value)


class TestTransform:
    @pytest.fixture
    def fitted(self) -> DataCleaner:
        cleaner = DataCleaner()
        cleaner.fit(
            frame(
                **{
                    TARGET_COLUMN: ["neg", "pos", "neg", "pos"],
                    "a": ["1", "2", "3", "4"],
                    "b": ["5", "6", "7", "8"],
                }
            )
        )
        return cleaner

    def test_requires_fitting_first(self) -> None:
        with pytest.raises(DataCleaningError, match="must be fitted"):
            DataCleaner().transform(frame(a=["1"]))

    def test_replays_the_learned_column_order(self, fitted: DataCleaner) -> None:
        out = fitted.transform(frame(b=["1"], a=["2"]))
        assert list(out.columns) == ["a", "b"]

    def test_missing_columns_are_added_as_nan(self, fitted: DataCleaner) -> None:
        out = fitted.transform(frame(a=["1"]))
        assert list(out.columns) == ["a", "b"]
        assert np.isnan(out["b"].iloc[0])

    def test_unknown_extra_columns_are_ignored(self, fitted: DataCleaner) -> None:
        out = fitted.transform(frame(a=["1"], b=["2"], surprise=["3"]))
        assert list(out.columns) == ["a", "b"]

    def test_the_target_column_is_ignored_if_present(self, fitted: DataCleaner) -> None:
        out = fitted.transform(frame(**{TARGET_COLUMN: ["neg"], "a": ["1"], "b": ["2"]}))
        assert TARGET_COLUMN not in out.columns

    def test_rows_are_never_dropped_at_scoring_time(self, fitted: DataCleaner) -> None:
        """Duplicate dropping is a training-only concern."""
        out = fitted.transform(frame(a=["1", "1", "1"], b=["2", "2", "2"]))
        assert len(out) == 3

    def test_output_is_float64(self, fitted: DataCleaner) -> None:
        out = fitted.transform(frame(a=["1"], b=["2"]))
        assert all(dtype == np.float64 for dtype in out.dtypes)
