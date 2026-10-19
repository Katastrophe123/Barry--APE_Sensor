"""Tests for the predictor, prediction/batch/drift services and notes."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.db.repository import DriftRepository, JobRepository, PredictionRepository
from app.db.session import Database
from app.llm.base import LLMClient, LLMPermanentError, LLMResponse, LLMTransientError
from app.ml.predictor import PredictionError, Predictor
from app.ml.registry import ModelNotFoundError
from app.ml.sampledata import generate_aps_like_frame, shift_frame_for_drift
from app.models.enums import DriftOutcome, JobKind, JobStatus, PredictionSource, RiskBand
from app.models.schemas import PredictionRequest
from app.services.batch_service import BatchError, BatchScoringService
from app.services.drift_service import DriftService
from app.services.insight_service import MaintenanceNoteService
from app.services.model_provider import ModelProvider
from app.services.prediction_service import PredictionService


class FailingLLM(LLMClient):
    """Raises on every call, to drive the template fallback."""

    name = "failing"
    model = "failing-model"

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or LLMTransientError("provider down")
        self.calls = 0

    async def generate(
        self, *, system: str, user: str, temperature: float = 0.0, max_tokens: int = 700
    ) -> LLMResponse:
        self.calls += 1
        raise self._error


class ShortReplyLLM(LLMClient):
    """Returns an unusably short note."""

    name = "short"
    model = "short-model"

    async def generate(
        self, *, system: str, user: str, temperature: float = 0.0, max_tokens: int = 700
    ) -> LLMResponse:
        return LLMResponse(text="ok", model=self.model)


class RecordingLLM(LLMClient):
    """Captures the prompt it was given."""

    name = "recording"
    model = "recording-model"

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    async def generate(
        self, *, system: str, user: str, temperature: float = 0.0, max_tokens: int = 700
    ) -> LLMResponse:
        self.prompts.append((system, user))
        return LLMResponse(
            text="Book the vehicle in for an APS check at the next depot visit.",
            model=self.model,
        )


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class TestPredictor:
    def test_probability_is_a_valid_probability(
        self, predictor: Predictor, sample_features: dict[str, float | None]
    ) -> None:
        outcome = predictor.predict_one(sample_features)
        assert 0.0 <= outcome.failure_probability <= 1.0

    def test_band_never_contradicts_the_flag(self, predictor: Predictor) -> None:
        frame = generate_aps_like_frame(n_rows=300, seed=55)
        scored = predictor.score_batch(frame.drop(columns=["class"]))
        assert (scored["predicted_failure"] == (scored["risk_band"] != "low")).all()

    @pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
    def test_bands_are_relative_to_the_threshold(
        self, predictor: Predictor, probability: float
    ) -> None:
        band = predictor.risk_band(probability)
        if probability < predictor.threshold:
            assert band == "low"
        elif probability >= predictor.threshold * 2:
            assert band == "high"
        else:
            assert band == "medium"

    def test_dropped_training_columns_are_accepted_not_flagged(
        self, predictor: Predictor, sample_features: dict[str, float | None]
    ) -> None:
        """Columns cleaning dropped are still valid input; sending them is fine."""
        outcome = predictor.predict_one(sample_features)
        assert outcome.unexpected_features == ()
        assert len(predictor.accepted_features) > len(predictor.required_features)

    def test_genuinely_unknown_columns_are_reported(
        self, predictor: Predictor, sample_features: dict[str, float | None]
    ) -> None:
        outcome = predictor.predict_one({**sample_features, "not_a_sensor": 1.0})
        assert outcome.unexpected_features == ("not_a_sensor",)

    def test_absent_features_are_reported_and_imputed(
        self, predictor: Predictor, sample_features: dict[str, float | None]
    ) -> None:
        partial = dict(list(sample_features.items())[:6])
        outcome = predictor.predict_one(partial)
        assert outcome.missing_features
        assert 0.0 <= outcome.failure_probability <= 1.0

    def test_null_values_are_accepted(self, predictor: Predictor) -> None:
        nulls = dict.fromkeys(predictor.required_features)
        outcome = predictor.predict_one(nulls)
        assert 0.0 <= outcome.failure_probability <= 1.0

    def test_entirely_unknown_payload_is_rejected(self, predictor: Predictor) -> None:
        with pytest.raises(PredictionError, match="known to the current model"):
            predictor.predict_one({"nonsense": 1.0})

    def test_empty_frame_scores_to_an_empty_array(self, predictor: Predictor) -> None:
        assert predictor.predict_frame(pd.DataFrame()).size == 0

    def test_score_batch_preserves_labels_under_a_new_name(self, predictor: Predictor) -> None:
        frame = generate_aps_like_frame(n_rows=50, seed=56)
        scored = predictor.score_batch(frame)
        assert "actual_class" in scored.columns
        assert "class" not in scored.columns

    def test_score_batch_is_row_aligned(self, predictor: Predictor) -> None:
        frame = generate_aps_like_frame(n_rows=40, seed=57).drop(columns=["class"])
        scored = predictor.score_batch(frame)
        assert len(scored) == len(frame)
        # Asserting *one distinct* version, not constancy for its own sake.
        assert scored["model_version"].nunique() == 1  # noqa: PD101

    def test_outcome_serialises(
        self, predictor: Predictor, sample_features: dict[str, float | None]
    ) -> None:
        payload = predictor.predict_one(sample_features).to_dict()
        assert set(payload) >= {
            "failure_probability",
            "predicted_failure",
            "risk_band",
            "model_version",
        }


# --------------------------------------------------------------------------- #
# ModelProvider
# --------------------------------------------------------------------------- #
class TestModelProvider:
    async def test_loads_the_current_artifact(self, models: ModelProvider) -> None:
        assert models.loaded is True
        assert models.version is not None
        assert models.require_predictor() is not None

    async def test_missing_artifact_loads_as_none_rather_than_raising(
        self, empty_models: ModelProvider
    ) -> None:
        """Booting without a model must be survivable."""
        assert await empty_models.load() is None
        assert empty_models.loaded is False
        assert empty_models.version is None

    def test_require_predictor_raises_with_a_usable_message(
        self, empty_models: ModelProvider
    ) -> None:
        with pytest.raises(ModelNotFoundError, match="scripts/train.py"):
            empty_models.require_predictor()

    async def test_reload_keeps_serving_the_same_version(self, models: ModelProvider) -> None:
        before = models.version
        await models.reload()
        assert models.version == before

    async def test_reload_of_an_empty_directory_returns_none(
        self, empty_models: ModelProvider
    ) -> None:
        assert await empty_models.reload() is None

    async def test_a_corrupt_artifact_does_not_crash_the_load(
        self, models: ModelProvider, tmp_path: Path
    ) -> None:
        import json
        import shutil

        broken_dir = tmp_path / "broken"
        shutil.copytree(models.registry.artifacts_dir, broken_dir)
        version = json.loads((broken_dir / "current.json").read_text())["version"]
        (broken_dir / f"model-{version}.joblib").write_bytes(b"not a joblib file")

        broken = ModelProvider(broken_dir)
        assert await broken.load() is None

    def test_summary_reports_measured_metrics(self, models: ModelProvider) -> None:
        summary = models.summary()
        assert summary.version == models.version
        assert summary.n_features > 0
        assert "roc_auc" in summary.metrics
        assert summary.available_versions

    def test_drift_detector_is_seeded_from_the_model(self, models: ModelProvider) -> None:
        detector = models.drift_detector()
        assert detector.model_version == models.version
        assert detector.training_stats


# --------------------------------------------------------------------------- #
# PredictionService
# --------------------------------------------------------------------------- #
class TestPredictionService:
    async def test_prediction_is_persisted_with_its_model_version(
        self,
        prediction_service: PredictionService,
        predictions: PredictionRepository,
        sample_features: dict[str, float | None],
        models: ModelProvider,
    ) -> None:
        request = PredictionRequest(vehicle_id="TRK-1", features=sample_features)
        prediction = await prediction_service.predict(request, predictions)

        assert prediction.model_version == models.version
        assert prediction.decision_threshold == models.require_predictor().threshold
        assert prediction.vehicle_id == "TRK-1"
        assert prediction.source == PredictionSource.API.value
        assert prediction.latency_ms is not None

    async def test_request_metadata_is_stored(
        self,
        prediction_service: PredictionService,
        predictions: PredictionRepository,
        sample_features: dict[str, float | None],
    ) -> None:
        request = PredictionRequest(features=sample_features, metadata={"depot": "Malmo"})
        prediction = await prediction_service.predict(request, predictions)
        assert prediction.request_metadata == {"depot": "Malmo"}

    async def test_source_is_recorded(
        self,
        prediction_service: PredictionService,
        predictions: PredictionRepository,
        sample_features: dict[str, float | None],
    ) -> None:
        prediction = await prediction_service.predict(
            PredictionRequest(features=sample_features),
            predictions,
            source=PredictionSource.SCHEDULED,
        )
        assert prediction.source == PredictionSource.SCHEDULED.value

    async def test_prediction_is_retrievable(
        self,
        prediction_service: PredictionService,
        predictions: PredictionRepository,
        sample_features: dict[str, float | None],
    ) -> None:
        created = await prediction_service.predict(
            PredictionRequest(features=sample_features), predictions
        )
        fetched = await predictions.get(created.id)
        assert fetched is not None
        assert fetched.failure_probability == created.failure_probability

    async def test_without_a_model_it_raises(
        self,
        empty_models: ModelProvider,
        predictions: PredictionRepository,
        sample_features: dict[str, float | None],
    ) -> None:
        service = PredictionService(models=empty_models)
        with pytest.raises(ModelNotFoundError):
            await service.predict(PredictionRequest(features=sample_features), predictions)


# --------------------------------------------------------------------------- #
# BatchScoringService
# --------------------------------------------------------------------------- #
class TestBatchParsing:
    def test_rejects_an_empty_upload(self, batch_service: BatchScoringService) -> None:
        with pytest.raises(BatchError, match="empty"):
            batch_service.parse_csv(b"   ")

    def test_rejects_a_header_only_file(self, batch_service: BatchScoringService) -> None:
        with pytest.raises(BatchError, match="no data rows"):
            batch_service.parse_csv(b"aa_000,ab_000\n")

    def test_rejects_unparseable_content(self, batch_service: BatchScoringService) -> None:
        with pytest.raises(BatchError):
            batch_service.parse_csv(b"\x00\x01\x02 not a csv")

    def test_rejects_a_file_with_no_matching_columns(
        self, batch_service: BatchScoringService
    ) -> None:
        with pytest.raises(BatchError, match="none of the csv columns|None of the CSV columns"):
            batch_service.parse_csv(b"alpha,beta\n1,2\n")

    def test_rejects_a_file_over_the_row_limit(
        self, models: ModelProvider, database: Database, sample_csv: bytes
    ) -> None:
        service = BatchScoringService(models=models, database=database, max_rows=10)
        with pytest.raises(BatchError, match="the limit is 10"):
            service.parse_csv(sample_csv)

    def test_extracts_the_vehicle_id_column(
        self, batch_service: BatchScoringService, sample_csv: bytes
    ) -> None:
        frame = pd.read_csv(pd.io.common.BytesIO(sample_csv), dtype=str)
        frame.insert(0, "vehicle_id", [f"TRK-{i}" for i in range(len(frame))])

        batch = batch_service.parse_csv(frame.to_csv(index=False).encode())

        assert batch.vehicle_ids[0] == "TRK-0"
        assert "vehicle_id" not in batch.frame.columns

    def test_vehicle_ids_default_to_none(
        self, batch_service: BatchScoringService, sample_csv: bytes
    ) -> None:
        batch = batch_service.parse_csv(sample_csv)
        assert set(batch.vehicle_ids) == {None}

    def test_filename_is_recorded(
        self, batch_service: BatchScoringService, sample_csv: bytes
    ) -> None:
        batch = batch_service.parse_csv(sample_csv, filename="nightly.csv")
        assert batch.filename == "nightly.csv"


class TestBatchScoring:
    async def test_a_batch_scores_every_row(
        self,
        batch_service: BatchScoringService,
        database: Database,
        sample_csv: bytes,
    ) -> None:
        batch = batch_service.parse_csv(sample_csv)
        job_id = await batch_service.create_job(batch)
        await batch_service.run_job(job_id, batch)

        async with database.session() as session:
            job = await JobRepository(session).get(job_id)
            assert job is not None
            assert job.status == JobStatus.SUCCEEDED.value
            assert job.scored_rows == batch.row_count
            assert job.failed_rows == 0
            assert job.duration_ms is not None
            assert job.finished_at is not None

            rows, total = await PredictionRepository(session).list_predictions(
                source=PredictionSource.BATCH, page_size=500
            )
            assert total == batch.row_count
            assert all(row.job_id == job_id for row in rows)

    async def test_summary_records_band_counts(
        self,
        batch_service: BatchScoringService,
        database: Database,
        sample_csv: bytes,
    ) -> None:
        batch = batch_service.parse_csv(sample_csv)
        job_id = await batch_service.create_job(batch)
        await batch_service.run_job(job_id, batch)

        async with database.session() as session:
            job = await JobRepository(session).get(job_id)
            assert job is not None
            assert sum(job.summary["risk_bands"].values()) == batch.row_count
            assert 0.0 <= job.summary["mean_probability"] <= 1.0

    async def test_chunking_does_not_change_the_row_count(
        self,
        models: ModelProvider,
        database: Database,
        sample_csv: bytes,
    ) -> None:
        service = BatchScoringService(models=models, database=database, chunk_size=7)
        batch = service.parse_csv(sample_csv)
        job_id = await service.create_job(batch)
        await service.run_job(job_id, batch)

        async with database.session() as session:
            job = await JobRepository(session).get(job_id)
            assert job is not None
            assert job.scored_rows == batch.row_count

    async def test_a_labelled_file_is_noted_in_the_summary(
        self,
        batch_service: BatchScoringService,
        database: Database,
        training_frame: pd.DataFrame,
    ) -> None:
        labelled = training_frame.head(60).to_csv(index=False).encode()
        batch = batch_service.parse_csv(labelled)
        job_id = await batch_service.create_job(batch)
        await batch_service.run_job(job_id, batch)

        async with database.session() as session:
            job = await JobRepository(session).get(job_id)
            assert job is not None
            assert job.summary["had_labels"] is True

    async def test_a_scoring_failure_is_recorded_not_raised(
        self,
        batch_service: BatchScoringService,
        database: Database,
        sample_csv: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = batch_service.parse_csv(sample_csv)
        job_id = await batch_service.create_job(batch)

        def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("model exploded")

        monkeypatch.setattr(
            batch_service._models.require_predictor(),  # noqa: SLF001
            "predict_frame",
            boom,
        )
        # Must not raise.
        await batch_service.run_job(job_id, batch)

        async with database.session() as session:
            job = await JobRepository(session).get(job_id)
            assert job is not None
            assert job.status == JobStatus.SUCCEEDED.value
            # Every chunk failed, so every row is counted as failed.
            assert job.failed_rows == batch.row_count
            assert job.scored_rows == 0

    async def test_a_missing_job_row_is_handled(
        self, batch_service: BatchScoringService, sample_csv: bytes
    ) -> None:
        batch = batch_service.parse_csv(sample_csv)
        await batch_service.run_job(uuid.uuid4(), batch)  # must not raise

    async def test_job_kind_is_recorded(
        self, batch_service: BatchScoringService, database: Database, sample_csv: bytes
    ) -> None:
        batch = batch_service.parse_csv(sample_csv)
        job_id = await batch_service.create_job(batch)

        async with database.session() as session:
            job = await JobRepository(session).get(job_id)
            assert job is not None
            assert job.kind == JobKind.BATCH_UPLOAD.value
            assert job.status == JobStatus.QUEUED.value


# --------------------------------------------------------------------------- #
# DriftService
# --------------------------------------------------------------------------- #
class TestDriftService:
    def test_outcome_is_inconclusive_below_the_row_floor(self) -> None:
        from app.ml.drift import DriftReport

        report = DriftReport(rows_analysed=50, features_analysed=10)
        assert DriftService.outcome_for(report, min_rows=200) is DriftOutcome.INCONCLUSIVE

    def test_outcome_is_inconclusive_with_no_shared_features(self) -> None:
        from app.ml.drift import DriftReport

        report = DriftReport(rows_analysed=500, features_analysed=0)
        assert DriftService.outcome_for(report, min_rows=200) is DriftOutcome.INCONCLUSIVE

    def test_outcome_maps_the_detector_verdict(self) -> None:
        from app.ml.drift import DriftReport

        clean = DriftReport(rows_analysed=500, features_analysed=10)
        dirty = DriftReport(rows_analysed=500, features_analysed=10, drift_detected=True)
        assert DriftService.outcome_for(clean, min_rows=200) is DriftOutcome.CLEAN
        assert DriftService.outcome_for(dirty, min_rows=200) is DriftOutcome.DRIFT_DETECTED

    async def test_no_predictions_records_an_inconclusive_check(
        self,
        drift_service: DriftService,
        predictions: PredictionRepository,
        drifts: DriftRepository,
    ) -> None:
        check = await drift_service.check_recent_predictions(predictions=predictions, drifts=drifts)
        assert check.outcome == DriftOutcome.INCONCLUSIVE.value
        assert check.rows_analysed == 0
        assert "nothing to compare" in check.note

    async def test_training_like_traffic_is_clean(
        self,
        drift_service: DriftService,
        prediction_service: PredictionService,
        predictions: PredictionRepository,
        drifts: DriftRepository,
    ) -> None:
        frame = generate_aps_like_frame(n_rows=250, seed=321).drop(columns=["class"])
        for _, row in frame.iterrows():
            features = {
                key: (None if value == "na" else float(value))
                for key, value in row.to_dict().items()
            }
            await prediction_service.predict(PredictionRequest(features=features), predictions)

        check = await drift_service.check_recent_predictions(predictions=predictions, drifts=drifts)

        assert check.outcome == DriftOutcome.CLEAN.value
        assert check.rows_analysed == 250

    async def test_shifted_traffic_is_flagged(
        self,
        drift_service: DriftService,
        prediction_service: PredictionService,
        predictions: PredictionRepository,
        drifts: DriftRepository,
    ) -> None:
        frame = shift_frame_for_drift(generate_aps_like_frame(n_rows=250, seed=321)).drop(
            columns=["class"]
        )
        for _, row in frame.iterrows():
            features = {
                key: (None if value == "na" else float(value))
                for key, value in row.to_dict().items()
            }
            await prediction_service.predict(PredictionRequest(features=features), predictions)

        check = await drift_service.check_recent_predictions(predictions=predictions, drifts=drifts)

        assert check.outcome == DriftOutcome.DRIFT_DETECTED.value
        assert check.critical_features

    async def test_analyse_frame_uses_the_models_cleaning(
        self, drift_service: DriftService
    ) -> None:
        frame = generate_aps_like_frame(n_rows=200, seed=9)
        report = drift_service.analyse_frame(frame)
        assert report.features_analysed > 0

    async def test_checks_are_queryable(
        self,
        drift_service: DriftService,
        predictions: PredictionRepository,
        drifts: DriftRepository,
    ) -> None:
        await drift_service.check_recent_predictions(predictions=predictions, drifts=drifts)
        latest = await drifts.latest()
        assert latest is not None
        rows, total = await drifts.list_checks()
        assert total == 1
        assert rows[0].id == latest.id


# --------------------------------------------------------------------------- #
# MaintenanceNoteService
# --------------------------------------------------------------------------- #
class TestMaintenanceNoteService:
    async def test_uses_the_llm_when_it_works(self, fake_sleep: Any) -> None:
        llm = RecordingLLM()
        service = MaintenanceNoteService(llm=llm, sleep=fake_sleep)

        note = await service.generate(
            failure_probability=0.8, risk_band=RiskBand.HIGH, vehicle_id="TRK-9"
        )

        assert note.generated_by == "llm"
        assert note.provider == "recording"
        assert "APS check" in note.text

    async def test_prompt_carries_the_prediction_context(self, fake_sleep: Any) -> None:
        llm = RecordingLLM()
        service = MaintenanceNoteService(llm=llm, sleep=fake_sleep)

        await service.generate(
            failure_probability=0.83,
            risk_band=RiskBand.HIGH,
            vehicle_id="TRK-9",
            top_signals=["aa_000 = 12345"],
            model_version="v-test",
        )

        _, user = llm.prompts[0]
        assert "TRK-9" in user
        assert "83.0%" in user
        assert "high" in user
        assert "aa_000 = 12345" in user
        assert "v-test" in user

    async def test_falls_back_to_the_template_when_the_llm_fails(self, fake_sleep: Any) -> None:
        llm = FailingLLM()
        service = MaintenanceNoteService(llm=llm, max_attempts=2, sleep=fake_sleep)

        note = await service.generate(
            failure_probability=0.9, risk_band=RiskBand.HIGH, vehicle_id="TRK-3"
        )

        assert note.generated_by == "template"
        assert "TRK-3" in note.text
        assert "90.0%" in note.text
        assert llm.calls == 2

    async def test_permanent_errors_are_not_retried(self, fake_sleep: Any) -> None:
        llm = FailingLLM(LLMPermanentError("bad key"))
        service = MaintenanceNoteService(llm=llm, max_attempts=3, sleep=fake_sleep)

        note = await service.generate(failure_probability=0.9, risk_band=RiskBand.HIGH)

        assert note.generated_by == "template"
        assert llm.calls == 1

    async def test_an_unusably_short_reply_falls_back(self, fake_sleep: Any) -> None:
        service = MaintenanceNoteService(llm=ShortReplyLLM(), max_attempts=1, sleep=fake_sleep)
        note = await service.generate(failure_probability=0.7, risk_band=RiskBand.MEDIUM)
        assert note.generated_by == "template"

    @pytest.mark.parametrize(
        ("band", "fragment"),
        [
            (RiskBand.HIGH, "highest-priority"),
            (RiskBand.MEDIUM, "next depot visit"),
            (RiskBand.LOW, "No APS action is required"),
        ],
    )
    def test_template_respects_the_band(self, band: RiskBand, fragment: str) -> None:
        note = MaintenanceNoteService.template_note(
            failure_probability=0.5, risk_band=band, vehicle_id=None, top_signals=[]
        )
        assert fragment in note

    def test_template_mentions_supplied_signals(self) -> None:
        note = MaintenanceNoteService.template_note(
            failure_probability=0.9,
            risk_band=RiskBand.HIGH,
            vehicle_id="TRK-1",
            top_signals=["aa_000 = 999"],
        )
        assert "aa_000 = 999" in note

    def test_top_signals_are_largest_magnitude_first(self) -> None:
        signals = MaintenanceNoteService.top_signals_from_features(
            {"a": 1.0, "b": -500.0, "c": 50.0, "d": None}, limit=2
        )
        assert signals[0].startswith("b =")
        assert signals[1].startswith("c =")
        assert len(signals) == 2

    def test_top_signals_skip_nulls(self) -> None:
        assert MaintenanceNoteService.top_signals_from_features({"a": None}) == []

    async def test_signals_are_capped(self, fake_sleep: Any) -> None:
        llm = RecordingLLM()
        service = MaintenanceNoteService(llm=llm, sleep=fake_sleep)

        await service.generate(
            failure_probability=0.5,
            risk_band=RiskBand.MEDIUM,
            top_signals=[f"f{index} = {index}" for index in range(50)],
        )

        _, user = llm.prompts[0]
        assert user.count(" = ") <= MaintenanceNoteService.MAX_SIGNALS
