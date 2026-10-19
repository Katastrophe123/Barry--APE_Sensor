"""Drift checking against recent traffic, with the outcome persisted."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.db.models import DriftCheck
from app.db.repository import DriftRepository, PredictionRepository
from app.logging_config import get_logger
from app.ml.drift import DriftReport
from app.models.enums import DriftOutcome
from app.services.model_provider import ModelProvider

logger = get_logger(__name__)

#: Per-feature detail stored per check: the worst offenders only.
STORED_FEATURE_DETAIL = 20


class DriftService:
    """Compares recent prediction inputs with the training distribution."""

    def __init__(self, *, models: ModelProvider, min_rows: int = 200) -> None:
        self._models = models
        self.min_rows = min_rows

    @staticmethod
    def outcome_for(report: DriftReport, *, min_rows: int) -> DriftOutcome:
        """Map a report onto a stored verdict.

        A batch too small to judge is ``inconclusive`` rather than ``clean`` -
        recording "no drift" from eight rows would be actively misleading.
        """
        if report.rows_analysed < min_rows or report.features_analysed == 0:
            return DriftOutcome.INCONCLUSIVE
        return DriftOutcome.DRIFT_DETECTED if report.drift_detected else DriftOutcome.CLEAN

    def analyse_frame(self, frame: pd.DataFrame) -> DriftReport:
        """Run the detector over an arbitrary feature frame."""
        predictor = self._models.require_predictor()
        detector = self._models.drift_detector()
        # Replay the training-time cleaning so the batch is comparable.
        return detector.analyse(predictor.prepare_for_drift(frame))

    async def check_recent_predictions(
        self,
        *,
        predictions: PredictionRepository,
        drifts: DriftRepository,
        sample_size: int = 1_000,
        job_id: Any = None,
    ) -> DriftCheck:
        """Sample recent prediction inputs and record a drift check."""
        predictor = self._models.require_predictor()
        payloads = await predictions.recent_features(limit=sample_size)
        frame = pd.DataFrame(payloads) if payloads else pd.DataFrame()

        if frame.empty:
            report = DriftReport(
                rows_analysed=0,
                features_analysed=0,
                model_version=predictor.version,
                note="No predictions have been recorded yet, so there is nothing to compare.",
            )
        else:
            report = self.analyse_frame(frame)

        return await self.record(report, drifts=drifts, job_id=job_id)

    async def record(
        self,
        report: DriftReport,
        *,
        drifts: DriftRepository,
        job_id: Any = None,
    ) -> DriftCheck:
        """Persist a drift report."""
        predictor = self._models.require_predictor()
        outcome = self.outcome_for(report, min_rows=self.min_rows)
        details = report.to_dict(include_features=STORED_FEATURE_DETAIL)

        check = await drifts.record(
            model_version=report.model_version or predictor.version,
            outcome=outcome,
            rows_analysed=report.rows_analysed,
            features_analysed=report.features_analysed,
            drift_share=report.drift_share,
            max_psi=report.max_psi,
            drifted_features=report.drifted_features,
            critical_features=report.critical_features,
            details=details,
            note=report.note,
            job_id=job_id,
        )

        log = logger.warning if outcome is DriftOutcome.DRIFT_DETECTED else logger.info
        log(
            "drift_check_recorded",
            check_id=str(check.id),
            outcome=outcome.value,
            rows=report.rows_analysed,
            drift_share=report.drift_share,
            max_psi=report.max_psi,
            critical=report.critical_features[:5],
            model_version=check.model_version,
        )
        return check
