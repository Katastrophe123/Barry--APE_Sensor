#!/usr/bin/env python
"""Prove the drift detector fires on data it should fire on.

Writes two CSVs into the scheduled input directory - one drawn from the
training distribution, one deliberately shifted - and reports what the detector
measures for each. Useful for demonstrating the automation without waiting for
real sensor drift.

Examples
--------
    python scripts/simulate_drift.py              # report only
    python scripts/simulate_drift.py --write      # also drop CSVs for the scheduler
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.ml.drift import DriftReport  # noqa: E402
from app.ml.sampledata import generate_aps_like_frame, shift_frame_for_drift  # noqa: E402
from app.services.model_provider import ModelProvider  # noqa: E402


def describe(label: str, report: DriftReport) -> None:
    verdict = "DRIFT DETECTED" if report.drift_detected else "clean"
    print(f"\n--- {label}: {verdict} ---")
    print(f"rows analysed      : {report.rows_analysed}")
    print(f"features analysed  : {report.features_analysed}")
    print(f"drifted share      : {report.drift_share:.1%}")
    print(f"max PSI            : {report.max_psi}")
    print(f"critical features  : {report.critical_features[:8] or '(none)'}")
    if report.note:
        print(f"note               : {report.note}")
    print(f"{'feature':<12}{'PSI':>10}{'KS':>9}{'miss delta':>13}  severity")
    for item in report.top(6):
        print(
            f"{item.feature:<12}{item.psi:>10.4f}{item.ks_statistic:>9.4f}"
            f"{item.missing_rate_delta:>13.4f}  {item.severity}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000, help="Rows per batch")
    parser.add_argument("--seed", type=int, default=2024, help="Sample seed")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Also write the CSVs into SCHEDULED_INPUT_DIR for the scheduler to pick up",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(level="WARNING", json_output=False)

    models = ModelProvider(
        settings.artifacts_dir,
        drift_kwargs={
            "psi_significant": settings.drift_psi_significant,
            "psi_critical": settings.drift_psi_critical,
            "drift_share_threshold": settings.drift_share_threshold,
            "min_rows": settings.drift_min_rows,
        },
    )
    if await models.load() is None:
        print(
            f"No model in {settings.artifacts_dir}. Run `python scripts/train.py` first.",
            file=sys.stderr,
        )
        return 2

    predictor = models.require_predictor()
    detector = models.drift_detector()
    print(f"model version      : {predictor.version}")
    print(f"PSI critical       : {settings.drift_psi_critical}")
    print(f"drift share needed : {settings.drift_share_threshold:.0%}")
    print(f"minimum rows       : {settings.drift_min_rows}")

    baseline = generate_aps_like_frame(n_rows=args.rows, seed=args.seed)
    drifted = shift_frame_for_drift(baseline)

    describe(
        "baseline (training-like)",
        detector.analyse(predictor.prepare_for_drift(baseline)),
    )
    describe(
        "shifted (sensors recalibrated)",
        detector.analyse(predictor.prepare_for_drift(drifted)),
    )

    if args.write:
        directory = settings.scheduled_input_dir
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in (("baseline.csv", baseline), ("drifted.csv", drifted)):
            path = directory / name
            frame.drop(columns=["class"]).to_csv(path, index=False)
            print(f"\nwrote {path}")
        print("The scheduled scoring job will pick these up on its next tick.")

    return 0


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
