#!/usr/bin/env python
"""Train the APS failure model and save it with its measured metrics.

Reports exactly what the run produced - every number printed comes from a fit
on the data you pointed it at. Nothing is a target or a placeholder.

Examples
--------
    # Train on the shipped synthetic sample (generated if absent)
    python scripts/train.py

    # Train on the real APS dataset
    python scripts/train.py --data path/to/aps_failure_training_set.csv

    # Compare candidates without saving an artifact
    python scripts/train.py --no-save
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.ml.evaluation import Evaluator  # noqa: E402
from app.ml.registry import ModelRegistry  # noqa: E402
from app.ml.sampledata import generate_aps_like_frame  # noqa: E402
from app.ml.schema import TARGET_COLUMN  # noqa: E402
from app.ml.training import (  # noqa: E402
    ModelTrainer,
    TrainingResult,
    available_model_factories,
)


def fingerprint(frame: pd.DataFrame) -> str:
    """Stable content hash, so an artifact can be tied to its training data."""
    digest = hashlib.sha256()
    digest.update(",".join(map(str, frame.columns)).encode())
    digest.update(str(frame.shape).encode())
    # Hash a deterministic sample rather than the whole frame: enough to detect
    # "this is different data", cheap on 60k x 170.
    sample = frame.head(500).to_csv(index=False).encode()
    digest.update(sample)
    return digest.hexdigest()[:16]


def load_or_generate(path: Path, *, rows: int, seed: int) -> tuple[pd.DataFrame, str]:
    """Read the CSV, generating the synthetic sample if it is absent."""
    if path.exists():
        frame = pd.read_csv(path, dtype=str, low_memory=False)
        return frame, f"file:{path.name}"

    print(f"[train] {path} not found; generating a synthetic sample ({rows} rows)")
    frame = generate_aps_like_frame(n_rows=rows, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"[train] wrote {path}")
    return frame, f"synthetic:seed={seed},rows={rows}"


def print_report(result: TrainingResult, source: str, synthetic: bool) -> None:
    """Print exactly what the run measured."""
    print()
    print("=" * 78)
    print("TRAINING REPORT")
    print("=" * 78)
    print(f"data source        : {source}")
    if synthetic:
        print("                     *** SYNTHETIC DATA - metrics are not real-world ***")
    print(f"training rows      : {result.training_rows}")
    print(f"held-out rows      : {result.test_rows}")
    print(f"positive rate      : {result.positive_rate:.4%}")
    print(f"features to model  : {len(result.pipeline.feature_names_out_)}")
    print(f"dropped (missing)  : {len(result.cleaner.report_.dropped_high_missing)}")
    print(f"dropped (constant) : {len(result.cleaner.report_.dropped_constant)}")
    print()
    print(
        f"{'candidate':<16}{'cost':>8}{'thresh':>9}{'roc_auc':>10}"
        f"{'pr_auc':>9}{'recall':>9}{'precision':>11}{'f1':>8}{'fit_s':>8}"
    )
    print("-" * 78)
    for item in result.comparison():
        metrics = item["metrics"]
        print(
            f"{item['name']:<16}{item['tuned_cost']:>8}{item['tuned_threshold']:>9}"
            f"{metrics['roc_auc']:>10.4f}{metrics['pr_auc']:>9.4f}"
            f"{metrics['recall']:>9.4f}{metrics['precision']:>11.4f}"
            f"{metrics['f1']:>8.4f}{item['fit_seconds']:>8.2f}"
        )
    print()
    best = result.best
    confusion = best.evaluation.confusion
    print(f"selected           : {result.best_name} (lowest Scania cost)")
    print(f"decision threshold : {result.best_threshold}")
    print(
        f"confusion          : TN={confusion.true_negatives} FP={confusion.false_positives} "
        f"FN={confusion.false_negatives} TP={confusion.true_positives}"
    )
    print(
        f"cost               : {best.evaluation.total_cost} "
        f"({best.evaluation.cost_per_1000_vehicles} per 1000 vehicles)"
    )
    print(f"v1 legacy metrics  : {best.evaluation.legacy_regression}")
    print("                     (regression metrics on a binary target - see README)")
    print("=" * 78)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=settings.training_data_path,
        help="Training CSV. Generated synthetically if it does not exist.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=settings.artifacts_dir,
        help="Where to write the model and metadata",
    )
    parser.add_argument("--rows", type=int, default=20_000, help="Synthetic sample size")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic sample seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out fraction")
    parser.add_argument(
        "--no-tune-threshold",
        action="store_true",
        help="Keep the 0.5 decision threshold instead of minimising Scania cost",
    )
    parser.add_argument("--no-save", action="store_true", help="Benchmark only; write no artifact")
    parser.add_argument("--json", type=Path, default=None, help="Write the report here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level="INFO", json_output=False)

    candidates = available_model_factories()
    print(f"[train] candidates: {', '.join(candidates)}")
    if "xgboost" not in candidates:
        print(
            "[train] WARNING: xgboost is unavailable (on macOS: `brew install libomp`). "
            "Training RandomForest only."
        )

    frame, source = load_or_generate(args.data, rows=args.rows, seed=args.seed)
    if TARGET_COLUMN not in frame.columns:
        print(
            f"[train] ERROR: the data has no '{TARGET_COLUMN}' column; "
            f"found {list(frame.columns)[:8]}...",
            file=sys.stderr,
        )
        return 2

    trainer = ModelTrainer(
        test_size=args.test_size,
        evaluator=Evaluator(),
        tune_threshold=not args.no_tune_threshold,
    )
    result = trainer.train(frame)
    print_report(result, source, synthetic=source.startswith("synthetic"))

    metadata = None
    if not args.no_save:
        registry = ModelRegistry(args.artifacts)
        metadata = registry.save(result, dataset_fingerprint=fingerprint(frame))
        print(f"\n[train] saved version {metadata.version} to {args.artifacts}")
        print(f"[train] model    : {registry.model_path(metadata.version)}")
        print(f"[train] metadata : {registry.metadata_path(metadata.version)}")
    else:
        print("\n[train] --no-save: nothing written")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "source": source,
                    "synthetic": source.startswith("synthetic"),
                    "selected": result.best_name,
                    "decision_threshold": result.best_threshold,
                    "training_rows": result.training_rows,
                    "test_rows": result.test_rows,
                    "positive_rate": result.positive_rate,
                    "candidates": result.comparison(),
                    "version": metadata.version if metadata else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[train] report written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
