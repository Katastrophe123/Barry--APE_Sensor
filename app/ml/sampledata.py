"""Synthetic APS-shaped dataset generator.

**This is not the real dataset.** The real APS Failure at Scania Trucks data
(60k rows, ~170 anonymised sensor columns) is not redistributed here. This
generator produces data with the same *shape and pathologies* so the pipeline,
the service and the tests are runnable by anyone cloning the repo:

* severe class imbalance (~1.7% positive, matching the real dataset)
* missing values encoded as the literal string ``na``, at per-column rates
  including a few columns that are mostly missing
* a correlated histogram-style feature family (``ag_000``..``ag_009``)
* constant and near-constant columns, so the cleaner has something to drop
* heavy-tailed counter-like values rather than tidy Gaussians

How the label is generated
--------------------------
Features are generated **first**, then the label is derived from a noisy latent
risk score over a handful of them:

    latent = Σ wᵢ · z(log1p(featureᵢ)) + noise_scale · ε

The top ``positive_rate`` fraction by latent score become failures. ``noise_scale``
controls difficulty: at 0 the problem is trivially separable, and the default of
0.8 puts held-out ROC-AUC around 0.98 with PR-AUC in the 0.5-0.7 range and a
non-zero Scania cost - a problem a model can learn well but not ace, which is
roughly the profile of the real dataset. Missing values are injected
*after* labelling, so missingness destroys some signal exactly as it does in
the real data.

Every metric reported anywhere in this repo that came from this generator is
labelled as synthetic. Point ``scripts/train.py`` at the real CSV for real
numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.schema import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    SAMPLE_FEATURE_NAMES,
    TARGET_COLUMN,
)

#: Features that drive the latent risk score, with their weights. Everything
#: else is pure noise, which is realistic: most APS sensors are irrelevant.
SIGNAL_WEIGHTS: dict[str, float] = {
    "aa_000": 1.0,
    "ac_000": -0.8,
    "ag_003": 0.9,
    "ah_000": 0.6,
    "an_000": -0.7,
    "aq_000": 0.8,
    "ay_001": 0.5,
    "bb_000": -0.5,
}

#: Columns that are mostly missing in the real data too (sensors not fitted to
#: every truck), so the cleaner's missing-fraction rule has work to do.
MOSTLY_MISSING: tuple[str, ...] = ("am_0", "av_000", "bd_000")

#: Columns with no variance at all.
CONSTANT_FEATURES: tuple[str, ...] = ("af_000", "az_000")

#: Default difficulty, calibrated so held-out ROC-AUC lands near 0.98 and the
#: two candidates disagree about which is best - which is the interesting case.
DEFAULT_NOISE_SCALE = 0.8

#: Default sample size. 20k rows at 1.7% positives gives ~340 failures, enough
#: that held-out metrics are not dominated by a handful of rows.
DEFAULT_SAMPLE_ROWS = 20_000


def _zscore(values: np.ndarray) -> np.ndarray:
    """Standardise on a log scale, because the raw columns are heavy-tailed."""
    logged = np.log1p(np.clip(values, 0.0, None))
    std = logged.std()
    return (logged - logged.mean()) / (std if std > 0 else 1.0)


def generate_aps_like_frame(
    *,
    n_rows: int = DEFAULT_SAMPLE_ROWS,
    positive_rate: float = 0.017,
    seed: int = 42,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    feature_names: tuple[str, ...] = SAMPLE_FEATURE_NAMES,
    missing_token: str = "na",
) -> pd.DataFrame:
    """Build a deterministic APS-shaped frame.

    Values are written as strings where missing, exactly as the real CSV does,
    so :class:`~app.ml.cleaning.DataCleaner` is exercised on realistic input.
    """
    if not 0.0 < positive_rate < 0.5:
        raise ValueError("positive_rate must be in (0, 0.5)")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be >= 0")
    rng = np.random.default_rng(seed)

    # --- 1. features, independent of any label --------------------------
    columns: dict[str, np.ndarray] = {}
    for index, name in enumerate(feature_names):
        if name in CONSTANT_FEATURES:
            columns[name] = np.zeros(n_rows)
            continue
        scale = 10.0 ** (1 + index % 4)
        columns[name] = rng.lognormal(mean=0.0, sigma=1.0, size=n_rows) * scale

    # The ag_* family behaves like histogram bins: correlated, sharing a total.
    histogram = [name for name in feature_names if name.startswith("ag_")]
    if histogram:
        totals = rng.lognormal(mean=2.0, sigma=0.8, size=n_rows) * 100
        weights = rng.dirichlet(np.ones(len(histogram)) * 2.0, size=n_rows)
        for position, name in enumerate(histogram):
            columns[name] = totals * weights[:, position]

    # --- 2. labels from a noisy latent function of the features ---------
    latent = np.zeros(n_rows)
    for name, weight in SIGNAL_WEIGHTS.items():
        if name in columns:
            latent += weight * _zscore(columns[name])
    latent += noise_scale * rng.standard_normal(n_rows)

    n_positive = max(1, int(round(n_rows * positive_rate)))
    cutoff = np.partition(latent, -n_positive)[-n_positive]
    labels = (latent >= cutoff).astype(int)

    # --- 3. round to counters, then inject missing tokens ---------------
    as_strings: dict[str, list[str]] = {}
    for name in feature_names:
        values = np.rint(columns[name]).astype("int64")
        if name in MOSTLY_MISSING:
            missing_rate = 0.82
        elif name in CONSTANT_FEATURES:
            missing_rate = 0.0
        else:
            # Deterministic per-column rate between 0% and ~12%.
            missing_rate = float(rng.uniform(0.0, 0.12))
        mask = rng.random(n_rows) < missing_rate
        as_strings[name] = [
            missing_token if is_missing else str(value)
            for value, is_missing in zip(values, mask, strict=True)
        ]

    out = pd.DataFrame(as_strings)
    out.insert(
        0,
        TARGET_COLUMN,
        [POSITIVE_LABEL if label == 1 else NEGATIVE_LABEL for label in labels],
    )
    return out


def shift_frame_for_drift(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...] = ("aa_000", "ac_000", "ag_003", "ah_000"),
    multiplier: float = 6.0,
    extra_missing_rate: float = 0.35,
    seed: int = 7,
    missing_token: str = "na",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with deliberate drift injected.

    Used by the drift tests and by ``scripts/simulate_drift.py`` to prove the
    detector fires on data it should fire on.
    """
    rng = np.random.default_rng(seed)
    drifted = frame.copy()
    for name in features:
        if name not in drifted.columns:
            continue
        numeric = pd.to_numeric(drifted[name], errors="coerce") * multiplier
        mask = rng.random(len(drifted)) < extra_missing_rate
        drifted[name] = [
            missing_token if is_missing else str(int(value))
            for value, is_missing in zip(numeric.fillna(0).to_numpy(), mask, strict=True)
        ]
    return drifted
