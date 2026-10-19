"""Dataset conventions for the APS Failure at Scania Trucks problem.

The real dataset (UCI / Scania) has ~170 anonymised numeric sensor columns named
``<two letters>_<index>``, a ``class`` target of ``pos``/``neg``, and missing
values encoded as the literal string ``na``. Those conventions are encoded here.

**The service never hardcodes a feature list.** The canonical feature list for a
deployment is whatever the trained model recorded in its metadata, so pointing
the trainer at the full 170-column dataset works without touching this module.
:data:`SAMPLE_FEATURE_NAMES` describes only the synthetic sample shipped in
``data/sample/`` so the generator, the tests and the docs agree on one schema.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_COLUMN = "class"
POSITIVE_LABEL = "pos"
NEGATIVE_LABEL = "neg"

#: The real dataset encodes missing values as this literal string, not as an
#: empty cell, so pandas reads the column as ``object`` unless told otherwise.
MISSING_TOKENS: tuple[str, ...] = ("na", "NA", "n/a", "", "?")

#: Feature schema of the synthetic sample in ``data/sample/``. The naming
#: follows the real dataset's convention, including a histogram-style family
#: (``ag_000``..``ag_009``) whose bins are correlated by construction.
SAMPLE_FEATURE_NAMES: tuple[str, ...] = (
    "aa_000",
    "ab_000",
    "ac_000",
    "ad_000",
    "ae_000",
    "af_000",
    "ag_000",
    "ag_001",
    "ag_002",
    "ag_003",
    "ag_004",
    "ag_005",
    "ag_006",
    "ag_007",
    "ag_008",
    "ag_009",
    "ah_000",
    "ai_000",
    "aj_000",
    "ak_000",
    "al_000",
    "am_0",
    "an_000",
    "ao_000",
    "ap_000",
    "aq_000",
    "ar_000",
    "as_000",
    "at_000",
    "au_000",
    "av_000",
    "ax_000",
    "ay_000",
    "ay_001",
    "ay_002",
    "az_000",
    "ba_000",
    "bb_000",
    "bc_000",
    "bd_000",
)

#: Official Scania misclassification costs. A missed failure (false negative)
#: means a truck breaks down in service; a false positive means an unnecessary
#: workshop check. The 50x asymmetry is why accuracy is a useless metric here.
COST_FALSE_POSITIVE = 10
COST_FALSE_NEGATIVE = 500


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Which columns a given dataset contributes."""

    feature_names: tuple[str, ...]
    target_column: str = TARGET_COLUMN

    @classmethod
    def sample(cls) -> DatasetSpec:
        return cls(feature_names=SAMPLE_FEATURE_NAMES)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def all_columns(self) -> tuple[str, ...]:
        return (self.target_column, *self.feature_names)
