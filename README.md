# APS Failure Prediction Service

[![CI](https://github.com/kanishksingh-arch/aps-failure-prediction-service/actions/workflows/ci.yml/badge.svg)](https://github.com/kanishksingh-arch/aps-failure-prediction-service/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://mypy-lang.org/)

A heavy-truck **air pressure system (APS) failure prediction** service: the
notebook benchmark turned into a tested Python package, an async **FastAPI**
inference API, **PostgreSQL** audit trail, scheduled batch scoring and
**data-drift detection**.

---

## Problem statement

A Scania truck's APS failure is expensive in one direction and cheap in the
other. The dataset's official cost metric says so explicitly:

| Mistake | Cost | Meaning |
|---|---|---|
| False positive | **10** | An unnecessary workshop check |
| False negative | **500** | A truck breaks down in service |

A missed failure costs **50 times** a needless inspection, on a dataset that is
only ~1.7% positive. That single fact invalidates most of how the v1 notebook
evaluated the problem.

### What v1 did, and what was wrong with it

v1 was a notebook: clean the sensor data, then benchmark Random Forest against
XGBoost on **R², RMSE and MAE**. It is preserved at
[`notebooks/v1_aps_benchmark.ipynb`](notebooks/v1_aps_benchmark.ipynb).

| v1 problem | Why it mattered | What this service does |
|---|---|---|
| **Regression metrics on a classification target.** R², RMSE, MAE on a 0/1 label | R² on a binary outcome is not interpretable, and "never predicts a failure" — the most useless possible model — *wins on MAE*. The notebook has a cell demonstrating exactly that | Reports ROC-AUC, PR-AUC, precision, recall, F1, the confusion matrix and the **Scania cost**; selects on cost. The three v1 metrics are still computed under `legacy_regression` so the numbers stay comparable |
| **Decision threshold left at 0.5** | At a 50:1 cost ratio the cost-optimal threshold is far lower; 0.5 discards most of the model's value | The threshold is **tuned to minimise cost** on the held-out split and stored with the model (it came out at **0.04**, not 0.5) |
| **Cleaning decisions lived in notebook cells** | Which columns were dropped, and the imputation medians, existed only in the kernel. A model trained there cannot be served faithfully | `DataCleaner` and `FeaturePipeline` are fit/transform classes serialised *with* the model, so scoring replays exactly what training did |
| **No model artifact, version or metadata** | No way to answer "which model made this prediction?" | Every run writes `model-<version>.joblib` + `metadata-<version>.json`; every stored prediction records its model version and threshold |
| **No monitoring** | Sensor recalibration would silently degrade predictions | PSI + KS + missing-rate drift detection against the recorded training distribution, on a schedule and on demand |
| **No tests** | — | 349 tests, 97% coverage |

---

## Architecture

```mermaid
flowchart TB
    subgraph clients["Clients"]
        OP["Operator / fleet system"]
        CSV["Nightly CSV drop"]
    end

    subgraph api["FastAPI service"]
        MW["RequestContextMiddleware<br/>(request_id, structured logs)"]
        AUTH["JWT bearer auth<br/>(all routes except /health, /auth/token)"]
        R1["POST /predict"]
        R2["POST /predict/batch → 202"]
        R3["GET /jobs/:id"]
        R4["GET /models/current"]
        R5["POST /drift/check"]
        R6["POST /insights/maintenance-note"]
    end

    subgraph worker["Scheduler container (APScheduler)"]
        J1["nightly: score dropped CSVs"]
        J2["nightly: drift check"]
    end

    subgraph ml["app/ml — the tested package"]
        CL["DataCleaner<br/>(fit/transform)"]
        FP["FeaturePipeline<br/>(impute, prune, record stats)"]
        TR["ModelTrainer<br/>(RF vs XGBoost)"]
        EV["Evaluator<br/>(Scania cost + ROC/PR)"]
        REG["ModelRegistry<br/>(artifact + metadata)"]
        PR["Predictor"]
        DR["DriftDetector<br/>(PSI / KS / missing-rate)"]
    end

    LLM["LLMClient<br/>openai | huggingface | stub<br/>(maintenance notes only)"]
    DB[("PostgreSQL<br/>predictions · jobs · drift_checks")]
    ART[("artifacts/<br/>model + metadata")]

    OP --> MW --> AUTH
    AUTH --> R1 --> PR
    AUTH --> R2 --> PR
    AUTH --> R3
    AUTH --> R4 --> REG
    AUTH --> R5 --> DR
    AUTH --> R6 --> LLM

    CSV --> J1 --> PR
    J2 --> DR

    CL --> FP --> TR --> EV --> REG
    REG --> ART
    ART --> PR
    ART -->|training stats| DR

    PR -->|prediction + model version| DB
    DR -->|drift verdict| DB
    J1 -->|job status| DB
```

### Training flow

```
raw CSV ─▶ DataCleaner.fit ─▶ stratified split ─▶ FeaturePipeline.fit
                                                        │
                        ┌───────────────────────────────┴──────────┐
                        ▼                                          ▼
                 RandomForest                                  XGBoost
                        │                                          │
                        └──────────▶ Evaluator ◀───────────────────┘
                                        │
                     tune threshold to minimise Scania cost
                                        │
                      select the cheapest ─▶ ModelRegistry.save
                                                   │
                                   model-<v>.joblib + metadata-<v>.json
```

### Project layout

```
app/
├── ml/                  # the package the notebook became
│   ├── schema.py         # dataset conventions + Scania costs
│   ├── cleaning.py       # DataCleaner (fit/transform)
│   ├── features.py       # FeaturePipeline + training FeatureStats
│   ├── training.py       # ModelTrainer, candidate registry
│   ├── evaluation.py     # Evaluator (cost + classification + legacy metrics)
│   ├── registry.py       # ModelRegistry, ModelMetadata
│   ├── predictor.py      # inference wrapper
│   ├── drift.py          # DriftDetector (PSI / KS / missing-rate)
│   └── sampledata.py     # synthetic APS-shaped generator
├── api/                 # routers, deps, error envelope, request-id middleware
├── services/            # prediction, batch, drift, notes, model provider
├── db/                  # SQLAlchemy models, async session, repositories
├── llm/                 # swappable client + retry (maintenance notes only)
├── models/              # Pydantic schemas, enums
├── scheduler.py         # APScheduler jobs
├── config.py            # pydantic-settings
└── main.py
notebooks/               # v1_aps_benchmark.ipynb, for reference
data/sample/             # committed synthetic training CSV
scripts/                 # train.py, simulate_drift.py
tests/                   # 349 tests, fully offline
```

---

## Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ (type hints throughout, `mypy --disallow-untyped-defs`) |
| API | FastAPI 0.143, async/await, Pydantic v2 |
| ML | scikit-learn 1.9 (RandomForest), XGBoost 3.4, pandas 3.0, NumPy 2.5 |
| Drift | PSI over recorded training quantiles + SciPy KS + missing-rate delta |
| Persistence | PostgreSQL 16, SQLAlchemy 2.1 async (asyncpg), Alembic |
| Scheduling | APScheduler 3.11 (cron triggers, in-process with the model) |
| Auth | JWT (HS256) via PyJWT |
| Serialisation | joblib for the model bundle, JSON for metadata |
| LLM (optional) | OpenAI-compatible / Hugging Face / offline stub |
| Logging | structlog, JSON in production, `request_id` + `job_id` context |
| Tests | pytest, pytest-asyncio, pytest-cov; SQLite in-memory + synthetic data |
| Quality | ruff (lint + format), mypy, GitHub Actions |

---

## Setup with Docker

`docker compose up` starts PostgreSQL, the API and a scheduler worker. It
applies migrations **and trains a first model** on the committed sample, so
`/predict` works immediately rather than returning 503. No API key is needed.

```bash
git clone https://github.com/kanishksingh-arch/aps-failure-prediction-service.git
cd aps-failure-prediction-service
cp .env.example .env          # optional; compose has safe local defaults
docker compose up --build     # first boot trains a model (~1 min)
```

- API docs: <http://localhost:8000/docs>
- PostgreSQL: `localhost:5435`
- Scheduler worker: internal, runs scoring + drift every 5 minutes in dev

### Running locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# macOS only: XGBoost's wheel needs an OpenMP runtime
brew install libomp

docker compose up -d db       # just PostgreSQL, on host port 5435
export DATABASE_URL="postgresql+asyncpg://aps:aps@localhost:5435/aps"
alembic upgrade head

python scripts/train.py       # writes artifacts/
uvicorn app.main:app --reload
```

If `libomp` is unavailable the trainer **degrades to RandomForest only** and
says so, rather than failing the run.

### Using the real dataset

The real **APS Failure at Scania Trucks** data (~60k rows, 171 columns) is not
redistributed here. Download it, then:

```bash
python scripts/train.py --data path/to/aps_failure_training_set.csv
```

Nothing is hardcoded to the sample's schema: the feature list the service
accepts is whatever the trained model recorded in its metadata.

---

## API examples

Everything except `/health` and `/auth/token` needs a bearer token.

### 1. Sign in

```bash
TOKEN=$(curl -sS -X POST http://localhost:8000/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"operator","password":"operator"}' | jq -r .access_token)
```

### 2. Health (no auth)

```bash
curl -sS http://localhost:8000/health | jq
```

```json
{
  "status": "ok",
  "app": "aps-failure-prediction-service",
  "environment": "local",
  "version": "1.0.0",
  "database": "up",
  "model_loaded": true,
  "model_version": "20261011T122659-0c3732",
  "llm_provider": "stub",
  "scheduler_running": false
}
```

A missing model is reported as `model_loaded: false` with a **200** — an
untrained deployment is alive and can accept an artifact. Only the database
being unreachable is a 503.

### 3. The live model and its measured metrics

```bash
curl -sS http://localhost:8000/models/current -H "Authorization: Bearer $TOKEN" \
  | jq '{version, model_type, decision_threshold, n_features, training_rows,
         metrics: {roc_auc: .metrics.roc_auc, pr_auc: .metrics.pr_auc,
                   recall: .metrics.recall, total_cost: .metrics.total_cost},
         candidates: [.candidate_comparison[] | {name, tuned_cost}]}'
```

### 4. Score one record

```bash
curl -sS -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{
        "vehicle_id": "TRK-00421",
        "features": {"aa_000": 76698, "ab_000": 0, "ac_000": 2130706438,
                     "ag_003": 0, "ah_000": 1240520}
      }' | jq
```

```json
{
  "id": "0b0f...",
  "vehicle_id": "TRK-00421",
  "failure_probability": 0.0525,
  "predicted_failure": true,
  "risk_band": "medium",
  "decision_threshold": 0.04,
  "model_version": "20261011T122659-0c3732",
  "source": "api",
  "missing_features": ["ab_000", "..."],
  "unexpected_features": [],
  "latency_ms": 11
}
```

Features arrive as a **mapping**, not 170 declared fields: the real schema is
wide and anonymised, and the authoritative list is whatever the loaded model
recorded. Absent features are median-imputed and reported back in
`missing_features`; genuinely unknown names come back in `unexpected_features`.
Columns that *cleaning* dropped at training time are accepted silently — sending
them is harmless.

**Risk bands are relative to the model's own threshold**, so they can never
contradict `predicted_failure`:

| Band | Condition |
|---|---|
| `low` | below the threshold (not predicted to fail) |
| `medium` | threshold .. 2× threshold |
| `high` | ≥ 2× threshold (act first) |

### 5. Batch scoring (background job)

```bash
JOB=$(curl -sS -X POST http://localhost:8000/predict/batch \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@fleet_snapshot.csv" | jq -r .job_id)

curl -sS "http://localhost:8000/jobs/$JOB" -H "Authorization: Bearer $TOKEN" | jq
```

```json
{
  "id": "6a1c...",
  "kind": "batch_upload",
  "status": "succeeded",
  "source_filename": "fleet_snapshot.csv",
  "model_version": "20261011T122659-0c3732",
  "total_rows": 400,
  "scored_rows": 400,
  "failed_rows": 0,
  "high_risk_rows": 15,
  "duration_ms": 231,
  "summary": {
    "risk_bands": { "low": 348, "medium": 37, "high": 15 },
    "mean_probability": 0.019342,
    "rows_per_second": 1731.6,
    "had_labels": true
  }
}
```

Returns **202** immediately — a 60k-row file takes long enough that holding the
connection would be the wrong design. A `vehicle_id` / `truck_id` / `id` column
is used as the vehicle identifier if present.

### 6. The audit trail

```bash
# Recent high-risk vehicles
curl -sS "http://localhost:8000/predictions?risk_band=high&page_size=5" \
  -H "Authorization: Bearer $TOKEN" | jq '{total, items: [.items[] | {vehicle_id, failure_probability}]}'

# Everything a given model version predicted
curl -sS "http://localhost:8000/predictions?model_version=20261011T122659-0c3732" \
  -H "Authorization: Bearer $TOKEN" | jq .total
```

Filterable by `risk_band`, `model_version`, `vehicle_id`, `source`
(`api` / `batch` / `scheduled`) and `predicted_failure`.

### 7. Drift check

```bash
curl -sS -X POST http://localhost:8000/drift/check -H "Authorization: Bearer $TOKEN" \
  | jq '{outcome, rows_analysed, drift_share, max_psi, critical_features}'
```

```json
{
  "outcome": "drift_detected",
  "rows_analysed": 1000,
  "drift_share": 0.114286,
  "max_psi": 5.822027,
  "critical_features": ["aa_000", "ac_000", "ag_003", "ah_000"]
}
```

### 8. Plain-English maintenance note (the optional LLM step)

```bash
curl -sS -X POST http://localhost:8000/insights/maintenance-note \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"prediction_id": "0b0f..."}' | jq
```

```json
{
  "failure_probability": 0.91,
  "risk_band": "high",
  "note": "Book this vehicle in for an air pressure system inspection before its next long run. ...",
  "generated_by": "llm",
  "llm_provider": "stub",
  "llm_model": "stub-notes-v1"
}
```

**This endpoint never fails because of the LLM.** If the provider is down,
unauthorised or slow, a deterministic template is used and `generated_by`
says `template`. The prediction is the product; the note is a convenience.

### Error envelope

```json
{
  "error": {
    "code": "service_unavailable",
    "message": "No model is loaded from artifacts. Run `python scripts/train.py` and restart, or POST /models/reload.",
    "details": { "model_loaded": false }
  },
  "request_id": "b09325f3278a4206af3acec330c63c20"
}
```

| Status | When |
|---|---|
| `200` | Reads, token issuing, model reload, maintenance note |
| `201` | Prediction stored, drift check recorded |
| `202` | Batch accepted for background scoring |
| `400` | The uploaded CSV is unparseable or has no matching columns |
| `401` | Missing, malformed, expired or wrongly-signed token |
| `404` | Unknown prediction / job / drift check |
| `413` | Upload exceeds `MAX_UPLOAD_BYTES` |
| `422` | Payload or query parameters failed validation |
| `500` | Unexpected error (internal detail logged, never returned) |
| `503` | Database unreachable (`/health`), or **no trained model** on a prediction route |

---

## Measured results

Everything below was produced by `python scripts/train.py` on the committed
sample (macOS, Python 3.12.13, scikit-learn 1.9.1, xgboost 3.4.1) and is
reproducible with that command.

Expect the last digits to move slightly on a different platform or thread
count: RandomForest probabilities are averages over trees, so the order of
parallel float reduction changes them in the final bits, and with a decision
threshold as low as 0.04 that can flip a handful of borderline rows. Running
the same command inside the Docker image (Linux, `OMP_NUM_THREADS=2`) produced
`cost=3560, recall=1.0000, roc_auc=0.9828` — the same picture, different
rounding. The authoritative record for any given run is the
`artifacts/metadata-<version>.json` it wrote.

> ### ⚠️ These numbers are from synthetic data
> The real Scania dataset is not redistributed here. `data/sample/` contains a
> **synthetic generator output** with the same shape and pathologies (~1.7%
> positive, `na`-encoded missingness, heavy-tailed counters, a correlated
> histogram family, constant columns). The numbers therefore demonstrate that
> the *pipeline and the cost-based selection work*; they say nothing about
> real-world accuracy. Run with `--data` pointing at the real CSV for real
> numbers.

```
data source        : synthetic:seed=42,rows=20000
training rows      : 16000
held-out rows      : 4000
positive rate      : 1.7000%
features to model  : 35   (3 dropped for missingness, 2 constant)

candidate           cost   thresh   roc_auc   pr_auc   recall  precision      f1   fit_s
------------------------------------------------------------------------------
random_forest       4640     0.04    0.9825   0.5518   0.9706     0.1535  0.2651    0.95
xgboost             5850     0.01    0.9861   0.7154   0.8676     0.3041  0.4504    1.06

selected           : random_forest (lowest Scania cost)
decision threshold : 0.04
confusion          : TN=3568 FP=364 FN=2 TP=66
cost               : 4640 (1160.0 per 1000 vehicles)
v1 legacy metrics  : {'r2': 0.235, 'rmse': 0.113, 'mae': 0.029}
```

### The interesting part: AUC and cost disagree

XGBoost wins on **ROC-AUC (0.9861 vs 0.9825), PR-AUC (0.715 vs 0.552) and F1
(0.450 vs 0.265)**. RandomForest wins on **cost (4640 vs 5850)** — and cost is
what gets deployed.

Why: RandomForest caught **66 of 68** failures, XGBoost caught 59. Those 7 extra
misses cost `7 × 500 = 3500`, while RandomForest's 229 extra false alarms cost
`229 × 10 = 2290`. Picking the better-AUC model would have been the more
expensive decision.

This is the whole argument for the upgrade in one table: v1's metrics could not
have told you this, and neither can AUC alone.

### Drift detection, both directions

From `python scripts/simulate_drift.py` on the same artifact:

| Batch | Verdict | Drifted share | Max PSI | Critical features |
|---|---|---|---|---|
| Training-like | **clean** | 0.0% | 0.056 | — |
| 4 sensors rescaled ×6 + 35% more missing | **drift detected** | 11.4% | 5.822 | `aa_000`, `ac_000`, `ag_003`, `ah_000` |

---

## Running the tests

Fully offline: SQLite in-memory, synthetic data, stubbed LLM. No API key, no
PostgreSQL, no network. The model is trained **once per session** and reused.

```bash
pytest                                              # 349 tests
pytest --cov=app --cov-report=term-missing          # coverage report
pytest tests/test_drift.py -v                       # just the drift detector
ruff check . && ruff format --check . && mypy app scripts
```

Measured on the committed code (Python 3.12.13, `pytest 9.1.1`):

```
349 passed in 28.35s
TOTAL  2310 statements   69 missed   97% coverage
```

| Area | Examples |
|---|---|
| Preprocessing | `na`/unparseable coercion, missingness and constant-column dropping, duplicate removal, **learned decisions replayed at transform time**, rows never dropped at scoring time |
| Features | median imputation, post-imputation variance pruning, column alignment/reordering, training stats computed pre-imputation |
| Evaluation | Scania cost weights, the 50:1 ratio, threshold tuning beating the 0.5 default, single-class folds not raising, legacy metrics namespaced |
| Training | selection follows **cost not AUC**, and the cost metric's **break-even positive rate** (below ~1.96%, "do nothing" is genuinely cheaper than "flag everything") |
| Registry | artifact round-trip, dangling/unreadable `current.json` pointers, unknown versions |
| Drift | PSI ≈ 0 on identical data and > 1 on shifted; missing-rate jumps; one critical feature flags a batch while one mild mover does not; small batches reported but not flagged; end-to-end against the real artifact |
| Prediction API | 201 shape, model version recorded, unknown vs dropped features, null values, **503 when untrained**, filters, pagination |
| Batch jobs | full-file scoring, chunk boundaries, per-chunk failure counted not fatal, missing job row handled, `vehicle_id` extraction, 400/413 paths |
| Scheduler | cron parsing, idempotent archiving (running twice does not rescore), one bad file does not stop the others, failures recorded as failed jobs, skipped when untrained |
| LLM | provider swap by config alone, 4xx→permanent / 5xx→transient, retry backoff asserted, **template fallback when the provider fails**, band-appropriate notes |

---

## Data model

```
predictions                          jobs                        drift_checks
──────────────────────────────       ────────────────────        ─────────────────────────
id                  uuid PK          id            uuid PK       id                uuid PK
vehicle_id          varchar idx      kind          varchar idx   model_version     varchar idx
features            jsonb            status        varchar idx   outcome           varchar idx
request_metadata    jsonb            source_filename varchar     rows_analysed     int
missing_features    jsonb            model_version varchar idx   features_analysed int
unexpected_features jsonb            total_rows    int           drift_share       float
failure_probability float            scored_rows   int           max_psi           float
predicted_failure   bool     idx     failed_rows   int           drifted_features  jsonb
risk_band           varchar  idx     high_risk_rows int          critical_features jsonb
decision_threshold  float            error         text          details           jsonb
model_version       varchar  idx     duration_ms   int           note              text
source              varchar  idx     summary       jsonb         job_id            uuid FK
job_id              uuid FK  idx     created_at    timestamptz   created_at        timestamptz
latency_ms          int              started_at    timestamptz
created_at          timestamptz idx  finished_at   timestamptz
```

Indexes follow the queries the service actually issues:

- `(risk_band, created_at)` — "recent high-risk vehicles", the operator query.
- `(model_version, created_at)` — "what has this artifact been predicting", the
  ML-engineer query.
- `(status, created_at)` — job polling.
- `(outcome, created_at)` — drift over time.

`JSONB` on PostgreSQL degrades to plain `JSON` on SQLite via `with_variant`,
which is what lets the test suite run with no database server.

---

## Automation

Two cron jobs, run by a dedicated `worker` container (APScheduler), both
disabled by default (`SCHEDULER_ENABLED=false`) so tests and local runs get no
surprise background writes.

**Nightly batch scoring** watches `SCHEDULED_INPUT_DIR` for CSVs, scores each
one, and **moves it to `SCHEDULED_OUTPUT_DIR`**. The move is what makes the
schedule idempotent — running twice does not rescore, and there is a test for
that. One malformed file is recorded as a failed job and the others still run.

**Nightly drift check** samples the most recent predictions' *inputs* and
compares them with the training distribution, recording a row whatever it finds.

Both appear in `GET /jobs`, so a failure is visible rather than silent.

---

## Design decisions and trade-offs

**Model selection by cost, not by AUC.** With a 50:1 false-negative penalty,
ranking metrics are the wrong tiebreak — and on this data they actively point at
the more expensive model (see "Measured results"). Cost is the number the
business carries.

**The threshold is part of the model.** A cost-tuned threshold is stored in the
metadata and applied at serving time. Shipping a model without its threshold
ships half a decision.

**`DataCleaner` is a fit/transform class, not a function.** This is the single
most important structural change from v1. "Drop columns with >70% missing" is a
*learned* decision; replaying it at inference is the only way serving matches
training. Same for the imputer's medians.

**Training statistics are recorded at fit time, pre-imputation.** The drift
detector drops NaN from an incoming batch before binning it, so the reference
distribution must exclude imputed medians too. Computing them post-imputation
inflated baseline PSI from 0.046 to 0.187 — manufacturing drift that was not
there.

**`min_rows` for a drift verdict is 200, and that number is measured.** PSI over
10 bins is badly behaved on small samples. Drawing two samples from the *same*
distribution gives a median PSI of 0.347 at 30 rows — above the 0.25
"significant" threshold — versus 0.042 at 200 rows. The table is in
`app/ml/drift.py`. Smaller batches are still measured and returned, but recorded
as `inconclusive` rather than `clean`, because "no drift, from 8 rows" is a lie.

**Two drift rules, not one.** A batch flags if ≥10% of features drift *or* any
single feature's PSI reaches 1.0. The share rule alone misses "four critical
sensors were recalibrated"; the per-feature rule alone cries wolf on one noisy
column.

**No feature scaling.** Both candidates are tree ensembles, invariant to
monotone rescaling, so a `StandardScaler` would add a fitted artefact that
changes nothing and still has to be versioned. `app/ml/features.py` is the one
place to change if a linear model is ever added.

**Batch scoring is a `202` plus a job row.** Scoring 60k rows takes long enough
that holding an HTTP connection would be wrong. Rows are written in chunks, and
a chunk that fails to score is *counted*, not fatal — one malformed row should
not throw away the other 59,999.

**A missing model is a 503, not a 500, and not a failed boot.** "Not trained
yet" is a deployment state with an actionable message, not a bug. The service
starts, `/health` reports `model_loaded: false`, and `POST /models/reload` picks
up an artifact without a restart. A *bad* new artifact leaves the previous model
serving.

**XGBoost is a runtime capability, not an assumption.** Its wheel needs a native
OpenMP runtime (`libomp` on macOS, `libgomp1` in the image). The trainer checks
whether it can load and falls back to RandomForest alone with a warning, rather
than failing a whole training run.

**The LLM is peripheral and always optional.** One endpoint, one task, and a
deterministic template fallback so a text-generator outage degrades wording
rather than operations. `LLM_PROVIDER=stub` means the whole stack runs with no
credentials.

**The synthetic generator derives labels from a noisy function of the features**
rather than shifting features by label. That makes difficulty tunable
(`noise_scale`) and the relationship realistic — the first version produced
ROC-AUC 1.0, which is useless as a demonstration.

**`top_signals` is magnitude, not importance.** The maintenance note highlights
the largest-magnitude readings. That is not an explanation of the prediction;
SHAP would be. It is labelled as such in the code and listed under "Future work".

---

## Limitations

- **The committed metrics are from synthetic data.** Stated again because it is
  the most important caveat in this README. Nothing here measures real-world
  APS prediction accuracy.
- **Cost alone is not a sufficient acceptance criterion.** Below a positive rate
  of `10/510 ≈ 1.96%`, "flag nothing" is genuinely cheaper than "flag
  everything" — so a model must be shown to beat *both* trivial baselines, not
  just score a low cost. There is a test documenting this; the trainer does not
  currently enforce it.
- **Auth is demo-grade.** One operator credential pair from the environment, no
  user store, no refresh tokens, no scopes or revocation. JWT verification is
  real; identity is a placeholder.
- **The threshold is tuned on the same held-out split used to report metrics**,
  so the reported cost is mildly optimistic. A three-way split (train / tune /
  report) would be more honest at this data size.
- **No feature-importance explanation** per prediction (see `top_signals` above).
- **Batch scoring runs in-process** via FastAPI background tasks. A restart
  mid-job leaves the job stuck in `running`; there is no resume or reaper.
- **Drift is detected, not acted upon.** It is logged, flagged and stored; no
  retraining is triggered and no alert is sent.
- **No model-performance monitoring**, only input drift — because ground truth
  arrives weeks later, when the truck does or does not fail.
- **Single tenant**, no per-fleet isolation.
- **No rate limiting** or quotas.

---

## Future work

- **Trivial-baseline gate**: fail a training run whose cost does not beat both
  "flag everything" and "flag nothing".
- **Three-way split** so the tuned threshold is not evaluated on its own
  tuning data.
- **SHAP per prediction**, surfaced through the API and fed into the maintenance
  note so the LLM explains real drivers rather than large numbers.
- **Delayed-label evaluation**: ingest actual failures, join them back to
  predictions by `vehicle_id` and date, and measure realised cost per model
  version.
- **Drift → action**: alerting (PagerDuty/Slack) and an automatic retraining
  trigger behind an approval step.
- **Durable job queue** (arq/Celery) so batch jobs survive a restart, with retry
  and a stuck-job reaper.
- **Model promotion workflow**: train, compare against the live model on a fixed
  evaluation set, and promote via `POST /models/reload` only on improvement.
- **Real auth**: OAuth2/OIDC, per-fleet isolation, scoped tokens.
- **Observability**: OpenTelemetry spans for training and scoring, plus
  Prometheus metrics for prediction latency and risk-band distribution.

---

## License

MIT — see [LICENSE](LICENSE).
