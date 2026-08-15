# MASTER PROMPT — Medical Device Failure Risk Prediction System
### (Paste this entire document into Antigravity, running the Opus model, as your task instructions)

---

## 0. ROLE AND MISSION

You are acting as a senior ML architect, data engineer, backend engineer, frontend engineer, and MLOps engineer building a complete, demoable, production-quality-*pattern* prototype for a Cognizant NPN hackathon.

**Problem statement:** Healthcare — Predicting Medical Equipment Failure. The goal is to use real device, event, and manufacturer data to identify which medical devices are at elevated risk of a future failure-related event, explain *why*, and translate that into a maintenance-prioritization recommendation — surfaced through an API, a dashboard, and a grounded GenAI copilot.

You will build this **incrementally, in the staged order specified in Section 5**, validating each stage before moving to the next. You will not skip ahead. You will not fabricate anything the data does not support.

This is a real, working system built on real CSV files, not a mockup. Every number shown anywhere in the final product (dashboard stats, model metrics, SHAP values, risk scores) must be computed from the actual data at `data/raw/`. Nothing is hardcoded, sampled-and-forgotten, or synthetically invented.

---

## 1. ABSOLUTE, NON-NEGOTIABLE CONSTRAINTS

1. **Real data only.** Three CSVs live at:
   ```
   data/raw/devices.csv        (~118,000 rows)
   data/raw/events.csv         (~125,000 rows)
   data/raw/manufacturers.csv  (~32,000 rows)
   ```
   These paths must be configurable (via `src/config.py` + `.env`), defaulting to the above. Never embed CSV content, row values, or column values into source code, prompts, or configuration. Never generate synthetic replacement data. If a file is missing, fail with a clear error — do not substitute fake data.

2. **No schema assumptions.** You do not know the exact column names, data types, or cardinalities of these files yet. **Do not guess or invent column names anywhere in this project.** Every downstream stage that references a column must first be justified by an inspection step that actually printed/profiled that column from the real file.

3. **No fabricated ML artifacts.** Model metrics, SHAP values, feature importances, thresholds, and risk scores must all be the genuine output of code run against the real data. If a metric can't be computed (e.g., insufficient positive examples for PR-AUC in a tiny slice), say so in logs/docs — do not invent a plausible-looking number.

4. **No target invented for convenience.** Do not default to `has_event = 1 / no_event = 0` as the prediction target without first proving (in Stage 3) that it represents a defensible, leakage-free, *future* failure-risk formulation. See Section 5, Stage 3 for the required decision process.

5. **Local-machine performance.** The three raw files total approximately 275,000 rows combined (devices.csv ≈ 118K, events.csv ≈ 125K, manufacturers.csv ≈ 32K). This is small enough for pandas but large enough that naive re-reads of CSV per script run, unvectorized `.apply()` row loops, or repeated joins will be noticeably slow and must be avoided. Use the caching/Parquet strategy in Section 4.

6. **No unnecessary infrastructure.** No Spark, Kafka, Kubernetes, message queues, or distributed compute. No microservice sprawl. One FastAPI backend, one frontend app, one training pipeline. Simplicity and explainability are scored higher than architectural sophistication in this hackathon.

7. **No secrets in code.** LLM API keys and provider config live only in `.env` (git-ignored), read via `os.environ` / `pydantic-settings`. Provide `.env.example` with empty/placeholder values.

8. **Healthcare framing discipline.** This is a decision-support prototype, not a certified medical device, not a regulatory tool, and not a guarantee of patient safety. This disclaimer must appear in the README, in the API's `/health` response metadata, and in the dashboard footer/about section. Never claim the system predicts an *exact* failure date unless the data genuinely supports time-to-event survival modeling — and even then, express it as an estimated distribution, not a certainty.

---

## 2. FIXED ARCHITECTURE (do not replace with a different shape)

```
                         RAW DATASETS
                              │
                              ▼
                 1. DATA INGESTION & VALIDATION
                              │
                              ▼
                    2. DATA ENGINEERING
                              │
                              ▼
                 3. TEMPORAL DATASET BUILDER
                              │
                              ▼
                    4. FEATURE ENGINEERING
                              │
                              ▼
                     5. ML RISK ENGINE
                              │
                              ▼
                    6. RISK SCORING ENGINE
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        7. EXPLAINABILITY          8. MAINTENANCE
             ENGINE               DECISION ENGINE
                 │                         │
                 └────────────┬────────────┘
                              ▼
                       9. FASTAPI BACKEND
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             10. WEB DASHBOARD     11. GENAI COPILOT
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       HOSPITAL USER
                              │
                              ▼
                    ACTUAL OUTCOME
                              │
                              ▼
                     FUTURE RETRAINING
```

Risk scoring and maintenance-priority are **separate concepts**: risk is a model-derived probability/level; maintenance priority is a rule-based combination of risk + device criticality + historical context. Do not collapse them into one number.

---

## 3. REPOSITORY STRUCTURE

```
medical-device-risk/
│
├── data/
│   ├── raw/                # devices.csv, events.csv, manufacturers.csv (git-ignored)
│   ├── processed/          # devices.parquet, events.parquet, manufacturers.parquet, merged.parquet
│   └── features/           # train.parquet, validation.parquet, test.parquet
│
├── notebooks/               # exploratory inspection only — logic must be mirrored into src/, notebooks are not the source of truth
│
├── src/
│   ├── config.py            # paths, env-driven settings, constants (no magic numbers scattered elsewhere)
│   ├── data/                # ingestion, validation, cleaning, joins, caching
│   ├── target/               # target discovery + construction + temporal cutoff logic
│   ├── features/             # feature engineering pipeline
│   ├── models/               # training, evaluation, model registry/versioning
│   ├── explainability/       # SHAP global + local explanation generation
│   ├── risk/                  # probability → risk score → LOW/MED/HIGH
│   └── recommendations/       # rule-based maintenance decision engine
│
├── models/
│   ├── production/           # risk_model.pkl, preprocessor.pkl, model_metadata.json
│   └── experiments/          # all trained candidates + their metrics, timestamped
│
├── artifacts/
│   ├── metrics/               # model comparison tables, confusion matrices (as data, e.g. json/csv)
│   ├── plots/                  # PR curves, ROC curves, calibration plots, SHAP summary plots (png)
│   └── explanations/           # cached per-device SHAP explanation payloads
│
├── backend/
│   ├── main.py
│   ├── schemas.py             # pydantic request/response models
│   ├── routes/
│   └── services/               # model-loading singleton, feature retrieval, copilot service
│
├── frontend/                   # React + TypeScript + Tailwind + Recharts app
│
├── tests/
│   ├── data/
│   ├── models/
│   ├── api/
│   └── recommendations/
│
├── docs/
│   ├── 01_dataset_inspection_report.md
│   ├── 02_target_definition_report.md
│   ├── 03_leakage_prevention.md
│   └── 04_model_comparison.md
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

You may refine folder names slightly if you have a compelling reason, but preserve the data / target / features / models / explainability / risk / recommendations / backend / frontend separation. Document any deviation in the README.

---

## 4. GLOBAL ENGINEERING PRINCIPLES (apply throughout)

- **Read raw CSVs exactly once per pipeline stage that needs them**, then persist to Parquet under `data/processed/`. All later stages read Parquet, never the original CSV.
- **Cache invalidation via manifest.** Maintain a small `data/processed/_manifest.json` recording a content hash (e.g., md5) of each raw CSV plus a pipeline-code version string. On pipeline run, recompute the hash; if unchanged and outputs exist, skip reprocessing (log this decision); if changed, reprocess and update the manifest. This avoids stale-cache bugs without needing a full workflow orchestrator.
- **Vectorize.** No per-row Python loops over 100K+ rows for feature computation. Use pandas/numpy vectorized operations, `groupby`, `merge_asof` (useful for as-of-cutoff historical aggregation), or Polars if you find it materially clearer/faster for a specific step — justify the choice inline in a comment if you introduce Polars alongside pandas.
- **Avoid redundant copies.** Prefer in-place-safe transformations and explicit `.copy()` only where mutation would otherwise corrupt a shared frame. Don't chain multiple full-dataframe copies for stylistic reasons.
- **Config over hardcoding.** Any path, threshold, random seed, model hyperparameter default, or environment-dependent value belongs in `src/config.py` or `.env`, not inline in business logic.
- **Reproducibility.** Fix random seeds for splits and model training. Every training run writes a metadata file capturing: git-independent run timestamp, data manifest hash, feature list, hyperparameters, and resulting metrics — so results are traceable, not just printed to stdout and lost.
- **Logging, not silent failure.** Each pipeline stage logs row counts in/out, dropped-row counts and reasons, and any schema anomalies. No `except: pass`.

---

## 5. STAGED EXECUTION PLAN (mandatory order, with STOP gates)

You must complete each stage, produce its stated deliverable, and — for the two gated stages — explicitly present findings before proceeding, rather than silently continuing on assumptions.

```
Stage 0
    ↓
Stage 1 — inspect real CSVs
    ↓
   STOP
    ↓
Stage 2
    ↓
Stage 3 — define/validate target
    ↓
   STOP
    ↓
Stage 4+ — features, ML, application
```

Antigravity must **not** train a model, construct final features, or make assumptions about the target before the required target-definition gate (the second STOP above) has been passed with all Stage 3 items validated against the real data.

### STAGE 0 — Bootstrap
Set up the repository structure from Section 3, `requirements.txt` (pandas, numpy, scikit-learn, xgboost, shap, fastapi, uvicorn, pydantic, pydantic-settings, pyarrow, python-dotenv, pytest, plus a frontend `package.json` with React/TypeScript/Tailwind/Recharts), `.gitignore` (exclude `data/raw/*.csv`, `data/processed/`, `data/features/`, `models/`, `.env`, `node_modules/`, `__pycache__/`), and `.env.example`. No business logic yet.

### STAGE 1 — Dataset Inspection & Data-Quality Report — **STOP GATE 1**
Before writing any transformation logic, write and run an inspection script (`src/data/inspect.py` or a notebook mirrored into `src/`) that, for each of the three CSVs:
- Prints shape, column names, dtypes.
- Profiles missingness per column (count + %).
- Profiles cardinality of categorical/ID-like columns.
- Detects duplicate rows and duplicate ID values.
- Identifies candidate primary keys (device identifier, manufacturer identifier, event identifier) by testing uniqueness.
- Identifies candidate foreign-key relationships between the three files (e.g., which column in `events.csv` matches which column in `devices.csv`; which column in `devices.csv` matches which column in `manufacturers.csv`) by measuring overlap of value sets — do not assume names like `device_id` exist; find whatever the real join keys are.
- Identifies all date/datetime-like columns in `events.csv` and `devices.csv`, parses them, and reports their min/max range and missingness. This is critical input to Stage 3.
- Identifies the column(s) that describe **event type / category / classification** (e.g., something resembling adverse event type, recall classification, complaint type, malfunction description) and lists the distinct values with counts. Do not assume a specific taxonomy exists — report what is actually present.
- Identifies any free-text columns (e.g., problem/event narrative descriptions) and reports typical length and non-null rate, for later NLP-optionality decisions.

**Deliverable:** `docs/01_dataset_inspection_report.md` summarizing all of the above with actual numbers, plus the discovered join-key mapping between the three tables (drawn exactly as `manufacturers → devices → events` or whatever the real relationship structure turns out to be).

**Do not proceed to Stage 2 until this report reflects the real files.** If the assumed `manufacturers → devices → events` relationship does not hold (e.g., events reference devices only indirectly, or there's a many-to-many pattern), document the actual relationship and adapt Stage 2 accordingly — do not force-fit the originally assumed hierarchy.

### STAGE 2 — Data Engineering Pipeline
Implement `src/data/pipeline.py` (invoked as `python -m src.data.pipeline`), which:
- Loads the three CSVs (chunked reading only if profiling in Stage 1 showed memory pressure — at ~125K rows this is unlikely to be necessary, but implement chunked reading as a configurable option if a file is later swapped for something larger).
- Applies cleaning found necessary in Stage 1: deduplication, missing-value handling per column (documented per-column strategy — do not blanket-fillna), date parsing/normalization to a consistent timezone-naive datetime, categorical value normalization (trimming, casing, collapsing obvious near-duplicates only where clearly justified).
- Validates the join keys discovered in Stage 1 (row counts before/after each join; log unmatched/orphan rows rather than silently dropping them without a count).
- Persists `devices.parquet`, `events.parquet`, `manufacturers.parquet`, and a joined `merged.parquet` to `data/processed/`, guarded by the manifest-based cache check from Section 4.
- Emits a short run summary (rows in, rows out, join match rates) to the log.

### STAGE 3 — Temporal Dataset Builder & Target Definition — **STOP GATE 2**
This is the most important and most error-prone stage. Work through it explicitly and do not shortcut it. **Before any feature engineering or model training code is written, you must explicitly determine and document each of the following from the real data — this stage is not complete until all six are answered with evidence, not assumption:**
1. The prediction unit.
2. The prediction cutoff definition.
3. The future outcome window.
4. What constitutes a failure-related event (with the mapping table from real category values).
5. The resulting class balance (positive/negative counts and rate).
6. A leakage analysis confirming no post-cutoff information is used anywhere in the target or planned features.

**Do not begin Stage 4 (feature engineering) or Stage 5 (model training) until all six items above are answered and internally consistent with the actual contents of `devices.csv`, `events.csv`, and `manufacturers.csv`.**

**3a. Determine the prediction unit and cutoff scheme from the real data**, based on what Stage 1 revealed about event dates and event-per-device frequency:
- Compute, from `merged.parquet`, the distribution of number of events per device. If most devices have multiple time-stamped events, you can build **rolling cutoff** examples: for a device with events at times `t1 < t2 < ... < tn`, choose a cutoff `tk`, compute features from events with date `< tk` only, and label the example positive if a failure-related event occurs in a defined future window `(tk, tk + W]` (or "any subsequent failure-related event" if a bounded window isn't well supported by the date range — document whichever choice you make and why).
- If most devices have only 0–1 events (verify this from Stage 1's actual counts before assuming it), a per-device rolling-cutoff formulation is not viable. In that case, reformulate the target at whatever granularity the data actually supports — for example, "will this device (using only its static device/manufacturer attributes and any single known historical event as of a global cutoff date) be associated with a failure-related event after that cutoff." State plainly in the report that a fine-grained recurrence model was not viable and why, and document the fallback formulation as a deliberate, data-driven pivot — not an accidental shortcut.
- Either way, define and write down explicitly: **prediction unit**, **cutoff definition**, **feature window (before cutoff)**, **outcome window (after cutoff)**, **target column definition**.

**3b. Determine what counts as a "failure-related" event** using the actual event-type/category values profiled in Stage 1 — build an explicit, documented mapping from real category values to a `is_failure_related` flag (e.g., values resembling malfunction, adverse event, injury, death, recall, corrective action would plausibly map to failure-related, while values resembling routine registration/listing updates would not — but you must confirm against the *actual* category values present, not assume FDA-style terminology if the file doesn't contain it). Include the mapping table itself in the report.

**3c. Verify the target is usable:**
- Confirm both classes have a non-trivial number of examples (report exact counts and the positive rate).
- Confirm the target is not trivially derivable from a feature you intend to keep (e.g., don't keep "has any failure-related event" as a feature if it's the target).
- Confirm no feature construction step used information dated at or after the cutoff.

**3d. If, after this analysis, no scientifically defensible future-failure target can be constructed** (e.g., dates are missing/unusable, or there is no way to separate historical from future information), **do not force one**. Instead select and clearly document the strongest defensible alternative that still serves the business problem — for example, a cross-sectional "historical risk profile" classification/ranking rather than a strict future-event prediction — and make this limitation explicit and prominent in the README (a section titled "What This Model Does and Does Not Predict").

**3e. Implement the time-aware split.** If cutoffs vary by date, split train/validation/test by cutoff date ranges (e.g., earliest 70% of cutoff timeline → train, next 15% → validation, most recent 15% → test) rather than randomly shuffling rows, so the model is evaluated on genuinely later information than it trained on. If the data only supports a single global cutoff, use a grouped split by device ID (with a fixed seed) instead, and document why a temporal split wasn't possible.

**3f. FINAL SERVING-POLICY REQUIREMENT — define the serving-snapshot policy.** Because a single device may have multiple valid temporal prediction snapshots (multiple cutoffs, per 3a), the pipeline must define, unambiguously, which snapshot is used whenever the application requests a device's *current* risk. For devices with multiple valid temporal prediction snapshots, the production/dashboard risk score must use the **latest valid prediction snapshot** whose cutoff date is actually supported by the available data (i.e., there is a genuinely computable feature window before that cutoff) and whose future outcome window is handled consistently with the target-definition methodology from 3a–3c — the selected serving cutoff must still respect the same outcome-window logic used to build the training examples, not an ad-hoc "most recent row" pick that breaks the target definition. The selected serving snapshot's cutoff date (and the snapshot/example identifier that produced it) must be stored alongside the prediction artifact so it is fully traceable, not re-derived arbitrarily at request time. If a device has no snapshot that qualifies as valid and scoreable under this policy (e.g., too little historical data before any usable cutoff), it has **no** production risk score; downstream consumers (API, dashboard) must surface this as an explicit "prediction unavailable" state rather than fabricating or defaulting to a score. This policy is binding on Stage 6 (risk scoring/serving artifact) and Stage 7 (API) below.

**Deliverable:** `docs/02_target_definition_report.md` covering: prediction unit, cutoff logic, failure-event mapping table, class balance, temporal coverage, split strategy, serving-snapshot policy, and explicit leakage-prevention statement. **Do not proceed to Stage 4 until this target is validated against the checks in 3c.**

### STAGE 4 — Feature Engineering
Implement `src/features/pipeline.py` (`python -m src.features.pipeline`), producing `data/features/{train,validation,test}.parquet`. Build only features computable strictly before each example's cutoff:

- **Device-level features**: whatever static attributes Stage 1 actually found (e.g., category/classification, regulatory attributes, geographic/country fields, listed dates) — use only columns confirmed to exist and be pre-cutoff-safe (e.g., a device's *registration* date is fine; a field only populated after an event occurred is not).
- **Historical event features (as-of-cutoff)**: count of prior events, count of prior failure-related events (per the 3b mapping), event frequency/recency (e.g., days since last event, days since first event), rolling counts over trailing windows if date density supports it.
- **Manufacturer-level features (as-of-cutoff)**: prior event count and failure-related event rate across the manufacturer's device population, historical device count, computed using only data dated before each example's cutoff (this requires care — recompute manufacturer aggregates per cutoff, not once globally, or you will leak future manufacturer behavior into earlier examples).
- **Failure-pattern categorical features**: if a structured category/description field exists, derive coarse categories (e.g., software-related, mechanical/component-related, manufacturing-related, design-related, packaging-related, maintenance-related, other) **only from values actually observed** in that column — do not invent categories that aren't represented.
- **Text/NLP (optional, secondary)**: only if Stage 1 showed a usable free-text field with reasonable coverage. Start with TF-IDF over a modest vocabulary size or simple keyword-flag features. Do not build embeddings/transformer pipelines unless a quick TF-IDF baseline demonstrably improves validation performance — if you try this, report the before/after comparison.
- **High-cardinality categoricals** (e.g., manufacturer name, device model/product code): use frequency encoding or top-N-category + "Other" bucketing for tree models; avoid uncontrolled one-hot explosion. If you use target-mean encoding anywhere, compute it strictly from the training fold only and apply it to validation/test without leakage.

**Deliverable:** run an automated leakage check (see Stage 10) immediately after building this dataset, before touching models.

### STAGE 5 — ML Risk Engine
Implement `src/models/train.py` (`python -m src.models.train`) and `src/models/evaluate.py` (`python -m src.models.evaluate`).

- Preprocessing: a `sklearn.Pipeline`/`ColumnTransformer` (persisted as `preprocessor.pkl`) handling missing values, numeric scaling where the model needs it (not needed for tree models, needed for logistic regression), and categorical encoding per the strategy above.
- **First measure the actual class imbalance ratio** from the training split and record it. Based on that number, choose `class_weight='balanced'` (LR, RF) / `scale_pos_weight` (XGBoost) as the default approach; only evaluate SMOTE/oversampling as an explicit additional experiment if the imbalance is severe, and compare results rather than assuming it helps.
- Train, in order, and keep all of them as artifacts under `models/experiments/`:
  1. A trivial baseline (majority-class / prior-rate predictor) — this anchors what "better than nothing" means.
  2. Logistic Regression (with regularization, class weighting).
  3. Random Forest.
  4. XGBoost (or another gradient boosting library if you have a specific justified reason).
- Evaluate all candidates on the **validation** split using: Precision, Recall, F1, ROC-AUC, PR-AUC, and a confusion matrix at a documented threshold. Given likely class imbalance, weight PR-AUC and recall/precision trade-offs in the write-up more heavily than raw accuracy.
- Produce `artifacts/metrics/model_comparison.csv` (or `.json`) and `docs/04_model_comparison.md` with the **real** numbers from these runs, plus PR/ROC/calibration plots in `artifacts/plots/`.
- **Select the final model based on validation performance, generalization behavior between train and validation, interpretability trade-offs, and computational cost — not simply "whichever number is highest."** Justify the choice in writing in the model comparison doc.
- Confirm final generalization by evaluating the selected model once on the held-out **test** split, reported separately and not used for any tuning decisions.
- Save the selected model + preprocessor + `model_metadata.json` (model type, version/timestamp, data manifest hash, full feature list, target definition summary, validation and test metrics, chosen operating threshold(s)) to `models/production/`.

### STAGE 6 — Risk Scoring, Explainability, Maintenance Decision Engine
- **Risk Scoring (`src/risk/`)**: convert model probability into a risk score.
  - **Calibration (if used) must be leakage-safe, using the calibration API appropriate to the installed scikit-learn version.** If the base model's predicted probabilities are poorly calibrated (check via a reliability curve / Brier score on validation), apply probability calibration (e.g., `CalibratedClassifierCV`). Before implementing this, check the installed scikit-learn version and use whichever calibration mechanism is correct and non-deprecated for that version (for example, some versions accept `cv="prefit"` on an already-fitted estimator, while others require a different mechanism, such as wrapping the fitted estimator — e.g., via `FrozenEstimator` — or fitting `CalibratedClassifierCV` internally with a proper time-respecting `cv` split). Do not blindly hardcode one specific syntax without checking; verify what the installed version actually supports and implement accordingly, and note the version and approach used in the docs. Regardless of which API is used, the calibration procedure must respect the temporal/cutoff structure established in Stage 3: fit the calibrator only on training-period examples (e.g., a held-out slice within the training window, respecting time ordering), and it must **never** use validation or test examples — or any example whose cutoff falls after the data used to fit the calibrator — for fitting the calibrator. Calibration must not introduce any future-data leakage beyond what the base model training already guards against. Document the before/after calibration comparison (reliability curve, Brier score); if calibration isn't needed, state that explicitly and use raw probabilities.
  - **Threshold selection must be evidence-based, not arbitrary.** Convert probability → LOW/MEDIUM/HIGH using operating threshold(s) derived from validation-set precision/recall behavior — **do not** pick fixed values such as 0.3/0.7 without justification, and **do not** default to unexamined percentile cutoffs either. Concretely: plot precision and recall against threshold on the validation set, and choose the threshold(s) by explicitly reasoning through the business trade-off between missing a genuinely high-risk device (a false negative — a safety-relevant miss) and over-flagging a low-risk device (a false positive — wasted inspection effort/alarm fatigue). State plainly which side of that trade-off the project prioritizes and why (e.g., "recall is weighted more heavily at the HIGH boundary because a missed high-risk device is more costly than an unnecessary inspection"). Report the exact resulting numeric threshold(s), the precision/recall achieved at each, and this full reasoning in `model_metadata.json` and the docs.
  - **Materialize production scores per the Stage 3f serving-snapshot policy.** When scoring the feature dataset to populate the values served by the API/dashboard, select each device's latest valid snapshot per Stage 3f (never simply the last row in the feature table) and persist a dedicated serving table (e.g., `artifacts/risk/device_risk_snapshot.parquet`) containing at minimum: device_id, selected cutoff date, snapshot/example identifier, risk probability, calibrated score, risk level, and model version. This table — not an ad-hoc re-query of the raw feature dataset — is the single source of truth the backend reads from. Devices with no valid snapshot are simply absent from this table, and the backend must treat that as "prediction unavailable," never as an error to silently paper over.
- **Explainability (`src/explainability/`)**: use `shap.TreeExplainer` if the selected model is tree-based (fast, exact) or the model's native coefficients / `shap.LinearExplainer` if it's logistic regression — do not default to slow `KernelExplainer` unless the selected model type requires it. Produce (a) a **global** explanation: overall feature importance / SHAP summary plot saved to `artifacts/plots/`, and (b) a **local** per-device explanation function that returns top positive and negative contributing features for a single prediction. Cache computed local explanations for served devices in `artifacts/explanations/` keyed by device ID + model version, to avoid recomputing SHAP on every API call.
- **Maintenance Decision Engine (`src/recommendations/`)**: purely rule-based, no additional ML model. Combine risk level + any real device-criticality signal found in the data (if the dataset has no explicit criticality field, state this and use risk level plus recency/severity of historical failure-related events as the closest available proxy — do not invent a criticality score not grounded in the data) + relevant historical event context into a `Critical / High / Medium / Low` maintenance priority and a small set of recommended actions (e.g., prioritize preventive inspection, schedule inspection, continue monitoring, review historical safety information). Document the exact rule table in `docs/`. Label all outputs clearly as decision-support suggestions, not medical or regulatory directives.

### STAGE 7 — FastAPI Backend
Implement under `backend/`, loading the model/preprocessor/explainer/feature artifacts **once at startup** (module-level singleton or FastAPI lifespan/dependency), never per-request. Endpoints (exact schemas defined in Section 6):
```
GET  /health
GET  /devices
GET  /devices/{id}
POST /predict
GET  /risk-summary
GET  /explanation/{id}
GET  /recommendation/{id}
POST /copilot
```
Validate all inputs with pydantic models in `backend/schemas.py`. Return structured error responses (404 for unknown device IDs, 422 for bad payloads) rather than uncaught exceptions. `GET /health` must include model version, data manifest hash, and the healthcare-prototype disclaimer text.

### STAGE 8 — Frontend Dashboard
React + TypeScript + Tailwind CSS + Recharts, consuming only the real FastAPI endpoints (no mock data files in the frontend). If you determine partway through that a simpler stack (e.g., server-rendered templates) would be materially faster to deliver a polished result within hackathon time, you may propose that trade-off explicitly in `docs/` before switching — otherwise build the React app as specified. Sections (detailed in Section 7): Overview, Device Search, Device Details, Explainability view.

### STAGE 9 — GenAI Copilot
Implement per the grounding contract in Section 8. Configurable provider via `.env` (`LLM_PROVIDER`, API key, `MODEL_NAME`). Must degrade gracefully (a deterministic templated response built from the same structured context, with no external call) if no API key is configured or the call fails — the demo must never break because of a missing LLM key.

### STAGE 10 — Testing, Leakage Audit, Integration, Documentation
Implement the full test matrix (Section 9) including an explicit automated leakage test, run the full pipeline end-to-end from raw CSVs through a live API response and a working dashboard, and finalize all documentation (Section 10).

---

## 6. API CONTRACT

- `GET /health` → `{ status, model_version, data_manifest_hash, trained_at, disclaimer }`
- `GET /devices?risk_level=&manufacturer=&category=&country=&search=&page=&page_size=` → paginated list of devices with id, key display attributes, and current risk level (precomputed/cached, not recomputed per list request).
- `GET /devices/{id}` → full device record (from `merged.parquet`/processed store) + the device's designated latest valid serving-snapshot risk score/level (per Stage 3f) — or an explicit "prediction unavailable" indicator if no valid snapshot exists — + maintenance priority summary.
- `POST /predict` → body contains a `device_id` only. The API retrieves that device's **designated latest valid serving snapshot** (per the Stage 3f serving-snapshot policy) from the persisted serving table — it must not arbitrarily select a row or recompute an ad-hoc feature vector at request time. Response includes probability, calibrated risk score, risk level, the serving snapshot's cutoff date, and model version. If the device has no valid scoreable snapshot, return a clear "prediction unavailable" response (not a fabricated score), with a reason. **Do not implement arbitrary raw-feature / "what-if" prediction in this version** — the endpoint must not accept an ad-hoc feature payload. Note this restriction and the possibility of a future "what-if" enhancement in `docs/`, but do not build it now.
- `GET /risk-summary` → aggregate counts (total devices, count/percent per risk level) computed from the real scored population **as defined by the Stage 3f serving-snapshot policy** (i.e., devices that have a valid designated serving snapshot), plus basic distribution stats (e.g., by category/manufacturer) for dashboard charts — no fabricated aggregates. Devices with no valid scoreable snapshot must be reported as a distinct "unscored / prediction unavailable" count rather than silently excluded without disclosure.
- `GET /explanation/{id}` → risk score, risk level, top positive contributing features, top negative contributing features, with real SHAP values (or a clear "insufficient data" response if a device wasn't scoreable).
- `GET /recommendation/{id}` → maintenance priority + recommended action(s) + the rule(s)/inputs that produced it, plus the decision-support disclaimer.
- `POST /copilot` → body includes device ID + user question; backend assembles the trusted structured context (Section 8), calls the LLM (or falls back), and returns a grounded natural-language answer plus the structured context it was grounded in (for UI transparency/debugging).

Document this contract precisely (request/response JSON shapes) either via FastAPI's auto-generated OpenAPI docs plus a short `docs/` note, or a dedicated `docs/api_contract.md`.

---

## 7. FRONTEND PAGE SPECS

- **Overview**: total devices, high/medium/low risk counts and percentages, risk distribution chart, event statistics, and a category/manufacturer breakdown — all sourced from `/risk-summary`.
- **Device Search**: filter by device ID, manufacturer, category, country, risk level; server-side filtering via `/devices` query params; paginated table.
- **Device Details**: device information, risk score + level, top risk factors (from `/explanation/{id}`), historical event summary (counts/timeline, not raw dumped rows), maintenance priority + recommended action (from `/recommendation/{id}`), and an embedded copilot Q&A panel for that device.
- **Explainability view**: global feature-importance/SHAP summary chart (dataset-wide), plus the per-device local contribution chart reused from Device Details.

Design guidance: clean cards, clear color-coded risk badges (e.g., red/amber/green), responsive layout, consistent typography, no default browser styling, no filler animations, no placeholder/fake numbers anywhere — if a metric can't be computed for a device, show an explicit "not available" state rather than a fabricated value.

---

## 8. GENAI GROUNDING CONTRACT

The LLM is a **natural-language explainer of trusted structured context**, never the source of predictions or facts. Implementation shape:
```
User question → FastAPI /copilot
             → retrieve real structured context for the device
                 (device info, real historical event summary,
                  real risk score/level, real top SHAP factors,
                  real maintenance priority/recommendation)
             → build a context block containing ONLY those real values
             → system prompt instructs the LLM to:
                 - answer only using the provided context
                 - explicitly separate "observed historical facts"
                   from "model prediction" from "decision-support
                   recommendation" in its answer
                 - say "not available in the data" when asked about
                   something outside the provided context
                 - never state a prediction as a confirmed fact
                 - never invent event history, dates, or maintenance
                   records
             → LLM response returned to the user
```
Keep this a single-context-block call, not a multi-hop RAG system. If the LLM call fails or no key is configured, fall back to a deterministic template that assembles the same structured context into a readable paragraph without calling any external API.

---

## 9. TESTING MATRIX

- **Data**: schema validation on processed Parquet, join correctness (row/match counts match Stage 1/2 findings), missing-value handling behaves as documented, target construction produces the expected class balance from Stage 3's report.
- **Leakage (mandatory, explicit test)**: for a sample of examples, assert that every feature value could only have been computed from events dated strictly before that example's cutoff (e.g., recompute a feature using a "future-truncated" copy of `events.parquet` filtered to `< cutoff` and assert the pipeline's cached feature matches — catching any code path that accidentally used the full history).
- **ML**: preprocessing pipeline round-trips without error on validation data, prediction output shape/probability range is correct (`[0,1]`, sums appropriately for the chosen formulation), saved model loads and reproduces the same metrics recorded in `model_metadata.json`.
- **API**: `/health` returns 200, `/predict` returns a valid schema for a known device and a 404/422 for invalid input, `/devices/{id}` lookup, `/explanation/{id}` returns SHAP-backed content for a scoreable device.
- **Recommendations**: risk-level → priority mapping matches the documented rule table for boundary cases (e.g., exactly at a threshold), and edge cases (missing criticality proxy, zero historical events) don't crash the engine.

All tests runnable via `pytest`, and should fail loudly with a clear message rather than passing silently on a skipped assertion.

---

## 10. SECURITY & HEALTHCARE DISCLAIMER REQUIREMENTS

- Prominent statement (README, `/health`, dashboard footer): *"This system is a decision-support prototype and does not replace qualified maintenance, biomedical engineering, regulatory, or clinical judgment. It is not a certified medical device and does not guarantee patient safety outcomes."*
- No claims of regulatory certification or clearance.
- No claims of exact future failure dates unless a genuine time-to-event/survival formulation was built and validated — if so, express results as an estimated distribution/range, not a guaranteed date.
- API input validation on every endpoint; no secrets committed; `.env` git-ignored; LLM provider/key fully configurable and never hardcoded.
- GenAI outputs must remain grounded per Section 8 at all times.

---

## 11. DOCUMENTATION DELIVERABLES

`README.md` must include, at minimum:
1. Project overview and business framing.
2. Full architecture diagram (Section 2) and repository layout.
3. **"What This Model Predicts and Why"** — the exact target definition, how it was derived from the real data, and explicitly what it does *not* claim (per Stage 3 findings).
4. **"Data Leakage Prevention"** section — cutoff logic, which columns were excluded and why, how the temporal/grouped split works, and a summary of the leakage test results.
5. Model comparison table with real metrics and the rationale for the final model choice.
6. Risk scoring methodology and the documented threshold values.
7. Maintenance decision rule table.
8. API reference (or link to the OpenAPI docs) and setup/run instructions for backend and frontend.
9. GenAI copilot design and grounding guarantees, including fallback behavior.
10. Healthcare/regulatory disclaimer (Section 10).
11. Known limitations and what a production version would need beyond this hackathon prototype.

---

## 12. EXECUTION COMMANDS

```bash
# Data preparation
python -m src.data.pipeline

# Feature generation
python -m src.features.pipeline

# Training
python -m src.models.train

# Evaluation
python -m src.models.evaluate

# API
uvicorn backend.main:app --reload

# Frontend
cd frontend && npm install && npm run dev

# Tests
pytest
```

---

## 13. ABSOLUTE DO'S AND DON'TS — FINAL CHECKLIST

**Never:**
- Invent dataset columns, categories, or values not actually present in the CSVs.
- Invent or guess a target label without completing the Stage 3 verification.
- Fabricate model metrics, SHAP values, predictions, or dashboard statistics.
- Hardcode risk scores or summary numbers instead of computing them.
- Use any information dated at or after an example's prediction cutoff as a feature for that example.
- Claim exact failure dates without a validated survival/time-to-event analysis behind the claim.
- Put API keys or secrets in source code.
- Silently swallow errors or skip a failed validation check.
- Replace real data with synthetic/fake data at any layer, including the frontend.

**Always:**
- Inspect the real data before writing logic that depends on its shape.
- Document every non-obvious assumption in `docs/`.
- Validate schemas and log pipeline row counts at each stage.
- Save versioned model artifacts with full metadata.
- Run and report the leakage test before considering the ML stage complete.
- Keep data, ML, backend, frontend, and GenAI concerns in clearly separated modules.
- Stop and report findings at the two gated stages (dataset inspection, target definition) before proceeding.

---

**Begin with Stage 0 and Stage 1. Do not train any model or write any feature logic until Stage 1's dataset inspection report and Stage 3's target definition report are complete and internally consistent with the actual contents of `devices.csv`, `events.csv`, and `manufacturers.csv`.**
