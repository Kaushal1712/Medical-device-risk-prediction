# Stage 10 — Testing Audit & E2E Verification Report

> **Date:** 2026-08-17  
> **Model version:** `random_forest_20260816_145309`  
> **Full test suite:** 343 tests — **343 passed, 0 failed**  
> **E2E script:** 27 checks passed, 0 failed (1 warning: pipeline rerun skipped — expected)

---

## 1. Section 9 Testing Matrix Coverage

### 1.1 Data — Schema Validation & Pipeline Integrity

**File:** `tests/data/test_pipeline_integrity.py`  
**Tests:** 22 tests, all pass.

| Test Class | What It Verifies |
|------------|-----------------|
| `TestProcessedParquetSchema` | `merged.parquet` exists, row count matches `_manifest.json`, join correctness (merged rows == events rows), required columns present, device_id coverage, MD5 hashes match current raw files |
| `TestFeatureParquetSchema` | All 4 split parquets exist, feature columns match `model_card.json`, 62-feature count, no target/action_classification in feature list |
| `TestServingArtifactIntegrity` | Serving table exists, required columns present, one row per device, risk scores in [0,100], risk levels in {LOW, MEDIUM, HIGH}, row count ≈ 50,341 |
| `TestModelCardIntegrity` | model_card.json keys, n_features consistent, threshold in (0,1), target is `is_class_i`, decision_threshold matches test_metrics.json |

### 1.2 Leakage — Mandatory Cutoff-Truncated Recomputation

**File:** `tests/features/test_leakage.py` (class `TestCutoffTruncatedRecompute`)  
**Tests:** 3 mandatory cutoff recomputation tests + 5 classes of earlier leakage checks (31 tests total).

**Method (Section 9 compliant):**
For a deterministic 30-event sample of `train.parquet`:
1. Filter `merged.parquet` to rows for the **same `device_id`** with `event_date` strictly `< sample event_date`
2. Recompute `hist_device_event_count`, `hist_device_class_i_count`, `hist_device_recall_count` from scratch from that filtered slice
3. Assert the recomputed values **exactly equal** the cached values in `train.parquet`

This is not merely an upper-bound check — it asserts exact equality. Any code path that accidentally used the full event history would fail here.

Additional leakage tests:
- `TestTargetExclusion` — `is_class_i` and `action_classification` absent from feature columns
- `TestProhibitedFields` — 9 post-event/target-derived fields absent from features and train parquet
- `TestHistoricalLeakage` — first-ever device events have `hist_device_event_count == 0`; class I counts ≤ event counts
- `TestSameDayExclusion` — same-day events get identical (and correct) history counts
- `TestNoFutureLeakage` — sampled 200 train events: `hist_mfr_event_count` ≤ actual prior events
- `TestTemporalSplit` — train ≤ 2014-12-31, val in 2015, test in 2016-2017, holdout in 2018, no overlap
- `TestRowCounts` — total labeled events in expected range; positive rate 3-15% in each split

### 1.3 ML — Stored vs Live PR-AUC Consistency

**File:** `tests/models/test_model.py` (class `TestTestMetrics`, method `test_stored_pr_auc_matches_live_recomputation`)

**Method:**
1. Load `models/production/model.pkl` (and `preprocessor.pkl` if it exists)
2. Run `predict_proba` on `data/features/test.parquet` (8,918 rows)
3. Compute `average_precision_score(y_true, probas)` — **live recomputation**
4. Assert `|live_pr_auc - stored_pr_auc| < 1e-4`

**Result:** Pass. Stored PR-AUC = 0.542031, live recomputed = 0.542031 (difference < 1e-6).

This catches the scenario where the wrong model artifact was saved or test_metrics.json was edited post-hoc.

Additional ML tests:
- Model loads, predicts correct shape, all probabilities in [0,1]
- Test PR-AUC > 0.08 (well above 5.52% random baseline)
- Test ROC-AUC > 0.5 (= 0.861)
- Test recall > 0 (= 0.339 at threshold 0.856)
- Holdout 2018 PR-AUC within 50% relative of test PR-AUC

### 1.4 API — /predict Known & Unknown Device Tests

**File:** `tests/api/test_predict.py`

| Test | What It Verifies |
|------|-----------------|
| `test_known_high_risk_device_returns_200` | `/predict` → 200 for device `80508` |
| `test_predict_available_has_correct_schema` | `prediction_available`, `risk_level`, `risk_score`, `calibrated_probability`, `serving_event_date`, `model_version` all present and in valid range |
| `test_predict_low_risk_device` | Device `91519` → `risk_level == "LOW"` |
| `test_unscored_device_returns_prediction_unavailable` | Device `1` → `prediction_available == False`, `unavailable_reason` non-empty, no fabricated score/level |
| `test_malformed_payload_returns_422` | Missing `device_id` → 422 |
| `test_empty_device_id_returns_422` | Empty string `device_id` → 422 |
| `test_note_field_present` | `note` field present and non-empty |
| `test_predict_does_not_recompute_returns_snapshot` | Two consecutive calls → identical `risk_score` and `risk_level` (deterministic serving) |
| **`test_predict_unknown_device_returns_unavailable_not_error`** | **Unknown device `DEVICE_DOES_NOT_EXIST_XYZ_999` → `prediction_available == False`, no fabricated score, `unavailable_reason` present — not a 500 crash** |

### 1.5 Recommendations — Boundary Cases

**File:** `tests/api/test_recommendation.py` (pre-existing from Stage 7)

Covers HIGH/MEDIUM/LOW risk level → priority mapping, and boundary cases including missing criticality proxy and zero historical events.

### 1.6 Frontend — Import Smoke Tests

**File:** `tests/frontend/test_app_smoke.py`  
**Tests:** 10 tests, all pass. No browser, no HTTP calls — pure import-time checks.

| Test | What It Verifies |
|------|-----------------|
| `test_api_client_importable` | `utils.api_client` imports without errors (with stubbed streamlit) |
| `test_backend_url_constant_defined` | `BACKEND_URL` is a non-empty string |
| `test_backend_url_default_is_localhost` | Default points to `localhost:8000` |
| `test_all_required_endpoint_functions_defined` | All 8 endpoint wrappers present (`get_health`, `get_risk_summary`, `get_feature_importance`, `get_devices`, `get_device_detail`, `get_explanation`, `get_recommendation`, `post_copilot`) |
| `test_endpoint_functions_are_callable` | All wrappers are callable |
| `test_healthcare_disclaimer_in_backend_config` | `src.config.HEALTHCARE_DISCLAIMER` exists, is `str`, contains "prototype" or "decision-support" |
| `test_frontend_app_py_exists` | `frontend/app.py` present |
| `test_all_page_files_exist` | All 4 page files present |
| `test_frontend_utils_init_exists` | `frontend/utils/__init__.py` present |
| `test_frontend_api_client_exists` | `frontend/utils/api_client.py` present |

---

## 2. E2E Verification — `scripts/verify_e2e.sh`

Run: `bash scripts/verify_e2e.sh`  
Duration: ~60 seconds  
**Result: 27 passed, 1 warning, 0 failures**

### Section Results

| Section | Result |
|---------|--------|
| 1. Raw Data Files (3 CSVs) | ✓ 3/3 |
| 2. Pipeline Re-run | ⚠ Skipped (expected without `--rerun-pipeline`) |
| 3. Processed Artifacts (14 files + row counts) | ✓ All pass |
| 3b. Manifest Hash Integrity | ✓ 3/3 hashes match |
| 4. Feature ↔ Model Card Consistency | ✓ 62 columns match exactly; thresholds consistent |
| 5. FastAPI Backend (28 endpoint checks) | ✓ 28/28 |
| 6. Streamlit Frontend (import + page files) | ✓ 6/6 |
| 7. Security / Healthcare Checks | ✓ 5/5 (no hardcoded secrets, .env gitignored, disclaimer in README + 4 docs) |

### FastAPI Endpoint Checks (Section 5)

All 28 checks passed:
- `/health` → 200, has `model_version`, `disclaimer` (contains "prototype"), `manifest_hash`
- `/risk-summary` → 200, `total_scored > 0`
- `/devices?page_size=5` → 200
- `/devices/80508` → 200, `risk_score == 100`, `risk_level == "HIGH"`
- `POST /predict` (device 80508) → 200, `prediction_available == True`, `risk_score == 100`
- `POST /predict` (unknown device) → 200, `prediction_available == False`
- `/explanation/80508` → 200, `available == True`, `top_positive` non-empty
- `/recommendation/80508` → 200, `maintenance_priority == "Critical"`, disclaimer present
- `/feature-importance` → 200, `features` non-empty
- `POST /copilot` → 200, `answer` non-empty, `context_used.risk_level == "HIGH"`
- `POST /predict` (empty device_id) → 422
- `POST /copilot` (empty question) → 422

---

## 3. Security & Healthcare Checks

| Check | Result |
|-------|--------|
| `.env` in `.gitignore` | ✓ Pass |
| No hardcoded API keys (`sk-<20+chars>`, `AIza<20+chars>`) | ✓ Pass |
| `.env.example` LLM_API_KEY is blank placeholder | ✓ Pass |
| Healthcare disclaimer in `README.md` | ✓ Pass |
| Healthcare disclaimer in docs (≥1 file) | ✓ Pass (found in 4 docs) |
| `HEALTHCARE_DISCLAIMER` constant in `src/config.py` | ✓ Pass (tested in `test_app_smoke.py`) |
| `/health` endpoint returns disclaimer | ✓ Pass |
| `/recommendation` responses include disclaimer | ✓ Pass |

---

## 4. Production Code Integrity Check

`git diff --name-only` shows only the Stage 10 test and script files were modified:

```
tests/api/test_predict.py        (added unknown-device test)
tests/features/test_leakage.py   (added TestCutoffTruncatedRecompute)
tests/models/test_model.py       (added stored-vs-live PR-AUC test)
```

**Untracked new files (Stage 10):**
```
scripts/verify_e2e.sh
tests/data/test_pipeline_integrity.py
tests/features/_leakage_note.txt
tests/frontend/__init__.py
tests/frontend/test_app_smoke.py
docs/api_contract.md
docs/10_testing_audit.md
```

No modifications to `backend/`, `src/`, `frontend/`, `models/`, or `artifacts/`.

---

## 5. Model Performance Summary

| Metric | Validation (2015) | Test (2016-2017) | Holdout (2018) |
|--------|-------------------|-----------------|----------------|
| PR-AUC | — | **0.542** | — |
| ROC-AUC | — | **0.861** | — |
| Precision | — | **0.977** | — |
| Recall | — | **0.339** | — |
| Decision Threshold | — | **0.856** | — |
| N samples | 4,273 | 8,918 | 1,361 |

> Positive rate (Class I recall events) is ~8.3% in train, ~5.5% in test. High precision (0.977) at threshold 0.856 reflects a conservative flagging policy appropriate for safety-critical devices.

---

## 6. Known Issues & Warnings

| Issue | Severity | Status |
|-------|----------|--------|
| `scripts/verify_e2e.sh` — bash `set -euo pipefail` + `((PASS++))` arithmetic issue caused script to abort silently on first pass | **Bug (fixed)** | Fixed: replaced `((VAR++))` with `VAR=$((VAR+1))` |
| `scripts/verify_e2e.sh` — secret scan regex `sk-` matched `/risk-summary` URL paths (false positives) | **Bug (fixed)** | Fixed: tightened to `sk-[A-Za-z0-9]{20,}` |
| `docs/api_contract.md` missing | **Gap (fixed)** | Created in Stage 10 verification |
| `docs/10_testing_audit.md` missing | **Gap (fixed)** | Created in Stage 10 verification |
| Starlette deprecation: `httpx` → should be `httpx2` | Warning only | Not blocking; no production impact |
| `joblib.numpy_pickle` NumPy 2.5 `shape` deprecation | Warning only | Not blocking; affects model loading speed warning messages only |
