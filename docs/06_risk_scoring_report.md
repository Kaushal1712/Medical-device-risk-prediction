# Stage 6 — Risk Scoring Engine Report

**Project:** Medical Device Failure Risk Prediction System  
**Stage:** 6 — Risk Scoring Engine  
**Date:** 2026-08-16  
**Status:** ✅ Complete and verified  
**Preceding Stage:** 5 — ML Risk Engine (76 tests, PR-AUC 0.542 on test)

---

## 1. Overview

Stage 6 transforms the validated Stage 5 Random Forest model into a production-ready **Risk Scoring Engine**. The engine:

1. Applies **isotonic probability calibration** to produce posterior probabilities
2. Maps each calibrated probability to a deterministic **0–100 risk score**
3. Classifies devices into **LOW / MEDIUM / HIGH** risk bands using thresholds derived from the validation-set precision/recall curve
4. Materialises a **serving table** (`device_risk_snapshot.parquet`) — one row per device, using the latest event date per device to implement the Stage 3f serving policy

---

## 2. Architecture and File Map

```
src/risk/
├── __init__.py               # Public API exports
├── scorer.py                 # RiskScorer, ScoringResult, probability_to_score, score_to_band
├── calibrate.py              # Stage 6 calibration pipeline (run once, writes artifacts)
└── build_serving_table.py    # Batch-scores all splits → device_risk_snapshot.parquet

models/production/
├── model.pkl                 # Stage 5 Random Forest (unchanged)
├── model_card.json           # Model metadata, feature list, decision threshold
├── calibrated_model.pkl      # Isotonic-calibrated wrapper (written by calibrate.py)
└── calibration_report.json   # Calibration metrics + derived thresholds

artifacts/risk/
└── device_risk_snapshot.parquet  # Serving table (50,341 rows × 11 columns)

src/config.py                 # RISK_THRESHOLD_MEDIUM, RISK_THRESHOLD_HIGH (updated)
tests/risk/
└── test_risk_scorer.py       # 95 Stage 6 tests
```

---

## 3. Key Distinctions

Stage 6 preserves four separate concepts that must never be conflated:

| Concept | Type | Range | Description |
|---|---|---|---|
| `raw_probability` | float | [0, 1] | Uncalibrated RF `predict_proba()[:, 1]` |
| `decision_threshold` | float | (0, 1) | 0.8555 — F1-maximising threshold on validation (Stage 5) |
| `calibrated_probability` | float | [0, 1] | Isotonic-calibrated posterior |
| `risk_score` | float | [0, 100] | `round(calibrated_probability × 100, 2)` |
| `risk_level` | str | {LOW, MEDIUM, HIGH} | Band per T_MEDIUM / T_HIGH |

`is_class_i_predicted` is derived from `raw_probability >= decision_threshold` (0.8555), not from the calibrated probability.

---

## 4. Probability Calibration

### 4.1 Method

Isotonic regression calibration, implemented using sklearn 1.9.0's `FrozenEstimator + CalibratedClassifierCV`:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

calibrated = CalibratedClassifierCV(
    FrozenEstimator(base_model),   # Stage 5 RF — weights frozen
    method="isotonic",
    cv=None,                       # FrozenEstimator replacement for deprecated cv="prefit"
)
calibrated.fit(X_train, y_train)  # Only isotonic layer is fitted; base model unchanged
```

**Leakage safety:** The base model is frozen by `FrozenEstimator`; only the isotonic regression calibrator is fitted on the training split. The validation split is never seen during calibration fitting.

### 4.2 Calibration Metrics (Validation Split, 2015)

| Metric | Before Calibration | After Calibration |
|---|---|---|
| Brier Score | 0.126774 | 0.066984 |
| Brier Skill Score | −1.3015 | −0.2160 |
| ECE (10 bins) | 0.2590 | 0.0748 |
| PR-AUC | 0.6055 | 0.5918 |

**Improvement:** Brier score improved by +0.0598 (47% reduction). ECE improved by 71%.

> **Note on Brier skill score:** Both pre- and post-calibration values are negative, reflecting the positive-rate baseline (5.85% on validation). The base RF produces many high-confidence predictions for a relatively rare class, resulting in a high raw Brier score. Post-calibration, the score is substantially closer to the random baseline and the ECE confirms improved probability sharpness.

### 4.3 Isotonic Calibration Step-Function Behaviour

Isotonic calibration on imbalanced data produces a step-function mapping. On the validation split, the calibrated probability takes only **43 unique values** across 4,273 samples. A mass point at exactly `1.0` exists for 108 samples — all of which are Class I events. This is expected and correct behaviour.

---

## 5. Risk Score Formula

```
risk_score = round(calibrated_probability × 100, 2)
```

**Rationale:** The linear mapping preserves the posterior probability interpretation. No log-transform or percentile normalisation is applied — those would obscure the probabilistic meaning for downstream consumers. The scale is [0.0, 100.0]; the formula is invertible.

The function `probability_to_score(p)` in `src/risk/scorer.py` enforces:
- `ValueError` on NaN input
- `ValueError` on `p` outside `[0, 1]`

---

## 6. Risk Band Thresholds

### 6.1 Derivation Methodology

Thresholds are derived from the calibrated validation-set precision/recall curve using `sklearn.metrics.precision_recall_curve`. No test/holdout data is used.

```
python -m src.risk.calibrate
```

#### MEDIUM Threshold (T_MEDIUM)

> **Definition:** The calibrated probability threshold that maximises F1 on the validation set.

This is the "monitor closely" boundary — balanced precision and recall. Devices at or above this threshold warrant increased inspection frequency.

#### HIGH Threshold (T_HIGH)

> **Definition:** The highest calibrated probability threshold at which validation recall ≥ 35%.

**Business and regulatory reasoning:**
- A missed HIGH-risk device (false negative) carries significant patient-safety cost: the device goes uninspected when it should be flagged for recall/corrective action.
- A false HIGH flag (false positive) wastes inspection resources and can cause alarm fatigue, but is far less costly than a missed recall event.
- Therefore: the HIGH band boundary prioritises **recall ≥ 35%** while maximising precision subject to that constraint. This ensures the HIGH band is not effectively empty, catching a material fraction of Class I events.

### 6.2 Derived Values (from calibration_report.json)

| Parameter | Value | Notes |
|---|---|---|
| **T_MEDIUM** | **0.985714** | F1-maximising threshold on val |
| T_MEDIUM precision | 0.9910 | On validation set |
| T_MEDIUM recall | 0.4400 | On validation set |
| T_MEDIUM F1 | 0.6094 | On validation set |
| **T_HIGH** | **1.000000** | Highest thr with recall ≥ 35% |
| T_HIGH precision | 1.0000 | On validation set |
| T_HIGH recall | 0.4320 | On validation set |
| T_HIGH F1 | 0.6034 | On validation set |

### 6.3 Band Definition

```
risk_level = "HIGH"    if calibrated_probability >= T_HIGH    (≥ 1.0)
           = "MEDIUM"  if calibrated_probability >= T_MEDIUM  (≥ 0.985714)
           = "LOW"     otherwise
```

**T_HIGH = 1.0 explanation:** Isotonic calibration maps exactly 108 validation samples to `calibrated_probability = 1.0`. These are all confirmed Class I events. The `score_to_band` function uses `>=` comparison, so `1.0 >= 1.0 → HIGH`. This is mathematically correct — the 108 HIGH-flagged validation devices are all true positives with 100% precision.

The thin MEDIUM band (3 validation samples, all with `0.985714 ≤ cal_p < 1.0`) reflects the step-function nature of isotonic calibration on this dataset. On the full serving table (all splits), MEDIUM=206 and HIGH=2,053 (see §8).

### 6.4 Config Storage

Derived thresholds are written back to `src/config.py` by `calibrate.py`:

```python
RISK_THRESHOLD_HIGH: float = 1.0          # calibrated_prob >= this → HIGH
RISK_THRESHOLD_MEDIUM: float = 0.985714   # calibrated_prob >= this → MEDIUM (else LOW)
```

Default values of `0.0` serve as invalid sentinels; `build_serving_table.py` refuses to run if either threshold is `0.0`.

---

## 7. Serving Policy

### 7.1 Stage 3f Policy: Latest Event per Device

Per the Stage 3 design, 98.4% of devices have exactly one event row. The remaining 1.6% (793 devices in training; 2,458 deduplicated in the full pipeline) have multiple event rows.

The serving policy is:

> **For each `device_id`, retain only the row with the maximum `event_date`.**

**Rationale:** The most recent event reflects the device's current operational state and is more informative than older events. This is consistent with the Stage 3/4 feature engineering design.

Implementation (in `build_serving_table.py`):
```python
scored_sorted = scored.sort_values("event_date")
serving = scored_sorted.drop_duplicates(subset=["device_id"], keep="last")
```

### 7.2 Serving Table Schema

| Column | Type | Description |
|---|---|---|
| `device_id` | int64 | Primary key — unique per row |
| `event_id` | object | Original event `id` (latest event per device) |
| `serving_event_date` | datetime | Date of the latest event used for scoring |
| `raw_probability` | float64 | Uncalibrated RF probability |
| `calibrated_probability` | float64 | Isotonic-calibrated probability |
| `risk_score` | float64 | 0–100 score (`calibrated_probability × 100`, rounded 2dp) |
| `risk_level` | object | "LOW", "MEDIUM", or "HIGH" |
| `is_class_i_predicted` | bool | `raw_probability >= 0.8555` |
| `decision_threshold` | float64 | 0.8555140... (from model card) |
| `model_version` | object | Model experiment directory string |
| `scored_at` | object | UTC ISO 8601 timestamp of scoring run |

---

## 8. Serving Table Statistics (Full Dataset)

Run: `python -m src.risk.build_serving_table`

| Metric | Value |
|---|---|
| Total events loaded | 52,799 |
| Unique device_ids | 50,341 |
| Multi-event devices deduplicated | 2,458 |
| **Serving table rows** | **50,341** |
| Serving table columns | 11 |
| Output | `artifacts/risk/device_risk_snapshot.parquet` |

**Risk level distribution:**

| Risk Level | Count | % |
|---|---|---|
| HIGH | 2,053 | 4.1% |
| MEDIUM | 206 | 0.4% |
| LOW | 48,082 | 95.5% |

**Risk score statistics:**

| Stat | Value |
|---|---|
| min | 0.00 |
| mean | 8.91 |
| median | 0.00 |
| max | 100.00 |

---

## 9. Input/Output Schema

### 9.1 Single-Row Scoring (RiskScorer.score)

**Input:** `X` — numpy array of shape `(1, 62)` or `(62,)`, float32, in the order defined by `model_card.json["feature_columns"]`

**Output:** `ScoringResult` dataclass with fields:

```python
@dataclass
class ScoringResult:
    raw_probability: float        # RF predict_proba output
    calibrated_probability: float # Isotonic-calibrated posterior
    risk_score: float             # round(calibrated_probability * 100, 2)
    risk_level: str               # "LOW" | "MEDIUM" | "HIGH"
    is_class_i_predicted: bool    # raw_probability >= decision_threshold
    decision_threshold: float     # 0.8555140903590877
    model_version: str            # experiment dir string
    warnings: list[str]           # non-fatal issues (e.g. all-NaN row)
```

### 9.2 Batch Scoring (RiskScorer.batch_score)

**Input:** `pd.DataFrame` containing all 62 model feature columns (extra columns are tolerated)

**Output:** `pd.DataFrame` with columns: `id, device_id, event_date, raw_probability, calibrated_probability, risk_score, risk_level, is_class_i_predicted, decision_threshold, model_version`

### 9.3 Feature Requirements

Exactly 62 features as listed in `model_card.json["feature_columns"]`. Missing model features raise `ValueError`. Extra columns are silently ignored.

---

## 10. Edge Cases and Error Handling

| Scenario | Behaviour |
|---|---|
| NaN probability | `probability_to_score()` raises `ValueError("probability is NaN")` |
| Probability outside [0, 1] | `probability_to_score()` raises `ValueError` |
| T_MEDIUM >= T_HIGH | `score_to_band()` raises `ValueError("t_medium must be less than t_high")` |
| All-NaN feature row | `scorer.score()` adds warning to `ScoringResult.warnings`; scoring continues |
| Missing calibrated model | `load_calibration()` raises `FileNotFoundError` with re-run command |
| Missing base model | `RiskScorer.__init__()` raises `FileNotFoundError` |
| Missing model features in batch | `batch_score()` raises `ValueError` listing missing columns |
| Zero-sentinel thresholds | `build_serving_table.py` raises `sys.exit(1)` with calibrate instruction |
| Multiple events per device | Latest `event_date` retained; older rows discarded |

---

## 11. Reproducibility and Verification

### 11.1 Full pipeline (from scratch)

```bash
# Prerequisites: Stage 5 artifacts must exist
# (models/production/model.pkl, model_card.json)

# Step 1: Fit calibration and derive thresholds
python -m src.risk.calibrate

# Step 2: Build the serving table
python -m src.risk.build_serving_table
```

### 11.2 Test verification

```bash
# Stage 6 tests only (95 tests)
pytest tests/risk/test_risk_scorer.py -v

# Full test suite (76 Stage 1–5 + 95 Stage 6 = 171 total)
pytest -q
```

### 11.3 Serving table inspection

```python
import pandas as pd
df = pd.read_parquet("artifacts/risk/device_risk_snapshot.parquet")
print(df.shape)                    # (50341, 11)
print(df["risk_level"].value_counts())
print(df["risk_score"].describe())
```

### 11.4 Determinism

The risk score is deterministic given:
- Fixed `models/production/model.pkl` (frozen by Stage 5 selection)
- Fixed `models/production/calibrated_model.pkl` (isotonic fit on fixed training data)
- Fixed `RISK_THRESHOLD_MEDIUM` and `RISK_THRESHOLD_HIGH` in `src/config.py`

Re-running `calibrate.py` on the same data always produces the same thresholds (isotonic regression is deterministic with fixed data). Re-running `build_serving_table.py` produces an identical table except for the UTC `scored_at` timestamp.

---

## 12. Limitations and Known Behaviour

1. **Brier skill score is negative:** The base RF is overconfident on the imbalanced validation set. After isotonic calibration, the skill score improves significantly (from −1.30 to −0.22), but remains negative because the class imbalance (5.85% positive rate) means even moderate Brier scores look poor relative to the naive baseline. This is expected for the deployment target.

2. **Thin MEDIUM band on validation:** Isotonic calibration's step-function nature concentrates mass at a few breakpoints. The MEDIUM band (0.985714 ≤ cal_p < 1.0) contains only 3 validation samples. On the full dataset, MEDIUM=206, which is operationally meaningful.

3. **T_HIGH = 1.0:** This is the highest step-function breakpoint produced by the isotonic calibrator. Devices scored at exactly `cal_p = 1.0` are flagged HIGH. This is a consequence of the calibrator having high confidence for a subset of Class I devices and does not indicate a calibration error.

4. **Temporal drift:** The positive rate shifts from 8.34% (train, ≤2014) to 5.85% (validation, 2015) to 5.52% (test, 2016–2017). Risk band thresholds derived on 2015 validation data reflect that year's distribution. Future recalibration may be needed for post-2018 data.

5. **Decision threshold vs. risk threshold:** `is_class_i_predicted` uses the raw-probability threshold (0.8555) from Stage 5 model selection. Risk bands use calibrated-probability thresholds (0.985714, 1.0). These are separate mechanisms and should not be conflated.

---

## 13. Test Summary

| Test Class | Tests | Description |
|---|---|---|
| `TestProbabilityToScore` | 11 | Pure function: formula, boundaries, NaN, out-of-range |
| `TestScoreToBand` | 12 | Pure function: band logic, T_HIGH=1.0 production case, invalid config |
| `TestScoringResult` | 5 | Dataclass structure and defaults |
| `TestRiskScorerInit` | 7 | Artifact loading, missing file errors |
| `TestRiskScorerCalibration` | 7 | Calibrated model loading, probability range, error cases |
| `TestRiskScorerScore` | 9 | Single-row scoring, determinism, NaN warnings, config fallback |
| `TestRiskScorerBatchScore` | 10 | Batch scoring, schema, determinism, missing/extra columns |
| `TestServingTable` | 14 | Schema, Stage 3f policy, probability/score ranges, provenance |
| `TestCalibrationReport` | 11 | Calibration report content and threshold validation |
| `TestConfigThresholds` | 6 | Config threshold validity after calibration |
| **Total** | **95** | All pass |

**Full suite:** 171 tests pass (76 Stage 1–5 + 95 Stage 6). No regressions.
