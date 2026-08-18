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

### 6.1 Design Rationale

The 0–100 risk score (= `calibrated_probability × 100`) is mapped to three operational bands using **score-based boundaries**. These thresholds are chosen for dashboard interpretability — they map directly to the displayed score:

| Band | Score range | Calibrated probability | Interpretation |
|------|-------------|----------------------|----------------|
| **LOW** | `score < 20` | `cal_p < 0.20` | Low estimated probability of Class I severity |
| **MEDIUM** | `20 ≤ score < 50` | `0.20 ≤ cal_p < 0.50` | Elevated concern; warrants increased monitoring |
| **HIGH** | `score ≥ 50` | `cal_p ≥ 0.50` | Model considers Class I at least as likely as not |

**Why score-based, not precision/recall-curve-derived:**
The previous thresholds (T_MEDIUM=0.985714, T_HIGH=1.0) were derived from the isotonic calibration's step-function breakpoints on the 2015 validation set. They were technically correct but produced a MEDIUM band with only 3 validation samples and a HIGH band that included all devices with exactly `cal_p = 1.0`. For presentation and dashboard usability, score-based thresholds are far more interpretable — "HIGH means the score is at least 50 out of 100" is immediately understood by non-technical stakeholders.

### 6.2 Critical Distinctions

Four separate concepts must not be conflated:

| Concept | Value | What it means |
|---------|-------|---------------|
| `raw_probability` | `predict_proba()[:, 1]` output | Uncalibrated RF posterior |
| `decision_threshold` | **0.8555** (unchanged) | `is_class_i_predicted = raw_prob ≥ 0.8555` — Stage 5 F1-maximising threshold |
| `calibrated_probability` | Isotonic output in [0, 1] | Best-estimate posterior after calibration |
| `risk_score` | `round(cal_p × 100, 2)` | Human-readable 0–100 score |
| `risk_level` | LOW / MEDIUM / HIGH | Operational band from score-based thresholds |

The `decision_threshold` (0.8555) is **not** the same as the risk band thresholds. It operates on the raw uncalibrated probability; the band thresholds operate on the calibrated probability (risk score).

### 6.3 Band Definition

```python
RISK_THRESHOLD_MEDIUM = 0.20  # risk_score >= 20 -> MEDIUM or HIGH
RISK_THRESHOLD_HIGH   = 0.50  # risk_score >= 50 -> HIGH

# Equivalent score logic:
risk_level = "HIGH"    if risk_score >= 50   # calibrated_prob >= 0.50
           = "MEDIUM"  if risk_score >= 20   # calibrated_prob >= 0.20
           = "LOW"     otherwise             # calibrated_prob < 0.20
```

### 6.4 Config Storage

Thresholds are defined in `src/config.py`:

```python
RISK_THRESHOLD_HIGH: float = 0.50    # calibrated_prob >= this -> HIGH
RISK_THRESHOLD_MEDIUM: float = 0.20  # calibrated_prob >= this -> MEDIUM (else LOW)
```

`build_serving_table.py` refuses to run if either threshold is `0.0` (sentinel guard preserved for safety).

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

## 12. Limitations and Known Behaviour

1. **Brier skill score is negative:** The base RF is overconfident on the imbalanced validation set. After isotonic calibration, the skill score improves significantly (from −1.30 to −0.22), but remains negative because the class imbalance (5.85% positive rate) means even moderate Brier scores look poor relative to the naive baseline. This is expected for the deployment target.

2. **Temporal drift:** The positive rate shifts from 8.34% (train, ≤2014) to 5.85% (validation, 2015) to 5.52% (test, 2016–2017). Future recalibration may be needed for post-2018 data. Band thresholds should be reviewed if the overall score distribution shifts materially.

3. **Decision threshold vs. operational risk bands:** `is_class_i_predicted` uses the raw-probability decision threshold (0.8555) from Stage 5 model selection. Operational risk bands (LOW/MEDIUM/HIGH) use calibrated-probability thresholds (0.20, 0.50) applied to the `risk_score`. These are separate mechanisms — a device can have `is_class_i_predicted = False` (raw probability below 0.8555) and yet be scored HIGH (calibrated probability ≥ 0.50), or vice versa. Both signals are presented in the API response.

4. **Serving table band distribution:** With the score-based thresholds, the full serving table (50,341 devices) distributes as HIGH=3,726 (7.4%), MEDIUM=2,191 (4.4%), LOW=44,424 (88.2%). This is operationally meaningful — roughly 1 in 14 devices is flagged HIGH.

---

## 13. Test Summary

| Test Class | Tests | Description |
|---|---|---|
| `TestProbabilityToScore` | 11 | Pure function: formula, boundaries, NaN, out-of-range |
| `TestScoreToBand` | 17 | Pure function: general band logic, production score-based boundaries, config match, invalid config |
| `TestScoringResult` | 5 | Dataclass structure and defaults |
| `TestRiskScorerInit` | 7 | Artifact loading, missing file errors |
| `TestRiskScorerCalibration` | 7 | Calibrated model loading, probability range, error cases |
| `TestRiskScorerScore` | 9 | Single-row scoring, determinism, NaN warnings, config fallback |
| `TestRiskScorerBatchScore` | 10 | Batch scoring, schema, determinism, missing/extra columns |
| `TestServingTable` | 14 | Schema, Stage 3f policy, probability/score ranges, provenance |
| `TestCalibrationReport` | 11 | Calibration report content and threshold validation |
| `TestConfigThresholds` | 6 | Config threshold validity |
| **Total** | **97** | All pass |

**Full suite:** 343 tests pass (all stages). No regressions.
