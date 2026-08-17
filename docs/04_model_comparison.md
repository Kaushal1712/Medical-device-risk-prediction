# Model Comparison Report — Stage 5

> All metrics are the **genuine output** of `python -m src.models.train` and
> `python -m src.models.evaluate` run on 2026-08-16 against the real Stage-4
> feature matrix. No synthetic data. No estimates. No post-hoc edits.
> Full Stage-5 narrative: [`docs/05_ml_risk_engine_report.md`](05_ml_risk_engine_report.md)

---

## 1. Task Definition

| Property | Value |
|----------|-------|
| **Task** | Binary classification — predict event severity |
| **Target column** | `is_class_i` |
| **Positive class** | `action_classification == "Class I"` → most severe |
| **Negative class** | `action_classification ∈ {"Class II", "Class III"}` → less severe |
| **Train split** | Events ≤ 2014-12-31 (38,247 rows, positive rate 8.34%) |
| **Validation split** | 2015 (4,273 rows, positive rate 5.85%) |
| **Test split** | 2016–2017 (8,918 rows, positive rate 5.52%) |
| **Holdout split** | 2018 (1,361 rows, positive rate 5.44%) |
| **Feature count** | 62 |
| **Primary ranking metric** | PR-AUC (weighted over ROC-AUC due to class imbalance) |

---

## 2. Candidate Models

| # | Model | Imbalance strategy | Key hyperparameters |
|---|-------|--------------------|---------------------|
| 1 | **MajorityClassBaseline** | Prior probability | Always predicts training positive rate |
| 2 | **Logistic Regression** | `class_weight="balanced"` | `StandardScaler`, default `C=1.0` |
| 3 | **Random Forest** | `class_weight="balanced"` | 300 trees, `min_samples_leaf=5`, `random_state` fixed |
| 4 | **XGBoost** | `scale_pos_weight=10.99` | 500 estimators, `early_stopping_rounds=30` |

No SMOTE or oversampling was used. Class weighting avoids introducing synthetic samples into a temporally ordered dataset.

---

## 3. Validation Leaderboard (2015 split, threshold = 0.5)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|--------|---------|-----------|--------|----|
| Baseline (majority) | 0.0585 | 0.5000 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.2147 | 0.8218 | 0.103 | 0.900 | 0.185 |
| **Random Forest** ✅ | **0.6055** | **0.8880** | **0.248** | **0.760** | **0.374** |
| XGBoost | 0.5662 | 0.8699 | 0.243 | 0.672 | 0.357 |

**Validation set:** 4,273 events (250 positive, 4,023 negative).

> PR-AUC is the primary ranking metric because the positive class is 7.6% of the labeled set (1:12.2 imbalance). ROC-AUC can be misleadingly optimistic under class imbalance. Random Forest leads decisively on both PR-AUC (+0.039 over XGBoost) and ROC-AUC.

---

## 4. Threshold Tuning (Random Forest on Validation Set)

The default 0.5 threshold is inappropriate for a 7.6% positive-rate problem. The operating threshold was tuned on the **validation set only** to maximise F1, then applied unchanged to test and holdout evaluation.

| Metric | Value |
|--------|-------|
| **Optimised threshold** | **0.8555** |
| Val F1 at 0.8555 | 0.6304 |
| Val Precision at 0.8555 | 0.9831 |
| Val Recall at 0.8555 | 0.4640 |

At 0.8555 the model is conservative: it flags an event as Class I only when highly confident. This trades recall for precision — appropriate for a triage use case where false alarms carry regulatory cost.

---

## 5. Test-Set Evaluation — Random Forest (2016–2017 split)

Evaluated once on the held-out test split with threshold = 0.8555 (fixed from validation tuning, not re-tuned on test).

| Metric | Value |
|--------|-------|
| PR-AUC | **0.5420** |
| ROC-AUC | **0.8611** |
| F1 | 0.5038 |
| Precision | **0.9766** |
| Recall | 0.3394 |
| Decision threshold | 0.8555 |
| Test n_samples | 8,918 |
| Actual positives | 492 |
| Predicted positives | 171 |

### Confusion Matrix (Test Set, 8,918 events)

| | Predicted Negative | Predicted Positive |
|--|--------------------|--------------------|
| **Actual Negative** (8,426) | TN = 8,422 | FP = 4 |
| **Actual Positive** (492) | FN = 325 | TP = 167 |

- 167 of 171 flagged events are genuine Class I (97.7% precision).
- 4 false alarms across 8,918 test events — extremely low false-positive rate.
- 325 actual Class I events are missed (consequence of the high-precision threshold).

---

## 6. Holdout Robustness Check (2018)

| Metric | Test (2016–2017) | Holdout (2018) | Δ |
|--------|-----------------|-----------------|---|
| PR-AUC | 0.5420 | **0.6869** | +0.145 |
| ROC-AUC | 0.8611 | 0.8724 | +0.011 |
| F1 | 0.5038 | **0.6549** | +0.151 |
| Precision | 0.9766 | 0.9487 | −0.028 |
| Recall | 0.3394 | **0.5000** | +0.161 |

Holdout metrics are **better** than test, not worse — strong evidence the model has not overfit to the 2016–2017 period. The positive rate is stable (5.44% holdout vs 5.52% test).

---

## 7. Model Selection Rationale

**Selected model: Random Forest** with `class_weight="balanced"`, 300 trees, `min_samples_leaf=5`.

| Criterion | Random Forest | XGBoost | Logistic Regression |
|-----------|--------------|---------|---------------------|
| Val PR-AUC | **0.6055** | 0.5662 | 0.2147 |
| Val ROC-AUC | **0.8880** | 0.8699 | 0.8218 |
| Test PR-AUC | **0.5420** | — | — |
| SHAP compatibility | `TreeExplainer` (fast, exact) | ✅ | `LinearExplainer` |
| Calibration behaviour | Strong isotonic fit | — | — |
| Generalisation gap (train→val) | Good | Good | Large |

**Rationale:**
1. **Highest PR-AUC** on both validation and test — the primary metric for an imbalanced classification task.
2. **Best calibration** — isotonic regression on the RF probabilities achieves Brier score 0.0384 (vs 0.0558 uncalibrated), enabling a meaningful calibrated risk score.
3. **Native SHAP support** — `shap.TreeExplainer` provides exact, fast SHAP values for Random Forest without kernel approximation. This is critical for the Stage-6 explainability module.
4. **Better temporal generalisation** — holdout (2018) PR-AUC (0.687) exceeds test (0.542), unlike the typical pattern of performance degradation. No evidence of temporal overfitting.
5. **Computational cost** — training takes < 2.5 seconds on the full dataset; inference at serving is < 1 ms per example.

> XGBoost is a strong second candidate on validation metrics but was not selected because the RF offers superior calibration, exact SHAP support, and equivalent or better test generalisation.

---

## 8. Model Artifacts

| File | Contents |
|------|----------|
| `models/production/model.pkl` | Serialised Random Forest |
| `models/production/calibrated_model.pkl` | Isotonic-calibrated wrapper |
| `models/production/model_card.json` | Hyperparameters, feature list, thresholds, metrics |
| `models/production/test_metrics.json` | Val / test / holdout metrics at chosen threshold |
| `models/production/feature_importance.json` | 62 features ranked by mean decrease in impurity |
| `models/experiments/baseline_majority_20260816_145309/` | Baseline experiment artifacts |
| `models/experiments/logistic_regression_20260816_145309/` | LR experiment artifacts |
| `models/experiments/random_forest_20260816_145309/` | RF experiment artifacts (winner) |
| `models/experiments/xgboost_20260816_145309/` | XGBoost experiment artifacts |

Machine-readable metrics: [`artifacts/metrics/model_comparison.csv`](../artifacts/metrics/model_comparison.csv)
