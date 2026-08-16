# ML Risk Engine Report — Stage 5

> All metrics computed from actual experiment artifacts produced by running  
> `python -m src.models.train` and `python -m src.models.evaluate` on 2026-08-16  
> against the real Stage-4 feature matrix. No synthetic data. No estimates.

---

## 1. Executive Summary

Stage 5 trains and evaluates four model candidates on the leakage-safe feature matrix produced by Stage 4. The winning model — **Random Forest with balanced class weights** — achieves:

| Metric | Validation (2015) | Test (2016–2017) | Holdout (2018) |
|--------|-------------------|-------------------|----------------|
| **PR-AUC** | **0.6055** | **0.5420** | **0.6869** |
| **ROC-AUC** | 0.8880 | 0.8611 | 0.8724 |
| F1 (at threshold) | 0.3740 | **0.5038** | **0.6549** |
| Precision | 0.2480 | **0.9766** | **0.9487** |
| Recall | 0.7600 | 0.3394 | 0.5000 |

The model is **10× better than the random baseline** on PR-AUC (baseline ≈ positive rate ≈ 0.055). The high precision (97.7%) on the test set means nearly every Class I event the model flags is a genuine one — a useful property for triage. The decision threshold was tuned on the validation set to maximise F1.

---

## 2. Task Definition

This model solves **Candidate B — Event Severity Classification** as approved in Stage 3:

```
Task:    Given that a medical device safety event has been initiated,
         predict whether it will be classified as Class I (most severe)
         versus Class II or III.

Target:  is_class_i (binary)
         Positive (1):  action_classification == "Class I"
         Negative (0):  action_classification ∈ {"Class II", "Class III"}

Split:   Train  ≤ 2014-12-31  (38,247 events, pos_rate=8.34%)
         Val    2015           (4,273 events,  pos_rate=5.85%)
         Test   2016–2017      (8,918 events,  pos_rate=5.52%)
         Holdout 2018          (1,361 events,  pos_rate=5.44%)
```

---

## 3. Candidate Models

| # | Model | Imbalance Strategy | Notes |
|---|-------|--------------------|-------|
| 1 | **MajorityClassBaseline** | Prior probability | Always predicts the training positive rate |
| 2 | **Logistic Regression** | `class_weight="balanced"` + `StandardScaler` | Interpretable linear model |
| 3 | **Random Forest** | `class_weight="balanced"` | 300 trees, `min_samples_leaf=5` |
| 4 | **XGBoost** | `scale_pos_weight=10.99` | 500 estimators, `early_stopping_rounds=30` |

No SMOTE or oversampling was used. Class weighting/reweighting avoids introducing synthetic samples into a temporally-ordered dataset.

---

## 4. Validation Leaderboard

All four models evaluated on the 2015 validation split at threshold 0.5 (default):

| Model | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|-------|--------|---------|--------|-----------|-----|
| Baseline | 0.0585 | 0.5000 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.2147 | 0.8218 | 0.900 | 0.103 | 0.185 |
| **Random Forest** ✓ | **0.6055** | **0.8880** | 0.760 | 0.248 | 0.374 |
| XGBoost | 0.5662 | 0.8699 | 0.672 | 0.243 | 0.357 |

**Selection metric: PR-AUC on validation set.**  
PR-AUC is the primary ranking metric because the positive class is 7.6% (1:12.2 imbalance); ROC-AUC can be misleadingly optimistic under class imbalance. Random Forest leads with PR-AUC=0.6055.

---

## 5. Threshold Tuning

The default 0.5 threshold is inappropriate for a 7.6% positive-rate problem — it produces low precision. The decision threshold was tuned on the **validation set** to maximise F1, then applied unchanged to test and holdout.

| | Value |
|-|-------|
| **Optimised threshold** | **0.8555** |
| Val F1 at 0.8555 | 0.6304 |
| Val Precision at 0.8555 | 0.9831 |
| Val Recall at 0.8555 | 0.4640 |

At 0.8555 the model is highly conservative: it only flags an event as Class I when it is very confident. This trades recall for precision — suitable for a triage use case where false alarms have regulatory costs.

---

## 6. Test-Set Evaluation (2016–2017)

**Best model: Random Forest. Threshold: 0.8555.**

| Metric | Value |
|--------|-------|
| PR-AUC | 0.5420 |
| ROC-AUC | 0.8611 |
| F1 | 0.5038 |
| Precision | **0.9766** |
| Recall | 0.3394 |
| Threshold | 0.8555 |

### Confusion Matrix (Test Set, 8,918 events)

|  | Predicted Negative | Predicted Positive |
|--|--------------------|--------------------|
| **Actual Negative** (8,426) | TN = 8,422 | FP = 4 |
| **Actual Positive** (492) | FN = 325 | TP = 167 |

**Interpretation:**
- The model flags 171 events as Class I; 167 are genuine (97.7% precision).
- Of 492 actual Class I events, 167 are caught (33.9% recall). The remaining 325 are missed — a known consequence of the high-precision threshold.
- Only 4 false positives across 8,918 test events. Extremely low false-alarm rate.

---

## 7. Holdout 2018 (Robustness Check)

| Metric | Test (2016–17) | Holdout (2018) | Δ |
|--------|----------------|-----------------|---|
| PR-AUC | 0.5420 | **0.6869** | +0.145 |
| ROC-AUC | 0.8611 | 0.8724 | +0.011 |
| F1 | 0.5038 | **0.6549** | +0.151 |
| Precision | 0.9766 | 0.9487 | −0.028 |
| Recall | 0.3394 | **0.5000** | +0.161 |

The holdout metrics are **better** than test, not worse — strong evidence that the model has not overfit to the 2016–2017 period. The positive rate is stable (5.44% holdout vs 5.52% test).

---

## 8. Feature Importance (Top 20)

Feature importances are mean decrease in impurity from the Random Forest (not SHAP).

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `hist_category_severity_rate` | 0.1319 |
| 2 | `hist_category_class_i_count` | 0.1181 |
| 3 | `hist_category_event_count` | 0.0830 |
| 4 | `mfr_parent_company_freq` | 0.0797 |
| 5 | `device_name_len` | 0.0613 |
| 6 | `device_description_len` | 0.0607 |
| 7 | `hist_mfr_recall_count` | 0.0590 |
| 8 | `hist_mfr_event_count` | 0.0561 |
| 9 | `hist_mfr_severity_rate` | 0.0538 |
| 10 | `hist_mfr_class_i_count` | 0.0326 |
| 11 | `device_classification_General Hospital...` | 0.0152 |
| 12 | `device_classification_missing` | 0.0138 |
| 13 | `device_classification_Anesthesiology Devices` | 0.0122 |
| 14 | `hist_category_severity_rate_available` | 0.0110 |
| 15 | `device_classification_Orthopedic Devices` | 0.0109 |
| 16 | `device_country_USA` | 0.0106 |
| 17 | `device_classification___MISSING__` | 0.0102 |
| 18 | `device_country_AUS` | 0.0101 |
| 19 | `hist_mfr_severity_rate_available` | 0.0100 |
| 20 | `device_classification_Radiology Devices` | 0.0098 |

**Key findings:**
- **Device category history dominates** — the historical Class I rate and count for the event's device category are the top two features. Prior severity patterns in a device category are the strongest predictor of a new event's severity.
- **Manufacturer profile** (parent company frequency, prior recall count, severity rate) contributes ~22% of importance. Well-known large manufacturers have different severity patterns.
- **Text length features** (`device_name_len`, `device_description_len`) rank 5th and 6th, indicating that the complexity/verbosity of device descriptions correlates with severity.
- **Static one-hot features** (device classification categories) contribute collectively but individually rank lower.

---

## 9. Model Summary

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| `n_estimators` | 300 |
| `max_depth` | None (unlimited) |
| `min_samples_leaf` | 5 |
| `class_weight` | `"balanced"` |
| `random_state` | Project seed |
| Features | 62 (Stage 4 matrix) |
| Decision threshold | 0.8555 (tuned on val) |
| Production artifacts | `models/production/` |

---

## 10. Limitations

1. **Not literal failure prediction.** The model predicts severity of already-initiated safety events, not future device failure.

2. **Geographic bias.** Labels exist only for USA (67.7%), Canada (25.6%), Australia (6.7%), El Salvador (0.02%). Model will not generalise to events from other regulatory systems.

3. **Temporal positive-rate drift.** Positive rate drops from 8.34% (train) to 5.52% (test). This is natural temporal drift and was expected from Stage 3.

4. **`reason` text excluded.** The most predictive single feature (the problem description text) was intentionally excluded due to borderline leakage. Including it would likely substantially improve recall.

5. **Device attribute sparsity.** Device classification, risk_class, and implanted status are missing for ~70% of events. The model relies heavily on the `__MISSING__` and `_missing` indicator features.

6. **High-precision / low-recall trade-off.** At threshold 0.8555, the model catches only 33.9% of Class I events on the test set (167/492). For use cases requiring higher recall, the threshold should be lowered, accepting more false positives.

7. **Feature importances are MDI, not SHAP.** Mean decrease in impurity over-emphasises high-cardinality features and should be interpreted directionally. SHAP analysis is deferred to Stage 6.

---

## 11. Reproducibility

```bash
# Activate virtual environment
source venv/bin/activate

# Step 1 — Feature engineering (Stage 4, already done)
python -m src.features.build_features

# Step 2 — Train all 4 candidates (~2.4 seconds)
python -m src.models.train

# Step 3 — Evaluate, select, and promote best model (~0.4 seconds)
python -m src.models.evaluate

# Step 4 — Run test suite (44 tests)
python -m pytest tests/models/test_model.py -v

# Step 5 — Run all leakage tests (32 tests, Stage 4)
python -m pytest tests/features/test_leakage.py -v
```

Total pipeline time (Stage 4 + Stage 5): **< 5 seconds**.

---

## 12. Test Results — 44/44 PASSED

```
============================= test session starts ==============================
platform darwin -- Python 3.14.2, pytest-9.1.1
collected 44 items

tests/models/test_model.py::TestDataLoading::test_parquet_exists[train] PASSED
tests/models/test_model.py::TestDataLoading::test_parquet_exists[validation] PASSED
tests/models/test_model.py::TestDataLoading::test_parquet_exists[test] PASSED
tests/models/test_model.py::TestDataLoading::test_parquet_exists[holdout_2018] PASSED
tests/models/test_model.py::TestDataLoading::test_train_rows PASSED
tests/models/test_model.py::TestDataLoading::test_validation_rows PASSED
tests/models/test_model.py::TestDataLoading::test_test_rows PASSED
tests/models/test_model.py::TestDataLoading::test_holdout_rows PASSED
tests/models/test_model.py::TestDataLoading::test_target_column_present PASSED
tests/models/test_model.py::TestDataLoading::test_target_is_binary PASSED
tests/models/test_model.py::TestDataLoading::test_positive_rate_in_range PASSED
tests/models/test_model.py::TestDataLoading::test_feature_count PASSED
tests/models/test_model.py::TestDataLoading::test_metadata_not_in_X PASSED
tests/models/test_model.py::TestDataLoading::test_no_all_nan_columns PASSED
tests/models/test_model.py::TestDataLoading::test_consistent_feature_columns_across_splits PASSED
tests/models/test_model.py::TestLeakageInModel::test_target_not_in_feature_columns[train] PASSED
... (4 parametrized)
tests/models/test_model.py::TestLeakageInModel::test_no_prohibited_features_in_X[train] PASSED
... (4 parametrized)
tests/models/test_model.py::TestLeakageInModel::test_action_classification_not_in_train PASSED
tests/models/test_model.py::TestLeakageInModel::test_reason_not_in_features PASSED
tests/models/test_model.py::TestMetricsComputation::test_returns_required_keys PASSED
tests/models/test_model.py::TestMetricsComputation::test_perfect_classifier PASSED
tests/models/test_model.py::TestMetricsComputation::test_random_classifier_pr_auc_near_positive_rate PASSED
tests/models/test_model.py::TestMetricsComputation::test_n_samples_correct PASSED
tests/models/test_model.py::TestMetricsComputation::test_confusion_matrix_sums PASSED
tests/models/test_model.py::TestProductionModel::test_model_pkl_exists PASSED
tests/models/test_model.py::TestProductionModel::test_model_card_exists PASSED
tests/models/test_model.py::TestProductionModel::test_test_metrics_exists PASSED
tests/models/test_model.py::TestProductionModel::test_model_loads PASSED
tests/models/test_model.py::TestProductionModel::test_model_predict_shape PASSED
tests/models/test_model.py::TestProductionModel::test_probas_in_unit_interval PASSED
tests/models/test_model.py::TestProductionModel::test_feature_importance_exists_for_tree_models PASSED
tests/models/test_model.py::TestTestMetrics::test_test_metrics_has_required_keys PASSED
tests/models/test_model.py::TestTestMetrics::test_pr_auc_above_baseline PASSED
tests/models/test_model.py::TestTestMetrics::test_roc_auc_above_0_5 PASSED
tests/models/test_model.py::TestTestMetrics::test_recall_nonzero PASSED
tests/models/test_model.py::TestTestMetrics::test_threshold_in_unit_interval PASSED
tests/models/test_model.py::TestTestMetrics::test_test_n_samples_correct PASSED
tests/models/test_model.py::TestTestMetrics::test_holdout_pr_auc_reasonable PASSED

============================== 44 passed in 1.47s ==============================
```

| Test Class | Tests | Status |
|-----------|-------|--------|
| `TestDataLoading` | 15 | ✅ All passed |
| `TestLeakageInModel` | 10 | ✅ All passed |
| `TestMetricsComputation` | 5 | ✅ All passed |
| `TestProductionModel` | 7 | ✅ All passed |
| `TestTestMetrics` | 7 | ✅ All passed |

---

## 13. Output Files

| File | Contents |
|------|----------|
| `models/experiments/baseline_majority_*/` | Baseline experiment artifacts |
| `models/experiments/logistic_regression_*/` | LR experiment artifacts |
| `models/experiments/random_forest_*/` | RF experiment artifacts (winner) |
| `models/experiments/xgboost_*/` | XGBoost experiment artifacts |
| `models/production/model.pkl` | Serialised Random Forest |
| `models/production/model_card.json` | Human-readable model card |
| `models/production/test_metrics.json` | Val/test/holdout metrics at chosen threshold |
| `models/production/feature_importance.json` | 62 features ranked by MDI |
| `models/production/feature_metadata.json` | Stage 4 feature inventory (copied) |
