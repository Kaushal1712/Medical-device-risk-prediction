# Feature Engineering Report — Stage 4

> All statistics verified from **actual** `data/features/*.parquet` outputs produced by running  
> `python -m src.features.build_features` on 2026-08-15 against the real project data.  
> No synthetic or sample data was used. No model trained.

---

## 1. Target Definition

| Property | Value |
|----------|-------|
| **Target column** | `is_class_i` |
| **Positive class (1)** | `action_classification == "Class I"` — immediate hazard to health |
| **Negative class (0)** | `action_classification ∈ {"Class II", "Class III"}` — less severe |
| **Excluded** | null `action_classification`, `"Unclassified Correction"` (3 rows), `"Voluntary recall"` (1 row) |
| **Total labeled** | 52,946 events |
| **Positive count** | 4,022 (7.6%) |
| **Negative count** | 48,924 (92.4%) |

---

## 2. Temporal Split

| Split | Period | Rows | Positive | Pos Rate | Negative | Date Range |
|-------|--------|------|----------|----------|----------|------------|
| **Train** | ≤ 2014-12-31 | 38,247 | 3,191 | 8.34% | 35,056 | 1991-08-07 → 2014-12-31 |
| **Validation** | 2015 | 4,273 | 250 | 5.85% | 4,023 | 2015-01-01 → 2015-12-31 |
| **Test** | 2016–2017 | 8,918 | 492 | 5.52% | 8,426 | 2016-01-04 → 2017-12-29 |
| **Holdout** | 2018 | 1,361 | 74 | 5.44% | 1,287 | 2018-01-02 → 2018-07-09 |
| **TOTAL** | | **52,799** | | | | |

- **147 labeled events** had no `event_date` and were excluded from all splits.
- The positive rate decreases from train (8.34%) to test (5.52%), reflecting a natural temporal shift — this is expected with temporal splits and must be accounted for during model evaluation.
- No random splitting was used. The primary evaluation is temporal.

---

## 3. Feature Inventory — 62 Features

### Tier 1: Safe Static Features (46 features after encoding)

| Source | Original Column | Encoding | # Encoded Cols |
|--------|----------------|----------|----------------|
| Device | `device_classification` | One-hot (17 categories + \_\_MISSING\_\_ + \_\_UNKNOWN\_\_) | 18 |
| Device | `device_risk_class` | One-hot (7 categories + \_\_MISSING\_\_ + \_\_UNKNOWN\_\_) | 9 |
| Device | `device_implanted` | One-hot (YES/NO + \_\_MISSING\_\_ + \_\_UNKNOWN\_\_) | 4 |
| Device | `device_country` | One-hot (3 countries + \_\_UNKNOWN\_\_) | 4 |
| Device | `device_description` | Length (chars) | 1 |
| Device | `device_name` | Length (chars) | 1 |
| Manufacturer | `mfr_parent_company` | Frequency encoding (1,075 categories) | 1 |
| Manufacturer | `mfr_source` | One-hot (3 sources + \_\_UNKNOWN\_\_) | 4 |
| Event | `country` | One-hot (3 countries + \_\_UNKNOWN\_\_) | 4 |
| Missing indicators | `device_classification_missing` | Binary (0/1) | 1 |
| Missing indicators | `device_risk_class_missing` | Binary (0/1) | 1 |
| Missing indicators | `device_implanted_missing` | Binary (0/1) | 1 |

### Tier 2a: Event Features (2 features after encoding)

| Column | Encoding | # Encoded Cols |
|--------|----------|----------------|
| `type` | One-hot (`Recall` + `__UNKNOWN__`) | 2 |

Note: In the labeled subset (USA/CAN/AUS/SLV), all events are type "Recall". The `type___UNKNOWN__` column handles potential unseen event types at inference.

### Tier 2b: Temporal Historical Features (14 features)

| Feature | Description | Same-Day Handling |
|---------|-------------|-------------------|
| `hist_device_event_count` | Prior events for this device | Excluded |
| `hist_device_class_i_count` | Prior Class I events for this device | Excluded |
| `hist_device_recall_count` | Prior recalls for this device | Excluded |
| `hist_mfr_event_count` | Prior events for this manufacturer | Excluded |
| `hist_mfr_class_i_count` | Prior Class I events for this manufacturer | Excluded |
| `hist_mfr_recall_count` | Prior recalls for this manufacturer | Excluded |
| `hist_mfr_severity_rate` | Historical Class I rate for manufacturer | Excluded |
| `hist_mfr_severity_rate_available` | Whether manufacturer has prior history | — |
| `hist_category_event_count` | Prior events for this device category | Excluded |
| `hist_category_class_i_count` | Prior Class I for this device category | Excluded |
| `hist_category_severity_rate` | Historical Class I rate for category | Excluded |
| `hist_category_severity_rate_available` | Whether category has prior history | — |

**All historical features use strict `event_date < T`.** Same-day events are excluded using the `groupby([group_col, event_date]).transform('min')` approach.

### Tier 3: Excluded (reason text)

`reason` is **excluded from the primary feature set**. It describes the problem that influences severity determination and represents a borderline causal/leakage concern. The architecture supports adding NLP features as an optional experiment in a future stage.

### Prohibited Features (fully blocked)

`action`, `action_summary`, `determined_cause`, `status`, `date_terminated`, `action_classification`, `action_level`, `reason`, `target_audience` — all post-event or target-derived fields, plus database/identifier columns listed in `PROHIBITED_FEATURES` frozenset in `build_features.py`.

---

## 4. Historical Feature Methodology

### Vectorized Cumulative Approach

Historical aggregates are computed using a fully vectorized pandas approach (`_cumcount_before` in `src/features/build_features.py`):

1. Sort all events by `event_date`
2. Compute `groupby(group_col).cumcount()` within each group
3. Compute cumulative sum of the target metric per group
4. Shift by subtracting the current row's contribution (to exclude itself)
5. **Same-day exclusion:** for each `(group_col, event_date)` group, take `transform('min')` of the cumulative sum — this ensures all events on the same day receive the count from *before* that date, not including same-day peers

The history source uses **all 116,514 dated events** from `merged.parquet` (including unlabeled), not just the 52,946 labeled events — this maximizes the informativeness of historical context.

### Groups computed

- **Device-level:** `device_id` (3 metrics: event count, Class I count, recall count)
- **Manufacturer-level:** `manufacturer_id` (3 metrics: event count, Class I count, recall count)  
- **Category-level:** `device_classification` (2 metrics: event count, Class I count)
- **Derived rates:** `hist_mfr_severity_rate = hist_mfr_class_i_count / hist_mfr_event_count`
- **Availability flags:** `hist_mfr_severity_rate_available`, `hist_category_severity_rate_available`

---

## 5. Missing-Value Strategy

| Strategy | Columns | Rationale |
|----------|---------|-----------| 
| `__MISSING__` category | All categoricals | Explicit handling; missingness is informative (~70% of device attributes are missing for non-US devices) |
| `__UNKNOWN__` category | All categoricals | Unseen categories at inference time are safely handled |
| Count → 0 | `hist_*_count` features | No prior history = zero events |
| Rate → 0.0 + indicator | `hist_*_severity_rate` | No history to compute rate; `_available` indicator preserves info |
| Missing indicator | `device_classification_missing`, `device_risk_class_missing`, `device_implanted_missing` | High-missingness fields (~70%) — the absence is itself informative |
| Length → 0 | `device_description_len`, `device_name_len` | Missing text = zero length |

**No rows were dropped due to missing values.** Missingness is meaningful in this dataset and is encoded as a feature.

---

## 6. Categorical Encoding Strategy

| Method | Threshold | Columns | Approach |
|--------|-----------|---------|----------|
| **One-hot** | ≤ 30 unique values | `device_classification`, `device_risk_class`, `device_implanted`, `device_country`, `mfr_source`, `country`, `type` | Binary columns per category |
| **Frequency** | > 30 unique values | `mfr_parent_company` (1,075 categories) | Train-set proportion mapping |

Encoding is fitted on **train data only**. Unseen categories in validation/test/holdout are mapped to `__UNKNOWN__`.

---

## 7. Leakage Prevention

### Methodology

1. **Prohibited features list** — `PROHIBITED_FEATURES` frozenset blocks all post-event fields at the code level with a runtime check.
2. **Historical features** — computed using vectorized cumulative sums with strict `event_date < T` and same-day exclusion via `transform('min')`.
3. **Encoding** — categorical vocabularies and frequency maps computed from train set only.
4. **Temporal split** — no random splitting; pure date-based partitioning.

### Test Results — 32/32 PASSED

Pipeline executed 2026-08-15. All tests passed in **0.58 seconds**.

```
python -m pytest tests/features/test_leakage.py -v

============================= test session starts ==============================
platform darwin -- Python 3.14.2, pytest-9.1.1
collected 32 items

tests/features/test_leakage.py::TestTargetExclusion::test_target_not_in_feature_columns PASSED
tests/features/test_leakage.py::TestTargetExclusion::test_action_classification_not_in_features PASSED
tests/features/test_leakage.py::TestProhibitedFields::test_prohibited_field_not_in_features[action] PASSED
... (9 parametrized prohibited field tests)
tests/features/test_leakage.py::TestProhibitedFields::test_prohibited_field_not_in_train_data[action] PASSED
... (9 parametrized data tests)
tests/features/test_leakage.py::TestHistoricalLeakage::test_hist_device_count_is_zero_for_first_events PASSED
tests/features/test_leakage.py::TestHistoricalLeakage::test_hist_mfr_count_nonnegative PASSED
tests/features/test_leakage.py::TestHistoricalLeakage::test_hist_class_i_leq_event_count PASSED
tests/features/test_leakage.py::TestSameDayExclusion::test_same_day_device_events_get_same_history PASSED
tests/features/test_leakage.py::TestNoFutureLeakage::test_no_future_events_in_train_history PASSED
tests/features/test_leakage.py::TestTemporalSplit::test_train_before_2015 PASSED
tests/features/test_leakage.py::TestTemporalSplit::test_validation_in_2015 PASSED
tests/features/test_leakage.py::TestTemporalSplit::test_test_in_2016_2017 PASSED
tests/features/test_leakage.py::TestTemporalSplit::test_holdout_in_2018 PASSED
tests/features/test_leakage.py::TestTemporalSplit::test_no_overlap_between_splits PASSED
tests/features/test_leakage.py::TestRowCounts::test_total_labeled_events PASSED
tests/features/test_leakage.py::TestRowCounts::test_positive_rate_reasonable PASSED

============================== 32 passed in 0.58s ==============================
```

| Test Class | Tests | Status |
|-----------|-------|--------|
| `TestTargetExclusion` | 2 | ✅ All passed |
| `TestProhibitedFields` | 18 | ✅ All passed |
| `TestHistoricalLeakage` | 3 | ✅ All passed |
| `TestSameDayExclusion` | 1 | ✅ All passed |
| `TestNoFutureLeakage` | 1 | ✅ All passed |
| `TestTemporalSplit` | 5 | ✅ All passed |
| `TestRowCounts` | 2 | ✅ All passed |

---

## 8. Feature Matrix Dimensions

| Split | Rows | Features | Target | Metadata | Total Columns |
|-------|------|----------|--------|----------|---------------|
| Train | 38,247 | 62 | 1 | 5 | 68 |
| Validation | 4,273 | 62 | 1 | 5 | 68 |
| Test | 8,918 | 62 | 1 | 5 | 68 |
| Holdout 2018 | 1,361 | 62 | 1 | 5 | 68 |

Metadata columns (`id`, `device_id`, `manufacturer_id`, `event_date`, `event_date_available`) are retained for traceability but must be dropped before model input.

---

## 9. Output Files

| File | Size | Contents |
|------|------|----------|
| `data/features/train.parquet` | 1,109 KB | 38,247 × 68 |
| `data/features/validation.parquet` | 173 KB | 4,273 × 68 |
| `data/features/test.parquet` | 301 KB | 8,918 × 68 |
| `data/features/holdout_2018.parquet` | 92 KB | 1,361 × 68 |
| `data/features/feature_metadata.json` | 6.3 KB | Feature inventory, encoding metadata, split stats |

---

## 10. Limitations

1. **All labeled events come from 4 countries** (USA 67.7%, CAN 25.6%, AUS 6.7%, SLV 0.02%). The model will not generalize to events from other regulatory systems.

2. **Event type has near-zero variance** in the labeled subset — all labeled events are "Recall" type. The one-hot columns `type_Recall` and `type___UNKNOWN__` provide minimal signal.

3. **Device attribute missingness is ~70%** for classification, risk_class, and implanted. The missing indicators capture this, but feature sparsity is high.

4. **`mfr_parent_company` frequency encoding** maps 1,075 categories to train-set frequencies. Rare manufacturers may have unreliable frequencies.

5. **Positive rate shifts temporally** — from 8.34% (train) to 5.52% (test). This temporal drift must be considered during model evaluation.

6. **`reason` text excluded** — the most predictive feature is intentionally excluded from the primary model due to borderline leakage. This limits model performance but ensures scientific defensibility.

7. **Holdout ends 2018-07-09** — only half-year 2018 data is available in the holdout set.

---

## 11. Reproducibility

```bash
# Activate virtual environment
source venv/bin/activate

# Run feature engineering pipeline (~0.6 seconds)
python -m src.features.build_features

# Run leakage tests (32 tests, ~0.6 seconds)
python -m pytest tests/features/test_leakage.py -v

# Explore in notebook
jupyter notebook notebooks/03_feature_engineering.ipynb
```

Pipeline execution time: **~0.6 seconds**.

---

## 12. Bug Fixes Applied During Stage 4

Three bugs were discovered and fixed during initial pipeline execution (by the previous agent):

1. **NA-safe comparison** (line 217): `(es[col_name] == value).astype(int)` crashed with `pd.NA` from ArrowDtype. Fixed by adding `.fillna(False)` before `.astype(int)`.

2. **Redundant column drop** (line 299): `hist.drop(columns=["_cat_group"])` crashed because `_cumcount_before()` already drops all `_`-prefixed columns internally. Fixed by removing the explicit drop.

3. **Leakage check scope** (line 438-441): The prohibited-feature check compared against `keep_cols` which included metadata columns. Since `id` appears in both `PROHIBITED_FEATURES` and `METADATA_COLS`, it falsely triggered. Fixed by checking only `feature_only = keep_cols - METADATA_COLS - {TARGET_BINARY_NAME}`.

All three fixes were verified through the 32 leakage tests.
