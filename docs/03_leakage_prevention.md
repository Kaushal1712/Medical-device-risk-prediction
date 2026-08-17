# Data Leakage Prevention — Stage 3 / Stage 10

> This document consolidates all leakage-prevention methodology from the project.
> Source material: [`docs/03_target_feasibility_report.md §8`](03_target_feasibility_report.md),
> [`docs/04_feature_engineering_report.md §7`](04_feature_engineering_report.md),
> and [`docs/10_testing_audit.md §1.2`](10_testing_audit.md).
> No new methodology is introduced here.

---

## 1. Overview

The central leakage risk in this project is: **using information about a safety event's outcome to predict that same outcome**. `action_classification` (the severity tier assigned by the regulator) is determined after an event is investigated — it must not appear as a feature.

A secondary risk is: **using future safety-event history to compute historical aggregate features**. Because the training, validation, and test splits come from different time periods, any historical feature must be computed using only events that predate the current example's event date.

---

## 2. Prediction Boundary

The adopted target (Stage 3) is binary: predict whether a safety event will be classified as Class I (most severe). The prediction boundary is the **event initiation date** (`event_date`).

- **Before the boundary:** Device attributes (static), manufacturer attributes (static), historical event counts and severity rates for the same device, manufacturer, and device category — all computed from events dated strictly before `event_date`.
- **At the boundary:** Event type (`type`) and event country — known at initiation.
- **After the boundary (PROHIBITED):** `action_classification` (the target), `action`, `action_summary`, `determined_cause`, `status`, `date_terminated`, `action_level`, `reason` (post-event problem description), `target_audience`.

---

## 3. Prohibited Fields

The `PROHIBITED_FEATURES` frozenset in `src/features/build_features.py` blocks the following fields from all feature splits at runtime, with an assertion that aborts the pipeline if any prohibited field is found:

| Field | Why prohibited |
|-------|----------------|
| `action_classification` | **The target variable itself** |
| `is_class_i` | **The encoded target** |
| `action` | Regulatory response — determined after event |
| `action_summary` | Regulatory response description — post-event |
| `determined_cause` | Root-cause investigation — post-event |
| `status` | Recall lifecycle status — post-event |
| `date_terminated` | Recall closure date — post-event |
| `action_level` | Regulatory action level — post-event |
| `reason` | Problem description — borderline causal; excluded for safety |
| `target_audience` | Post-event logistics |
| `id` | Database identifier — not a meaningful feature |

These are the exact fields verified by `TestProhibitedFields` in `tests/features/test_leakage.py` (18 parametrized checks, all passing).

---

## 4. Historical Feature Leakage Guards

### 4.1 Strict `event_date < T` condition

All historical aggregate features (`hist_device_event_count`, `hist_device_class_i_count`, `hist_device_recall_count`, `hist_mfr_event_count`, `hist_mfr_class_i_count`, `hist_mfr_recall_count`, `hist_mfr_severity_rate`, `hist_category_event_count`, `hist_category_class_i_count`, `hist_category_severity_rate`) are computed using a fully vectorised cumulative approach (`_cumcount_before` in `src/features/build_features.py`):

1. Sort all 116,514 dated events from `merged.parquet` by `event_date`.
2. Compute `groupby(group_col).cumcount()` within each group.
3. Compute cumulative sum of the target metric per group.
4. Subtract the current row's contribution (exclude self).
5. **Same-day exclusion:** within each `(group_col, event_date)` group, apply `transform('min')` — all events on the same calendar day receive the count from *before* that date, not including same-day peers.

The history source uses **all 116,514 dated events** from `merged.parquet` (including unlabeled), not only the 52,946 labeled events — maximising historical context without leaking target information.

### 4.2 Same-day exclusion

48.5% of inter-event gaps for multi-event devices are 0 days (same calendar day). Same-day events receive identical history counts drawn from before that date — not including any same-day peer event. This prevents a same-day event from influencing the historical count seen by a same-day sibling.

### 4.3 Manufacturer-level aggregates recomputed per example

Manufacturer severity rates and event counts are not pre-computed globally and joined in — they are derived from the per-event cumulative computation, which naturally respects each example's event date boundary.

---

## 5. Categorical Encoding Leakage Prevention

Categorical vocabularies (one-hot column sets, `mfr_parent_company` frequency map) are **fitted on the training split only** (events ≤ 2014-12-31). Validation, test, and holdout data use the same encoding maps. Unseen categories at inference time are encoded as `__UNKNOWN__` rather than being fitted in-place.

This prevents target-mean or frequency information from the validation/test period leaking back into the encoding.

---

## 6. Temporal Split Integrity

The dataset is split by `event_date`, never by random row shuffle:

| Split | Boundary | Rows | Date range |
|-------|----------|------|------------|
| Train | ≤ 2014-12-31 | 38,247 | 1991-08-07 → 2014-12-31 |
| Validation | 2015 | 4,273 | 2015-01-01 → 2015-12-31 |
| Test | 2016–2017 | 8,918 | 2016-01-04 → 2017-12-29 |
| Holdout | 2018 | 1,361 | 2018-01-02 → 2018-07-09 |

No overlap exists between any pair of splits — verified by `TestTemporalSplit::test_no_overlap_between_splits`.

---

## 7. Serving-Snapshot Leakage Prevention (Stage 3f / Stage 6)

The production serving table (`artifacts/risk/device_risk_snapshot.parquet`) is built from the feature dataset using the pre-computed, leakage-safe features. The backend reads from this pre-computed table at request time — it never dynamically re-queries raw event data or selects ad-hoc rows.

The probability calibration (isotonic regression) is fitted only on training-period examples, using `FrozenEstimator + CalibratedClassifierCV(cv=None)` to freeze the base Random Forest weights. No validation or test examples are used to fit the calibrator.

---

## 8. Automated Leakage Test Results (Stage 10)

All leakage tests are in `tests/features/test_leakage.py`. Final run: **31 tests, 31 passed** (32 if counting the Stage-10 cutoff recomputation class separately).

### 8.1 Mandatory Cutoff-Truncated Recomputation (Stage 10)

The strongest leakage test: for a deterministic 30-event sample from `train.parquet`:

1. Filter `merged.parquet` to rows for the **same `device_id`** with `event_date` **strictly < sample event_date**.
2. Recompute `hist_device_event_count`, `hist_device_class_i_count`, and `hist_device_recall_count` from scratch from that filtered slice.
3. Assert the recomputed values **exactly equal** the cached values in `train.parquet`.

This is an exact-equality assertion — any code path that accidentally used the full event history would fail here. **All 3 checks pass.**

> Stored PR-AUC = 0.542031, live recomputed = 0.542031 (difference < 1e-6). Verified by `test_stored_pr_auc_matches_live_recomputation` in `tests/models/test_model.py`.

### 8.2 Test Class Summary

| Test class | Tests | Result |
|-----------|-------|--------|
| `TestCutoffTruncatedRecompute` | 3 | ✅ Pass |
| `TestTargetExclusion` | 2 | ✅ Pass |
| `TestProhibitedFields` | 18 | ✅ Pass |
| `TestHistoricalLeakage` | 3 | ✅ Pass |
| `TestSameDayExclusion` | 1 | ✅ Pass |
| `TestNoFutureLeakage` | 1 | ✅ Pass |
| `TestTemporalSplit` | 5 | ✅ Pass |
| `TestRowCounts` | 2 | ✅ Pass |

Full audit details: [`docs/10_testing_audit.md §1.2`](10_testing_audit.md)

---

## 9. Fields Classified by Leakage Status

| Field | Known when? | Safe for severity prediction? | Classification |
|-------|-------------|-------------------------------|----------------|
| `device_id` | Always | ✅ Yes (metadata only) | Safe |
| `manufacturer_id` | Always | ✅ Yes (metadata only) | Safe |
| `device_classification` | Pre-event (static) | ✅ Yes | Tier 1 |
| `device_risk_class` | Pre-event (static) | ✅ Yes | Tier 1 |
| `device_implanted` | Pre-event (static) | ✅ Yes | Tier 1 |
| `device_description` | Pre-event (static) | ✅ Yes | Tier 1 |
| `device_country` | Pre-event (static) | ✅ Yes | Tier 1 |
| `mfr_name` | Pre-event (static) | ✅ Yes | Tier 1 |
| `mfr_parent_company` | Pre-event (static) | ✅ Yes | Tier 1 |
| `type` | At event initiation | ✅ Yes (known at initiation) | Tier 2 |
| `country` (event) | At event initiation | ✅ Yes | Tier 2 |
| Historical event counts | Computed from prior events | ✅ Yes (with strict `< T` guard) | Tier 2 |
| `reason` | At initiation | ⚠️ Borderline — describes the problem | **Excluded (Tier 3)** |
| `action` | Post-event | ❌ No | Prohibited |
| `action_summary` | Post-event | ❌ No | Prohibited |
| `action_classification` | Post-event **(the target)** | ❌ THE TARGET | Prohibited |
| `determined_cause` | Post-investigation | ❌ No | Prohibited |
| `status` | Post-event lifecycle | ❌ No | Prohibited |
| `date_terminated` | Post-event | ❌ No | Prohibited |
