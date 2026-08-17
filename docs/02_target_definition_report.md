# Target Definition Report — Stage 3

> **Stage 3 deliverable** — all statistics computed from actual processed Parquet files.
> No model trained. No target implemented. No feature engineering performed.
> Full feasibility analysis: [`docs/03_target_feasibility_report.md`](03_target_feasibility_report.md)

---

## 1. Prediction Unit

**Unit:** One safety event row from `events.parquet` where `action_classification` is not null.

Each training example is a single FDA-style safety event (recall, field safety notice, or safety alert) linked to a device and manufacturer. The model predicts the outcome of that event — specifically, whether it will be classified as Class I (the most severe tier) by the relevant regulatory authority.

---

## 2. Cutoff Definition

### Why a rolling per-device cutoff was not used

Stage 1 revealed that **96.98% of the 118,249 devices have exactly one event**. Only 3,576 devices (3.02%) have multiple events. A rolling per-device cutoff formulation (features from events before cutoff T, label from events after T) is not viable because there is insufficient recurrence data to define a meaningful outcome window.

See [`docs/03_target_feasibility_report.md §3`](03_target_feasibility_report.md) for the full per-device event-count distribution.

### Adopted cutoff: Global temporal split by event date

The dataset covers events from 1991-08-07 to 2019-06-18. A global date-based cutoff partitions the dataset:

| Split | Period | Rows | Notes |
|-------|--------|------|-------|
| **Train** | ≤ 2014-12-31 | 38,247 labeled events | Positive rate: 8.34% |
| **Validation** | 2015-01-01 – 2015-12-31 | 4,273 labeled events | Positive rate: 5.85% |
| **Test** | 2016-01-01 – 2017-12-31 | 8,918 labeled events | Positive rate: 5.52% |
| **Holdout** | 2018-01-01 – 2018-07-09 | 1,361 labeled events | Positive rate: 5.44% |

147 labeled events with no `event_date` are excluded from all splits.

**Feature window:** All static device and manufacturer attributes (pre-event by construction), plus historical event aggregates computed from all events dated **strictly before** each example's `event_date`.

**Outcome window:** N/A — the target is the severity classification of the event itself, not a future event. The severity is a regulatory determination made after the event is initiated.

---

## 3. Failure-Event Mapping Table

### Why \"will this device fail?\" is not the target

Stage 3 analysis found three structural disqualifiers:

1. **Zero devices without events** — `devices.parquet` is not a general device registry; it is a safety-event database. Every one of the 118,249 devices has at least one event. There is no natural negative class.
2. **All event types are safety/failure-related** — every event type (Recall, Field Safety Notice, Safety Alert) represents an adverse safety outcome. There are no routine registration events.
3. **96.98% of devices have exactly one event** — a repeat-event prediction formulation would have 1:32 class imbalance with no defensible negative class.

### Adopted target: Event severity classification

Given that a safety event has been initiated, predict whether it will be classified as **Class I** (the most severe tier):

| `action_classification` value | `is_class_i` label | FDA meaning |
|-------------------------------|-------------------|-------------|
| `"Class I"` | **1 (positive)** | Reasonable probability of serious adverse health consequences or death |
| `"Class II"` | **0 (negative)** | May cause temporary or reversible health consequences |
| `"Class III"` | **0 (negative)** | Not likely to cause adverse health consequences |
| `null` / unlabeled | **Excluded** | From countries without this classification scheme (72,019 events) |
| `"Unclassified Correction"` | **Excluded** | 3 events |
| `"Voluntary recall"` | **Excluded** | 1 event |

Severity labels exist only for events from 4 countries:

| Country | Class I | Class II | Class III | Total |
|---------|---------|----------|-----------|-------|
| USA | 2,483 | 31,282 | 2,058 | 35,823 |
| CAN | 632 | 8,216 | 4,705 | 13,553 |
| AUS | 895 | 2,336 | 327 | 3,558 |
| SLV | 12 | 0 | 0 | 12 |

---

## 4. Class Balance

**Binary formulation: Class I vs Class II + Class III**

| Class | Count | Percentage |
|-------|-------|------------|
| Positive (Class I) | 4,022 | **7.6%** |
| Negative (Class II + Class III) | 48,924 | **92.4%** |
| **Total labeled** | **52,946** | 100% |
| **Class ratio** | **1 : 12.2** | — |

The 1:12.2 imbalance is within the manageable range for standard class-weighting techniques. No SMOTE or oversampling was used; `class_weight="balanced"` was applied at training time.

---

## 5. Temporal Coverage and Split Strategy

### Why a temporal split (not random shuffle)

The data has a clear time dimension (events from 1991–2019 with consistent volume from 2003–2017). Randomly shuffling rows would allow the model to train on 2016 events and validate on 2014 events — the opposite of real deployment, where the model is always applied to future events.

**Implementation:** Pure date-based partitioning on `event_date`. No random splitting.

| Split | Date boundary | Rows | Positive | Positive rate |
|-------|---------------|------|----------|---------------|
| Train | ≤ 2014-12-31 | 38,247 | 3,191 | 8.34% |
| Validation | 2015-01-01 – 2015-12-31 | 4,273 | 250 | 5.85% |
| Test | 2016-01-01 – 2017-12-31 | 8,918 | 492 | 5.52% |
| Holdout (optional robustness) | 2018 | 1,361 | 74 | 5.44% |

The decline in positive rate from train (8.34%) to test (5.52%) reflects genuine temporal drift and was expected and disclosed.

---

## 6. Serving-Snapshot Policy (Stage 3f)

Because 96.98% of devices have exactly one event, the serving-snapshot policy is straightforward: for each device, the **latest event** (by `event_date`) for which the model has a valid score is designated as the device's production prediction snapshot.

**Implementation:**
- The serving table (`artifacts/risk/device_risk_snapshot.parquet`) contains one row per device, selected as the latest-dated labeled event from the feature dataset.
- The backend reads from this pre-computed table at request time; it never re-selects or re-scores rows dynamically.
- The selected event date (`serving_event_date`) is stored alongside the prediction and returned in the API response.

**Devices with no valid snapshot:** Devices that have no labeled event (i.e., `action_classification` is null for all their events, which covers 67,908 devices) are absent from the serving table. The API surfaces this as `"prediction_available": false` with a non-empty `unavailable_reason`. No fabricated score is returned.

**Production counts:**
- Devices with a valid serving snapshot: **50,341**
- Devices without a serving snapshot (unscored): **67,908**

---

## 7. Leakage-Prevention Statement

All six leakage conditions required by Stage 3c are satisfied:

1. **Target not in features** — `is_class_i` and `action_classification` are absent from all feature columns. Verified by automated tests (`TestTargetExclusion`, `TestProhibitedFields`).

2. **No post-event fields in features** — The following columns are blocked by the `PROHIBITED_FEATURES` frozenset in `src/features/build_features.py`: `action`, `action_summary`, `determined_cause`, `status`, `date_terminated`, `action_level`, `reason`, `target_audience`, and `id`. None appear in any feature split.

3. **Historical features use strict `event_date < T`** — All historical aggregate features (`hist_device_event_count`, `hist_mfr_event_count`, `hist_category_class_i_count`, etc.) are computed using events dated **strictly before** the current event's date. Same-day events are excluded via `transform('min')`.

4. **Encoding fitted on training data only** — Categorical vocabularies and frequency maps (`mfr_parent_company` frequency encoding) are derived from the training split only. Unseen categories in validation/test are encoded as `__UNKNOWN__`.

5. **No future data in historical counts** — Automated test `TestNoFutureLeakage` samples 200 training events and verifies that `hist_mfr_event_count` ≤ number of actual prior events (confirmed by recomputation from `merged.parquet`).

6. **Cutoff-truncated recomputation (Stage 10, mandatory)** — For a deterministic 30-event sample, `hist_device_event_count`, `hist_device_class_i_count`, and `hist_device_recall_count` are recomputed from scratch using only events with `event_date < sample_event_date`, and the results are asserted to exactly equal the cached values in `train.parquet`. **All 3 recomputation checks pass.** This is an exact-equality assertion, not an upper-bound check.

Full leakage documentation: [`docs/03_leakage_prevention.md`](03_leakage_prevention.md)

---

## 8. Target Validation Checklist (Stage 3c)

| Check | Result |
|-------|--------|
| Both classes have non-trivial examples | ✅ 4,022 positive (7.6%), 48,924 negative (92.4%) |
| Target not derivable from retained features | ✅ `action_classification` / `is_class_i` excluded from all feature splits |
| No post-cutoff information in features | ✅ Enforced by `PROHIBITED_FEATURES` + strict `event_date < T` |
| Temporal split respects chronological order | ✅ Train ≤ 2014, Val = 2015, Test = 2016–2017 |
| Cutoff-truncated recomputation passes | ✅ Exact equality for 30-event sample (Stage 10) |
| Serving-snapshot policy defined | ✅ Latest valid event per device, pre-computed in serving table |

---

## 9. Known Limitations

1. **This is NOT literal failure prediction.** The model predicts severity of already-initiated safety events, not future device failure before it occurs. The business framing reflects \"safety event severity prediction.\"

2. **Geographic bias.** Severity labels exist only from 4 countries (USA, Canada, Australia, El Salvador). The model will not generalize to other regulatory systems.

3. **Selection bias.** Only devices that have experienced at least one safety event appear in the dataset. The model makes no statement about devices that have never had an event.

4. **Temporal positive-rate drift.** Positive rate drops from 8.34% (train) to 5.52% (test). This is natural temporal drift and is documented and expected.

5. **Feature sparsity.** Device classification, risk_class, and implanted status are missing for approximately 70% of events. The model handles this via explicit `__MISSING__` encoding.
