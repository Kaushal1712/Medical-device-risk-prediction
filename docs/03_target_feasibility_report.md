# Target Feasibility & Definition Report

> **Stage 3 — Analysis Only**
>
> All statistics computed from actual processed Parquet files.
> No model trained. No target implemented. No feature engineering performed.

---

## 1. Executive Summary

The original business problem is **"Predicting medical equipment/device failure before it occurs."** This report investigates whether the supplied dataset genuinely supports this formulation and evaluates four candidate prediction targets.

### Critical Structural Findings

1. **There are ZERO devices without events.** Every device in `devices.parquet` (118,249) has at least one safety event in `events.parquet`. The devices table is **not a registry of all marketed devices** — it is a table of devices that have already experienced safety events. This eliminates the possibility of a simple "will this device ever fail?" prediction with naturally occurring negative examples.

2. **All events are safety/failure-related.** Every event type (Recall, Field Safety Notice, Safety Alert, and combinations) represents an adverse safety outcome. There are no routine registration, compliance, or listing events.

3. **96.98% of devices have exactly 1 event.** Only 3,576 devices (3.02%) have multiple events. This severely constrains any repeat-event prediction formulation.

4. **Severity labels exist for only 42.4% of events**, and only from 4 countries (USA, CAN, AUS, SLV). Despite this, severity prediction is the most scientifically defensible candidate.

### Recommendation

**Candidate B — Event Severity Classification** is recommended as the primary target. It is the only candidate that provides genuine predictive value without requiring fabricated negative examples, avoids temporal leakage, and has sufficient labeled data (52,950 events) for a hackathon-quality model.

However, this must be honestly framed: it predicts **the severity class of a safety event that has already been initiated**, not literal device failure before it occurs. See Section 13 for the full target definition and Section 14 for required disclosures.

---

## 2. Dataset Evidence

### Processed Data Verification

| File | Rows | Columns |
|------|------|---------|
| `devices.parquet` | 118,249 | 15 |
| `events.parquet` | 124,969 | 33 |
| `manufacturers.parquet` | 31,827 | 10 |
| `merged.parquet` | 124,969 | 56 |

### Key Column Statistics

| Column | Non-Null | Coverage | Notes |
|--------|---------|---------|-------|
| `event_date` | 116,514 | 93.2% | Coalesced from 4 raw date columns |
| `event_date_available` | 116,514 TRUE / 8,455 FALSE | — | Explicit flag |
| `action_classification` | 52,950 | 42.4% | Severity labels: Class I/II/III |
| `type` | 124,968 | ~100% | All safety/failure event types |
| `reason` | 65,670 | 52.5% | Recall reason text |
| `determined_cause` | 35,818 | 28.7% | Post-investigation root cause |
| `device_classification` | 35,601 | 30.1% | FDA device category (16 classes) |
| `device_risk_class` | 32,948 | 27.9% | FDA risk class (1/2/3) |

---

## 3. Device Event History Analysis

### Device-Event Relationship

| Metric | Value |
|--------|-------|
| Total unique devices in `devices.parquet` | 118,249 |
| Total unique `device_id`s in `events.parquet` | 118,249 |
| **Devices with zero events** | **0** |
| Devices with exactly 1 event | 114,673 (96.98%) |
| Devices with 2 events | 2,639 (2.23%) |
| Devices with 3 events | 474 (0.40%) |
| Devices with 4 events | 173 (0.15%) |
| Devices with 5–10 events | 224 |
| Devices with 11–50 events | 63 |
| Devices with 51+ events | 3 |
| Maximum events per device | 116 |
| Mean events per device | 1.057 |
| Median events per device | 1 |
| P95 events per device | 1 |
| P99 events per device | 2 |

### Critical Finding

**There are zero devices without events.** `devices.parquet` is not a general device registry — it exclusively contains devices that have had at least one safety event. This means:

- We cannot construct a naturally occurring "no failure" (negative) class for future-event prediction
- Any negative examples would have to be fabricated or artificially defined
- The dataset is fundamentally a **safety-event database**, not a device-lifecycle database

---

## 4. Candidate A — Future Safety Event

### Formulation

> "Given information available at time T, will this device have a safety-related event after T within a defined future window?"

### Feasibility Assessment

| Criterion | Finding | Verdict |
|-----------|---------|---------|
| **Positive examples** | 118,249 devices with events | Exists, but ALL devices are positive |
| **Negative examples** | 0 devices without events | **Does not exist** |
| **Temporal ordering** | 93.2% of events have dates | Partially feasible |
| **Train/test split** | Could split by time | Theoretically possible |
| **Future-failure prediction** | Every device already has an event | **Invalid** |

### Why This Target is Invalid

1. **No negative class exists.** Every device in the dataset has had a safety event. There is no population of "healthy" devices to predict against.

2. **The dataset is selection-biased.** Devices appear in the database *because* they had a safety event. This is survivorship bias in reverse — we only see devices that failed.

3. **Fabricating negatives is not scientifically defensible.** We could artificially create negative examples by:
   - Choosing an arbitrary cutoff date and calling devices with events only before the cutoff "negative for the future window" — but this doesn't represent genuine non-failure; it represents censoring.
   - The 96.98% of single-event devices would all be classified the same way, providing no discriminative signal.

4. **Even for multi-event devices (3.02%):** 28.6% of multi-event devices have all events on the same date (span = 0 days), making temporal ordering impossible for them.

**Verdict: INVALID for genuine future-failure prediction.**

---

## 5. Candidate B — Event Severity

### Formulation

> "Given that a safety event has been initiated for a device, predict the severity class (Class I, Class II, or Class III) of that event."

### Target Variable: `action_classification`

| Class | Count | % of Labeled | % of Total | FDA Meaning |
|-------|-------|-------------|-----------|------------|
| **Class I** | 4,022 | 7.6% | 3.2% | Most severe — reasonable probability of serious health consequences or death |
| **Class II** | 41,834 | 79.0% | 33.5% | Moderate — may cause temporary or reversible health consequences |
| **Class III** | 7,090 | 13.4% | 5.7% | Least severe — not likely to cause adverse health consequences |
| Other | 4 | 0.0% | 0.0% | Unclassified Correction (3), Voluntary recall (1) |
| **Unlabeled** | 72,019 | — | 57.6% | From countries that don't use this classification |

### Label Source Analysis

Severity labels come exclusively from 4 countries:

| Country | Class I | Class II | Class III | Total |
|---------|---------|----------|-----------|-------|
| USA | 2,483 | 31,282 | 2,058 | 35,823 |
| CAN | 632 | 8,216 | 4,705 | 13,553 |
| AUS | 895 | 2,336 | 327 | 3,558 |
| SLV | 12 | 0 | 0 | 12 |

The 72,019 unlabeled events are from countries (primarily European) that use different regulatory frameworks and don't assign US-style recall severity classes. These are **structurally missing**, not randomly missing.

### Feature Availability for Severity Prediction

Features that are **available before severity is determined:**

| Feature | Coverage (of labeled) | Leakage Risk |
|---------|----------------------|-------------|
| `device_classification` | 67.7% | None — static device attribute |
| `device_risk_class` | 66.3% | None — static device attribute |
| `device_implanted` | 66.3% | None — static device attribute |
| `device_description` | 91.5% | None — static device text |
| `device_country` | 100.0% | None — static |
| `mfr_name` | 100.0% | None — static |
| `mfr_parent_company` | 77.0% | None — static |
| `type` (event type) | 100.0% | **Low** — event type is known at initiation |
| `reason` (recall reason) | 98.7% | **Borderline** — see leakage audit |
| `country` | 100.0% | **Low** — but highly correlated with label availability |

### Temporal Feasibility

Labeled events with dates: 52,803 (99.7% of labeled events have dates)

| Year | Labeled Events |
|------|---------------|
| 2003 | 1,794 |
| 2004 | 2,144 |
| 2005 | 2,144 |
| 2006 | 2,110 |
| 2007 | 2,407 |
| 2008 | 3,103 |
| 2009 | 3,427 |
| 2010 | 3,804 |
| 2011 | 3,350 |
| 2012 | 3,364 |
| 2013 | 4,035 |
| 2014 | 4,170 |
| 2015 | 4,273 |
| 2016 | 4,460 |
| 2017 | 4,462 |
| 2018 | 1,361 |

Consistent volume from 2003–2017 (~2,000–4,500 per year). Temporal train/val/test split is feasible.

### Sample Size Assessment

| Formulation | Positive | Negative | Ratio | Usable N | Verdict |
|------------|---------|---------|-------|----------|---------|
| Binary: Class I vs rest | 4,022 | 48,928 | 1:12.2 | 52,950 | **Viable** — class imbalance manageable |
| Binary: High severity (I+III) vs II | 11,112 | 41,834 | 1:3.8 | 52,946 | **Strong** — moderate imbalance |
| Multi-class: I vs II vs III | 4,022 / 41,834 / 7,090 | — | — | 52,946 | **Viable** — majority class dominates |

**Verdict: STRONG candidate — sufficient labeled data, temporal split possible, scientifically defensible.**

---

## 6. Candidate C — Repeat/Recurrent Event

### Formulation

> "After a device's first safety event, will it experience another safety event?"

### Feasibility Assessment

| Metric | Value |
|--------|-------|
| Total devices | 118,249 |
| Positive (has repeat event) | 3,576 (3.02%) |
| Negative (single event only) | 114,673 (96.98%) |
| **Class ratio** | **1:32.1** |

### Inter-Event Gap Analysis (Multi-Event Devices)

| Metric | Value |
|--------|-------|
| Total gap observations | 5,834 |
| Mean gap | 145 days |
| Median gap | 3 days |
| P25 gap | 0 days |
| P75 gap | 140 days |
| Gap = 0 (same day) | 2,829 (48.5%) |
| Gap 1–30 days | 616 |
| Gap 31–365 days | 1,686 |
| Gap > 365 days | 703 |

### Problems

1. **Extreme class imbalance** — 1:32 ratio. Only 3.02% of devices have repeat events.

2. **48.5% of inter-event gaps are 0 days** — nearly half of "repeat" events occur on the same day as the first event. These are likely the same safety action recorded as multiple events (e.g., multi-country regulatory filings for the same recall), not genuinely separate failure incidents.

3. **Negative class is suspicious** — the 114,673 "single-event" devices may simply have had only one recall recorded in the database, not a genuine absence of future failure. The database covers 1991–2019; we have no follow-up data beyond 2019.

4. **Censoring bias** — devices with a first event in 2018 have at most 1 year of follow-up, while devices with a first event in 2005 have 14 years. The negative class is contaminated by right-censoring.

**Verdict: WEAK — extreme imbalance, questionable negative class, censoring bias, many same-day "repeats."**

---

## 7. Candidate D — Manufacturer/Device Risk Scoring

### Formulation

> "Score manufacturers or devices by their safety-event risk profile."

### Data Available

| Metric | Value |
|--------|-------|
| Total manufacturers with events | 31,716 |
| Manufacturers with ≥5 events | 4,199 (13.2%) |
| Events from ≥5-event manufacturers | 85,288 (68.2%) |
| Mean events per manufacturer (≥5 events) | 20.3 |
| Median | 9 |
| Max | 1,560 |

### Assessment

This is **not a prediction problem** — it is **descriptive analytics / risk ranking**. The distinction matters:

| Approach | What It Does | Is It Prediction? |
|----------|-------------|-------------------|
| Count events per manufacturer | Summarizes historical safety record | No — descriptive |
| Compute % Class I events | Measures historical severity | No — descriptive |
| Rank manufacturers by event rate | Orders by historical risk | No — ranking |
| Predict future event count | Forecasts future safety events | Yes — but requires out-of-sample data |

For a genuine prediction, we would need to predict a manufacturer's future safety-event rate using only historical data — which loops back to Candidate A's fundamental problem (no negative examples, selection bias).

However, **risk scoring is valuable even without ML prediction** and can be implemented as a structured analytics pipeline with SHAP-like feature importance for interpretability.

**Verdict: MODERATE as risk scoring (not ML prediction), valuable for the dashboard and business context.**

---

## 8. Leakage Audit

### Field Classification

| Field | Known When? | Safe for Severity Prediction? | Safe for Future-Event Prediction? |
|-------|-----------|------------------------------|----------------------------------|
| `device_id` | Always | ✅ Yes | ✅ Yes |
| `manufacturer_id` | Always | ✅ Yes | ✅ Yes |
| `device_classification` | Pre-event (static) | ✅ Yes | ✅ Yes |
| `device_risk_class` | Pre-event (static) | ✅ Yes | ✅ Yes |
| `device_implanted` | Pre-event (static) | ✅ Yes | ✅ Yes |
| `device_description` | Pre-event (static) | ✅ Yes | ✅ Yes |
| `device_country` | Pre-event (static) | ✅ Yes | ✅ Yes |
| `mfr_name` | Pre-event (static) | ✅ Yes | ✅ Yes |
| `mfr_parent_company` | Pre-event (static) | ✅ Yes | ✅ Yes |
| `type` | At event initiation | ⚠️ Borderline | ❌ No — describes the event |
| `reason` | At event initiation | ⚠️ Borderline | ❌ No — describes the event |
| `action` | Post-event | ❌ No — response to event | ❌ No |
| `action_summary` | Post-event | ❌ No — response to event | ❌ No |
| `action_classification` | Post-event (the target) | ❌ THE TARGET | ❌ No |
| `determined_cause` | Post-investigation | ❌ No — investigation result | ❌ No |
| `status` | Post-event lifecycle | ❌ No — recall closure status | ❌ No |
| `date_terminated` | Post-event | ❌ No — closure date | ❌ No |
| `date_posted` | At/after posting | ⚠️ Borderline — timing known | ⚠️ Borderline |
| `event_date` | At event initiation | ✅ Yes — temporal ordering only | ✅ Yes |

### Leakage Summary by Candidate

| Candidate | High-Risk Leaky Fields | Borderline Fields | Safe Fields Available | Overall Leakage Risk |
|-----------|----------------------|-------------------|----------------------|---------------------|
| **A. Future Event** | type, reason, action, determined_cause, all event fields | — | Device/mfr attributes only | **High** (most features describe already-occurred events) |
| **B. Severity** | action, action_summary, determined_cause, status | type, reason | Device/mfr attributes, country, type (borderline) | **Moderate** (manageable with careful feature selection) |
| **C. Repeat Event** | Same as A for future events | — | Device/mfr attributes + first-event info | **High** |
| **D. Risk Scoring** | Not applicable (descriptive) | — | All historical data | **Low** (no prediction boundary) |

### Detailed Leakage Assessment for Candidate B (Severity)

**`reason` text — Borderline leaky:**
- The recall reason is the problem description submitted when the recall is initiated
- It is written *before* the regulatory severity classification is assigned
- However, it describes the nature and potential impact of the problem, which directly influences the severity determination
- Using `reason` text would be predicting severity from the problem description — this is **defensible** if clearly disclosed
- Mean reason length varies by severity: Class I = 302 chars, Class II = 211 chars, Class III = 220 chars

**`type` (event type) — Low leakage risk:**
- Whether an event is a "Recall" vs "Field Safety Notice" vs "Safety Alert" is known at initiation
- This is a legitimate predictor of severity (recalls tend to be more severe)

---

## 9. Temporal Feasibility

### Date Coverage

| Metric | Value |
|--------|-------|
| Dated events | 116,514 (93.2%) |
| Date range | 1991-08-07 to 2019-06-18 |
| Usable span | ~28 years |
| Bulk data period | 2003–2018 |

### Events by Year

| Year | Events | Devices (first event) |
|------|--------|----------------------|
| ≤2002 | 2,824 | 2,399 |
| 2003 | 1,983 | 1,780 |
| 2004 | 2,326 | 2,119 |
| 2005 | 2,868 | 2,782 |
| 2006 | 3,154 | 2,995 |
| 2007 | 3,551 | 3,379 |
| 2008 | 4,753 | 4,437 |
| 2009 | 5,235 | 4,916 |
| 2010 | 5,956 | 5,644 |
| 2011 | 6,434 | 6,278 |
| 2012 | 7,997 | 7,790 |
| 2013 | 10,353 | 9,929 |
| 2014 | 10,682 | 10,285 |
| 2015 | 11,917 | 11,450 |
| 2016 | 13,844 | 13,206 |
| 2017 | 14,053 | 13,257 |
| 2018 | 7,894 | 7,172 |
| 2019 | 461 | 420 |

### Proposed Temporal Split (for Candidate B)

| Split | Period | Labeled Events (est.) |
|-------|--------|----------------------|
| Train | ≤ 2014 | ~28,000 |
| Validation | 2015 | ~4,273 |
| Test | 2016–2018 | ~10,283 |

This split provides sufficient volume in all partitions and respects temporal ordering.

**Note:** 2018 data is incomplete (1,361 labeled events vs 4,000+ in prior years), and 2019 is too sparse (461 events total). The test set should use 2016–2017, with 2018 as an optional robustness check.

---

## 10. Class Balance / Sample Size

### Candidate B — Severity (Recommended)

**Binary formulation: High Severity (Class I) vs Lower Severity (Class II + III):**

| Class | Count | Percentage |
|-------|-------|-----------|
| Positive (Class I) | 4,022 | 7.6% |
| Negative (Class II + III) | 48,924 | 92.4% |
| **Total labeled** | **52,946** | **100%** |
| **Class ratio** | **1:12.2** | — |

**Alternative binary: High Severity (Class I + III) vs Lower (Class II):**

| Class | Count | Percentage |
|-------|-------|-----------|
| High severity (I + III) | 11,112 | 21.0% |
| Lower severity (II) | 41,834 | 79.0% |
| **Total** | **52,946** | **100%** |
| **Class ratio** | **1:3.8** | — |

**Multi-class: I vs II vs III:**

| Class | Count | Percentage |
|-------|-------|-----------|
| Class I | 4,022 | 7.6% |
| Class II | 41,834 | 79.0% |
| Class III | 7,090 | 13.4% |

### Sample Size Verdict

52,946 labeled events is **more than sufficient** for a hackathon-quality model. The 1:12.2 class imbalance for binary Class I prediction is manageable with standard techniques (SMOTE, class weights, stratified sampling).

### Other Candidates

| Candidate | Positive | Negative | Ratio | Usable N | Sufficient? |
|-----------|---------|---------|-------|----------|------------|
| A. Future Event | 118,249 (all) | 0 | ∞:0 | 0 | **No** — no negatives |
| B. Severity (Class I vs rest) | 4,022 | 48,924 | 1:12.2 | 52,946 | **Yes** |
| C. Repeat Event | 3,576 | 114,673 | 1:32.1 | 118,249 | **Marginal** — extreme imbalance |
| D. Risk Scoring | N/A | N/A | N/A | 31,716 mfrs | **N/A** — not prediction |

---

## 11. Candidate Comparison Matrix

| Candidate | Scientific Validity | Temporal Feasibility | Leakage Risk | Sample Size | Business Alignment | Hackathon Value | Recommendation |
|-----------|---------------------|----------------------|--------------|-------------|---------------------|-----------------|----------------|
| **A. Future Safety Event** | **Invalid** — no negative examples exist | Moderate — dates available | High — event fields are post-event | Invalid — no negatives | Strong (matches business question) | Invalid | **Reject** |
| **B. Event Severity** | **Strong** — well-defined classes, sufficient data | **Strong** — temporal split feasible | **Moderate** — manageable with care | **Strong** — 52,946 labeled | **Moderate** — related but reframed | **Strong** — clean ML problem | **✅ Recommend** |
| **C. Repeat Event** | **Weak** — extreme imbalance, censoring, same-day "repeats" | Moderate — but 48.5% same-day gaps | High — first-event info leaks | Weak — 1:32 ratio | Moderate | Weak | **Reject** |
| **D. Risk Scoring** | **Moderate** — valid as analytics, not prediction | N/A — descriptive | Low | Moderate — 31,716 mfrs | Strong — directly useful | Moderate — complements B | **Secondary** |

---

## 12. Recommended Target

### Primary Target: Event Severity Classification (Candidate B)

**What exactly are we predicting?**
Given that a medical device safety event (recall, field safety notice, or safety alert) has been initiated, predict whether it will be classified as **Class I** (most severe — reasonable probability of serious adverse health consequences or death) versus **Class II/III** (lower severity).

**What is one training example?**
A single row from `events.parquet` where `action_classification` is not null. Features are the device attributes (classification, risk class, implanted status, country), manufacturer attributes (name, parent company), and the event type. The label is binary: 1 if `action_classification == "Class I"`, 0 otherwise.

**What is the prediction cutoff?**
Temporal: train on events ≤ 2014, validate on 2015, test on 2016–2017.

**What information is allowed as input?**
- Device attributes: classification, risk_class, implanted, description, country, name, code
- Manufacturer attributes: name, parent_company, source
- Event type (Recall, FSN, Safety Alert, etc.)
- Event country
- Event date (for temporal ordering only, not as a feature)
- Historical counts: number of prior events for the same device, manufacturer, or device category (computed from events before the current event's date)
- Optional (with disclosure): `reason` text — the problem description is written at event initiation and could provide predictive signal, but it directly influences severity determination

**What is the target label?**
Binary: `is_class_i = 1` if `action_classification == "Class I"`, else `0`.

**What constitutes a positive?**
Class I recall/safety event — the most severe FDA classification.

**What constitutes a negative?**
Class II or Class III recall/safety event — lower severity.

**What future window is used?**
None — this is not a future-prediction problem. It predicts the severity of an event that has already been initiated but not yet fully classified by regulators.

**Why is this target scientifically defensible?**
- Clear, well-defined target variable with regulatory meaning
- Sufficient labeled data (52,946 events) from authoritative sources (FDA, Health Canada, TGA Australia)
- Temporal split is feasible with consistent data volume across years
- Class imbalance (1:12.2) is within manageable range
- Features (device attributes, manufacturer profile) are genuinely available before severity is determined
- No fabricated labels or artificially constructed negative examples

**Why does it align with the original business problem?**
While it does not predict "failure before it occurs" (which is impossible with this dataset), it addresses a closely related and highly valuable question: **"When a safety event occurs, how severe will it be?"** This enables:
- Prioritization of regulatory response resources
- Early warning triage for recalls likely to be Class I
- Risk-aware monitoring of device categories and manufacturers
- Proactive alerting for high-severity patterns

**What leakage risks remain?**
- `reason` text, if used, describes the problem that determines severity — this is borderline
- `type` (event type) is known at initiation and is a legitimate feature, but may correlate with severity
- Geographic bias: labels come from 4 countries only (USA 67.7%, CAN 25.6%, AUS 6.7%, SLV 0.02%)

**What limitations must be disclosed to hackathon judges?**
See Section 14.

---

## 13. Target Definition

```
Target Name:     is_class_i (binary)
Target Column:   action_classification
Positive Class:  "Class I" → 1
Negative Class:  "Class II" or "Class III" → 0
Excluded:        Events with null action_classification (72,019)
                 Events with "Unclassified Correction" (3) or "Voluntary recall" (1)
Labeled Total:   52,946
Positive Count:  4,022 (7.6%)
Negative Count:  48,924 (92.4%)

Alternative multi-class formulation:
  Target: severity_class ∈ {"Class I", "Class II", "Class III"}
  Labeled Total: 52,946
```

### Temporal Split

```
Train:       event_date ≤ 2014-12-31    (~28,000 labeled events)
Validation:  2015-01-01 to 2015-12-31   (~4,273 labeled events)
Test:        2016-01-01 to 2017-12-31   (~8,922 labeled events)
Held-out:    2018 (optional robustness)  (~1,361 labeled events)
```

### Feature Tiers

| Tier | Features | Leakage Status |
|------|---------|---------------|
| **Tier 1 (Safe)** | Device classification, risk_class, implanted, country, code | No leakage |
| **Tier 1 (Safe)** | Manufacturer name, parent_company, source country | No leakage |
| **Tier 2 (Low Risk)** | Event type (Recall/FSN/Safety Alert) | Known at initiation |
| **Tier 2 (Low Risk)** | Historical event counts (device, manufacturer, category) | Pre-event temporal aggregation |
| **Tier 3 (Borderline)** | Reason text (NLP features) | Describes the problem; borderline |
| **Tier 4 (Prohibited)** | action, action_summary, determined_cause, status, date_terminated | Post-event fields |

---

## 14. Known Limitations

1. **This is NOT literal failure prediction.** The model predicts severity of already-initiated safety events, not device failure before it occurs. The business framing must be adjusted: "predicting the severity of medical device safety events" rather than "predicting device failure."

2. **Geographic bias.** Severity labels exist only for events from USA (67.7%), Canada (25.6%), Australia (6.7%), and El Salvador (0.02%). The model will not generalize to events from other regulatory systems.

3. **Selection bias.** Only devices with safety events are in the dataset. The model says nothing about devices that have never had an event.

4. **Temporal coverage.** Data spans 1991–2018 with bulk in 2003–2017. The model reflects historical regulatory patterns that may not represent future trends.

5. **Feature sparsity.** Key device attributes (classification, risk_class, implanted) are only available for ~30% of devices. The model will have to handle significant missingness.

6. **Label source.** `action_classification` is a regulatory determination, not an objective clinical severity measure. Different regulatory agencies may apply different standards.

7. **The `reason` text dilemma.** If used, `reason` text provides strong predictive signal but describes the very problem whose severity we're predicting. This is defensible but must be disclosed.

---

## 15. Recommended Next Steps

1. **Approve the recommended target** (Candidate B — Event Severity Classification) or request modifications.

2. **Stage 4 — Feature Engineering:** Build the feature matrix using Tier 1 and Tier 2 features. Encode categoricals, compute historical aggregates with temporal guards, and optionally extract NLP features from `reason` text (with disclosure).

3. **Stage 5 — Model Training:** Train XGBoost and/or Random Forest classifiers with class-weight balancing. Use the temporal split defined above.

4. **Complementary:** Implement Candidate D (Risk Scoring) as a non-ML analytics module for the dashboard, providing manufacturer and device-level safety profiles.

5. **Framing adjustment:** Update all project documentation and the business narrative to reflect "safety event severity prediction" rather than "device failure prediction." Ensure the healthcare disclaimer acknowledges this distinction.
