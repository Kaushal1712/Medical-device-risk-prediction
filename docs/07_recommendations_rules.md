# Stage 7 — Maintenance Decision Rules

## Overview

The Maintenance Decision Engine (`src/recommendations/engine.py`) is **purely rule-based** — no additional ML model is used. It combines the model-derived **risk level** (LOW / MEDIUM / HIGH) from the Stage 6 serving table with a **device criticality proxy** derived from real data columns.

---

## Criticality Proxy Rationale

> **No explicit device criticality field exists in the dataset.**

The dataset (`devices.csv` → `merged.parquet`) does not contain a standalone "device criticality" column. The closest available proxy is `device_risk_class`, which reflects the FDA recall classification of the device:

| `device_risk_class` value | FDA meaning | Criticality tier used |
|---|---|---|
| `1` | Most severe — Class I recall (immediate hazard) | **HIGH** |
| `HDE` | Humanitarian Device Exemption (high-stakes patients) | **HIGH** |
| `2` | Less severe — Class II recall | **MEDIUM** |
| `Not Classified` | Not yet classified | **MEDIUM** |
| `Unclassified` | Not yet classified | **MEDIUM** |
| `3` | Least severe — Class III recall (unlikely to cause harm) | **LOW** |
| Missing | Field absent from data | Fall back to `hist_device_class_i_count` |

**Fallback when `device_risk_class` is missing:**
- If `hist_device_class_i_count > 0` → **HIGH** criticality (device has had Class I-level events historically)
- Otherwise → **MEDIUM** (conservative default)

---

## Rule Table

| Risk Level | Criticality Tier | Maintenance Priority | Recommended Actions |
|---|---|---|---|
| HIGH | HIGH | **Critical** | Remove from service pending review; emergency preventive inspection; escalate to biomedical engineering and risk officer; review all Class I event records |
| HIGH | MEDIUM | **Critical** | Immediate preventive inspection; escalate within 24 hours; review failure-related events |
| HIGH | LOW | **High** | Schedule inspection this week; review event records; monitor closely |
| MEDIUM | HIGH | **High** | Inspection within 7 days; review Class I records; enhanced monitoring |
| MEDIUM | MEDIUM | **Medium** | Inspection within 30 days; review event records; standard monitoring |
| MEDIUM | LOW | **Medium** | Inspection within 30 days; standard monitoring |
| LOW | HIGH | **Medium** | Monitor closely; review safety history; ensure preventive maintenance is up to date |
| LOW | MEDIUM | **Low** | Continue routine monitoring and standard maintenance schedule |
| LOW | LOW | **Low** | Continue routine monitoring and standard maintenance schedule |

---

## Rule Inputs Exposed in API Response

Every `/recommendation/{id}` response includes a `rule_inputs` dict showing exactly what data drove the recommendation:

```json
{
  "risk_level": "HIGH",
  "criticality_tier": "HIGH",
  "device_risk_class": "1",
  "hist_device_event_count": 5.0,
  "hist_device_class_i_count": 3.0,
  "hist_device_recall_count": 2.0,
  "calibrated_probability": 0.9921,
  "risk_score": 99.21,
  "serving_event_date": "2015-07-14 00:00:00",
  "criticality_proxy_note": "device_risk_class used as criticality proxy. No explicit criticality field was found in the dataset."
}
```

---

## Healthcare Disclaimer

All recommendation responses include:

> *"This system is a decision-support prototype and does not replace qualified maintenance, biomedical engineering, regulatory, or clinical judgment. It is not a certified medical device and does not guarantee patient safety outcomes."*

This disclaimer must always accompany any maintenance recommendation surfaced to the user — in the API response, the dashboard, and any copilot output.

---

## Known Limitations

1. **No explicit criticality field**: `device_risk_class` is used as a proxy. A production system would incorporate biomedical engineering-validated criticality scores.
2. **Threshold-free rules**: The rule table uses only three categorical risk levels (LOW/MEDIUM/HIGH) from the ML model, not continuous probability values. A more granular rule system could use sub-bands.
3. **Static rule table**: Rules are hard-coded. A production system would make thresholds and actions configurable without code changes.
