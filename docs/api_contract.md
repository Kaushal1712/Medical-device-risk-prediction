# API Contract — Medical Device Risk Prediction

> **⚠️ Healthcare Disclaimer:** This API is a **decision-support prototype** and does not replace qualified maintenance, biomedical engineering, regulatory, or clinical judgment. It is not a certified medical device and does not guarantee patient safety outcomes.

Auto-generated OpenAPI docs are also available at `http://localhost:8000/docs` and `http://localhost:8000/redoc` when the backend is running.

---

## Base URL

```
http://localhost:8000
```

All endpoints accept and return `application/json`. Authentication is not required (prototype environment).

---

## Endpoints

### `GET /health`

Returns service health status including model version, data manifest hash, and the healthcare disclaimer.

**Response 200:**
```json
{
  "status": "ok",
  "model_version": "random_forest_20260816_145309",
  "data_manifest_hash": "<md5-hex-32-chars>",
  "disclaimer": "This system is a decision-support prototype..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Always `"ok"` when service is up |
| `model_version` | `string` | Full versioned model name |
| `data_manifest_hash` | `string` | MD5 of the pipeline manifest |
| `disclaimer` | `string` | Healthcare/regulatory disclaimer |

---

### `GET /risk-summary`

Returns aggregate risk statistics across all 50,341 scored devices.

**Response 200:**
```json
{
  "total_scored": 50341,
  "risk_levels": { "HIGH": 12345, "MEDIUM": 21000, "LOW": 16996 },
  "positive_rate": 0.0834,
  "model_version": "random_forest_20260816_145309",
  "category_breakdown": [...],
  "manufacturer_breakdown": [...]
}
```

---

### `GET /devices`

Returns a paginated list of devices with risk scores.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | `integer` | `1` | Page number (1-indexed) |
| `page_size` | `integer` | `20` | Items per page (max 100) |
| `risk_level` | `string` | `null` | Filter: `HIGH`, `MEDIUM`, or `LOW` |
| `manufacturer_id` | `string` | `null` | Filter by manufacturer ID |
| `search` | `string` | `null` | Substring search on device name/ID |

**Response 200:**
```json
{
  "total": 50341,
  "page": 1,
  "page_size": 20,
  "devices": [
    {
      "device_id": "80508",
      "risk_score": 100,
      "risk_level": "HIGH",
      "calibrated_probability": 0.9998,
      "model_version": "random_forest_20260816_145309"
    }
  ]
}
```

---

### `GET /devices/{device_id}`

Returns full risk detail for a single device.

**Response 200 (device scored):**
```json
{
  "device_id": "80508",
  "risk_score": 100,
  "risk_level": "HIGH",
  "calibrated_probability": 0.9998,
  "serving_event_date": "2017-08-15",
  "model_version": "random_forest_20260816_145309",
  "prediction_available": true,
  "note": "Risk score is based on the most recent FDA event record for this device."
}
```

**Response 200 (device not scored):**
```json
{
  "device_id": "1",
  "prediction_available": false,
  "unavailable_reason": "Device not in risk scoring snapshot.",
  "risk_score": null,
  "risk_level": null
}
```

**Response 404:** Device ID not found in any dataset.

---

### `POST /predict`

Returns the pre-materialized risk prediction for a device (served from the static snapshot; does not re-run inference).

**Request Body:**
```json
{ "device_id": "80508" }
```

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `device_id` | `string` | Yes | Non-empty string |

**Response 200 (available):**
```json
{
  "device_id": "80508",
  "prediction_available": true,
  "risk_score": 100,
  "risk_level": "HIGH",
  "calibrated_probability": 0.9998,
  "serving_event_date": "2017-08-15",
  "model_version": "random_forest_20260816_145309",
  "note": "Risk score is based on the most recent FDA event record for this device."
}
```

**Response 200 (unavailable — device not in snapshot):**
```json
{
  "device_id": "DEVICE_DOES_NOT_EXIST_XYZ_999",
  "prediction_available": false,
  "unavailable_reason": "Device not in risk scoring snapshot.",
  "risk_score": null,
  "risk_level": null
}
```

**Response 422:** Validation error — empty `device_id`.

> **Design note:** `/predict` always returns HTTP 200. A missing or unscored device is signalled by
> `prediction_available: false` and a descriptive `unavailable_reason`, not a 4xx/5xx error.

---

### `GET /explanation/{device_id}`

Returns SHAP-based local feature contributions and global feature importance.

**Response 200 (available):**
```json
{
  "device_id": "80508",
  "available": true,
  "model_version": "random_forest_20260816_145309",
  "top_positive": [
    {"feature": "hist_device_event_count", "contribution": 0.42},
    {"feature": "hist_device_class_i_count", "contribution": 0.31}
  ],
  "top_negative": [
    {"feature": "mfr_country_us", "contribution": -0.08}
  ],
  "global_importance": [
    {"feature": "hist_device_event_count", "importance": 0.18}
  ]
}
```

**Response 200 (unavailable):**
```json
{
  "device_id": "1",
  "available": false,
  "reason": "No feature snapshot for this device."
}
```

---

### `GET /recommendation/{device_id}`

Returns the rule-based maintenance priority recommendation.

**Response 200:**
```json
{
  "device_id": "80508",
  "maintenance_priority": "Critical",
  "recommended_action": "Immediate inspection and removal from service pending clinical review.",
  "risk_level": "HIGH",
  "risk_score": 100,
  "rule_inputs": {
    "risk_level": "HIGH",
    "device_criticality": "high",
    "hist_class_i_count": 5,
    "hist_recall_count": 3
  },
  "disclaimer": "This is a decision-support prototype. All recommendations require qualified clinical and biomedical engineering review."
}
```

**Maintenance Priority Values:**

| Priority | Risk Level | Criticality |
|----------|-----------|-------------|
| `Critical` | HIGH | any |
| `High` | MEDIUM | high |
| `Medium` | MEDIUM | medium/low |
| `Low` | LOW | any |

See [`docs/07_recommendations_rules.md`](07_recommendations_rules.md) for the full rule table.

---

### `POST /copilot`

Answers a natural-language question about a device using only real, structured context from the ML pipeline. The LLM is a natural-language explainer — it never generates predictions or fabricates data.

**Request Body:**
```json
{
  "device_id": "80508",
  "question": "What is the risk level and why?"
}
```

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `device_id` | `string` | Yes | Non-empty string |
| `question` | `string` | Yes | Non-empty string |

**Response 200:**
```json
{
  "device_id": "80508",
  "answer": "Device 80508 has a HIGH risk level (risk score: 100/100)...",
  "context_used": {
    "risk_level": "HIGH",
    "risk_score": 100,
    "calibrated_probability": 0.9998,
    "maintenance_priority": "Critical",
    "top_risk_factors": ["hist_device_event_count", "hist_device_class_i_count"],
    "model_version": "random_forest_20260816_145309"
  },
  "llm_used": true,
  "fallback": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `string` | Natural-language response grounded in `context_used` |
| `context_used` | `object` | The structured context passed to the LLM |
| `llm_used` | `boolean` | True if an external LLM was called |
| `fallback` | `boolean` | True if a template response was used (LLM unavailable) |

**Response 422:** Validation error — empty `device_id` or `question`.

**Fallback:** If no LLM is configured or the call fails, a deterministic template response is returned from the same structured context. Never an error.

See [`docs/09_copilot.md`](09_copilot.md) for the full GenAI grounding contract.

---

## Error Schema

For 422 validation errors:
```json
{
  "detail": [
    {
      "loc": ["body", "device_id"],
      "msg": "String should have at least 1 character",
      "type": "string_too_short"
    }
  ]
}
```

---

## Security & Healthcare Notes

- Healthcare disclaimer is included in `/health` and `/recommendation` responses.
- Input validation via FastAPI/Pydantic on every endpoint.
- No secrets in source; LLM key configured via `.env` (see `.env.example`).
- Every metric, score, and explanation is computed from real pipeline artifacts — nothing is hardcoded or fabricated.
- For interactive API exploration: `http://localhost:8000/docs` (Swagger UI).
