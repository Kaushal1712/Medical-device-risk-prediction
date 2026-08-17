# Stage 9 — GenAI Copilot

**Project:** Medical Device Failure Risk Prediction System
**Stage:** 9 — GenAI Copilot
**Date:** 2026-08-17
**Status:** ✅ Complete
**Stack:** FastAPI (`POST /copilot`) + optional OpenAI / Google Gemini + deterministic fallback

---

## Overview

The GenAI copilot is a **natural-language explainer of trusted structured context**. It is
not the source of predictions or facts — those come entirely from the real ML pipeline,
SHAP explainer, and rule-based recommendation engine.

This follows the grounding contract defined in **Section 8 of the master prompt** exactly.

---

## Grounding Contract

```
User question → POST /copilot
             → retrieve real structured context for the device:
                 - device info            (GET /devices/{id})
                 - historical event summary (feature row)
                 - real risk score/level  (serving snapshot table)
                 - real top SHAP factors  (GET /explanation/{id})
                 - real maintenance recommendation (GET /recommendation/{id})
             → build a context block containing ONLY those real values
             → system prompt instructs the LLM to:
                 - answer ONLY using the provided context
                 - explicitly distinguish:
                     "observed historical fact" |
                     "model prediction"         |
                     "decision-support recommendation"
                 - say "not available in the data" for anything outside context
                 - never state a model prediction as a confirmed fact
                 - never invent event history, dates, or maintenance records
             → LLM response (or deterministic fallback) returned to the user
             → response always includes context_used for transparency
```

The context assembly is a **single-context-block call** — not a multi-hop RAG system.

---

## Implementation

### Files

| File | Role |
|---|---|
| [`backend/routes/copilot.py`](../backend/routes/copilot.py) | Route handler, context assembly, provider dispatch |
| [`backend/schemas.py`](../backend/schemas.py) | `CopilotRequest`, `CopilotResponse`, `CopilotContext` |
| [`src/config.py`](../src/config.py) | `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL_NAME` env reads |
| [`tests/api/test_copilot.py`](../tests/api/test_copilot.py) | 28-test pytest suite |

### System Prompt

The system prompt (hardcoded in `backend/routes/copilot.py`) enforces the grounding contract:

```
You are a medical device maintenance decision-support assistant.
You have been given structured context about a specific device.
Your job is to answer the user's question ONLY using the provided context below.

Rules:
- Clearly distinguish "observed historical fact" from "model prediction" from
  "decision-support recommendation".
- Say "not available in the data" when asked about something outside the provided context.
- Never state a model prediction as a confirmed fact.
- Never invent event history, dates, or maintenance records.
- Do not speculate about patient outcomes.
- Be concise and clear. Format your answer in plain English for a biomedical engineer.

Disclaimer: This system is a decision-support prototype. Recommendations do not replace
qualified maintenance, biomedical engineering, regulatory, or clinical judgment.
```

### Context Block

The context block passed to the LLM (and returned in `context_used`) contains:

```
=== DEVICE CONTEXT ===
Device ID:            <id>
Device Name:          <name or "Not available">
Classification:       <classification or "Not available">

=== RISK PREDICTION ===
Risk Level:           HIGH | MEDIUM | LOW | "Prediction unavailable"
Risk Score (0-100):   <score>
Calibrated Probability: <probability>
Prediction Snapshot Date: <cutoff date from serving table>
Model Version:        <model_version>

=== HISTORICAL EVENT SUMMARY ===
Total historical events:        <count>
Class I (most severe) events:   <count>
Historical recall events:       <count>

=== TOP RISK FACTORS (SHAP) ===
  - feature_name (↑ risk, SHAP=+0.XXXX)
  - ...

=== MAINTENANCE RECOMMENDATION ===
Priority: Critical | High | Medium | Low
Recommended Actions:
  - <action 1>
  - <action 2>
=== END CONTEXT ===
```

---

## Supported LLM Providers

### OpenAI

```bash
# Install SDK (optional — not required for fallback path)
pip install openai>=1.0.0
```

```bash
# .env
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL_NAME=gpt-4o-mini    # or gpt-4o, gpt-3.5-turbo
```

Implementation: `_call_openai()` in `backend/routes/copilot.py`  
Uses the official `openai` Python SDK v1+ (`OpenAI` client with `chat.completions.create`).  
Temperature = 0.2, max_tokens = 512.

### Google Gemini

```bash
# Install SDK (optional — not required for fallback path)
pip install google-generativeai>=0.7.0
```

```bash
# .env
LLM_PROVIDER=gemini
LLM_API_KEY=AIza...
LLM_MODEL_NAME=gemini-1.5-flash    # or gemini-1.5-pro, gemini-pro
```

Implementation: `_call_gemini()` in `backend/routes/copilot.py`  
Uses the `google.generativeai` SDK (`GenerativeModel.generate_content`).

### Adding a New Provider

To add another provider (e.g., Anthropic):
1. Implement `_call_<provider>(context_block, question, model, api_key) → str` in `copilot.py`
2. Add the dispatch branch in the `if provider_cfg == ...` block
3. Add the `.env.example` entry
4. Add the SDK to `requirements.txt` (optional/commented section)
5. Add tests to `tests/api/test_copilot.py`

---

## Deterministic Fallback

If `LLM_PROVIDER` is blank, the key is missing, or the LLM call raises **any exception**,
the endpoint falls back automatically to `_deterministic_answer()` — a pure Python
string-formatting function that produces a structured paragraph from the same
`CopilotContext` without any external API call.

**The fallback path:**
- Never makes a network request
- Is fully deterministic (same input → same output)
- Contains the same healthcare disclaimer as the LLM path
- Is marked with `llm_used=False` and `provider="fallback"` in the response

This guarantees the demo works regardless of API key configuration.

---

## API Contract

### Request

```
POST /copilot
Content-Type: application/json

{
  "device_id": "80508",
  "question":  "Why is this device flagged as high risk?"
}
```

Constraints:
- `device_id`: min_length=1 (required)
- `question`: min_length=1, max_length=2000 (required)

### Response

```json
{
  "device_id": "80508",
  "question":  "Why is this device flagged as high risk?",
  "answer":    "Based on the structured data for device '80508' (VITROS ECI...):\n\n...",
  "context_used": {
    "device_id":               "80508",
    "device_name":             "VITROS ECI IMMUNODIAGNOSTIC ANALYZER - CLASS II",
    "device_classification":   "Clinical Chemistry Devices",
    "risk_level":              "HIGH",
    "risk_score":              100.0,
    "calibrated_probability":  1.0,
    "maintenance_priority":    "Critical",
    "recommended_actions":     ["Prioritize for immediate preventive inspection.", "..."],
    "top_risk_factors":        ["device_country_USA (↑ risk, SHAP=+0.1034)", "..."],
    "hist_device_event_count": 0.0,
    "hist_device_class_i_count": 0.0,
    "hist_device_recall_count": 0.0,
    "serving_event_date":      "2002-01-24 00:00:00",
    "model_version":           "random_forest_20260816_145309"
  },
  "llm_used": false,
  "provider": "fallback"
}
```

---

## LLM SDK Installation

The `openai` and `google-generativeai` SDKs are **optional** — they are not listed as
required dependencies in `requirements.txt` because the fallback path (which is the
default and CI/test path) does not need them.

Install the SDK for your chosen provider when you have an API key and want live LLM
responses:

```bash
# OpenAI
pip install openai>=1.0.0

# Google Gemini
pip install google-generativeai>=0.7.0
```

See the commented section in `requirements.txt` for the pinned versions.

---

## Testing

```bash
# Run all copilot tests only
source venv/bin/activate
python -m pytest tests/api/test_copilot.py -vv

# Run full suite (must stay at 0 failures)
python -m pytest -q
```

Tests are in `tests/api/test_copilot.py` and organised into five classes:

| Class | What it tests |
|---|---|
| `TestCopilotSchema` | Response envelope shape and field presence |
| `TestCopilotFallback` | Deterministic fallback path (`llm_used=False`, `provider="fallback"`) |
| `TestCopilotGrounding` | `context_used` contains real values from serving table / SHAP |
| `TestCopilotEdgeCases` | Unscored devices, unknown devices — graceful, no crash, no fabrication |
| `TestCopilotValidation` | Pydantic 422 on bad inputs (empty question, missing fields, over-length) |
| `TestCopilotConsistency` | Idempotency, cross-device differentiation, cross-endpoint consistency |

All 28 tests pass in CI without any LLM key configured.

---

## Limitations

1. **No conversation history.** Each `/copilot` request is stateless — a follow-up question
   has no memory of the previous exchange. This is intentional (Section 8: single-context-block
   call, not a multi-hop RAG system).

2. **LLM SDKs must be installed manually.** They are not in the core `requirements.txt`
   because they add significant install size and are not needed for the demo's default
   (fallback) path.

3. **Rate limits / API costs.** When a live LLM key is configured, every Streamlit page
   refresh that triggers the copilot will make a real API call. The `post_copilot` function
   in `frontend/utils/api_client.py` is intentionally **not cached** to ensure fresh responses.

4. **No Anthropic support.** Claude is not implemented in this version. The architecture
   pattern (`_call_<provider>`) makes it straightforward to add.
