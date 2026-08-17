"""
backend/routes/copilot.py
==========================
POST /copilot  — GenAI copilot Q&A grounded in real device context.

Grounding contract (Section 8 of master prompt):
  1. Retrieve real structured context for the device.
  2. Build a context block containing ONLY those real values.
  3. Call the LLM (if configured) with a system prompt that instructs it
     to answer ONLY using the provided context.
  4. If LLM call fails or no key is configured, return a deterministic
     template-based response using the same structured context.
  5. Always return both the answer AND the context used.

Supported LLM providers (via .env):
  LLM_PROVIDER=openai   + LLM_API_KEY=sk-...  + LLM_MODEL_NAME=gpt-4o-mini
  LLM_PROVIDER=gemini   + LLM_API_KEY=...      + LLM_MODEL_NAME=gemini-1.5-flash
  LLM_PROVIDER=         (empty → deterministic fallback, no external call)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter

from backend.schemas import CopilotContext, CopilotRequest, CopilotResponse
from backend.services.explainability_service import get_explanation
from backend.services.model_service import get_model_service
from backend.services.recommendation_service import get_recommendation
from src.config import LLM_API_KEY, LLM_MODEL_NAME, LLM_PROVIDER

log = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Grounded system prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a medical device maintenance decision-support assistant.

You have been given structured context about a specific device. Your job is to answer
the user's question ONLY using the provided context below.

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
"""


def _build_context_block(ctx: CopilotContext) -> str:
    """Format the context dict as a readable block for the LLM prompt."""
    lines = [
        "=== DEVICE CONTEXT ===",
        f"Device ID: {ctx.device_id}",
        f"Device Name: {ctx.device_name or 'Not available'}",
        f"Classification: {ctx.device_classification or 'Not available'}",
        "",
        "=== RISK PREDICTION ===",
        f"Risk Level: {ctx.risk_level or 'Prediction unavailable'}",
        f"Risk Score (0-100): {ctx.risk_score or 'N/A'}",
        f"Calibrated Probability: {ctx.calibrated_probability or 'N/A'}",
        f"Prediction Snapshot Date: {ctx.serving_event_date or 'N/A'}",
        f"Model Version: {ctx.model_version or 'N/A'}",
        "",
        "=== HISTORICAL EVENT SUMMARY ===",
        f"Total historical events: {ctx.hist_device_event_count or 0}",
        f"Class I (most severe) events: {ctx.hist_device_class_i_count or 0}",
        f"Historical recall events: {ctx.hist_device_recall_count or 0}",
        "",
        "=== TOP RISK FACTORS (SHAP) ===",
        "\n".join(f"  - {f}" for f in ctx.top_risk_factors) if ctx.top_risk_factors else "  Not available",
        "",
        "=== MAINTENANCE RECOMMENDATION ===",
        f"Priority: {ctx.maintenance_priority or 'Not available'}",
        "Recommended Actions:",
        "\n".join(f"  - {a}" for a in ctx.recommended_actions) if ctx.recommended_actions else "  Not available",
        "=== END CONTEXT ===",
    ]
    return "\n".join(lines)


def _deterministic_answer(ctx: CopilotContext, question: str) -> str:
    """
    Produce a deterministic template-based answer without any LLM call.
    Uses the same structured context — purely string formatting.
    """
    risk_str = f"{ctx.risk_level} (score: {ctx.risk_score}/100)" if ctx.risk_level else "unavailable"
    prob_str = f"{ctx.calibrated_probability:.3f}" if ctx.calibrated_probability is not None else "N/A"
    factors_str = (
        ", ".join(ctx.top_risk_factors[:3])
        if ctx.top_risk_factors
        else "not available"
    )
    actions_str = (
        "; ".join(ctx.recommended_actions[:2])
        if ctx.recommended_actions
        else "not available"
    )

    return (
        f"Based on the structured data for device '{ctx.device_id}' "
        f"({ctx.device_name or 'name not available'}):\n\n"
        f"**Risk Assessment:** The model predicts a risk level of {risk_str} "
        f"with a calibrated probability of {prob_str}. "
        f"This prediction is based on the most recent event snapshot "
        f"dated {ctx.serving_event_date or 'N/A'}.\n\n"
        f"**Historical Context:** The device has {int(ctx.hist_device_event_count or 0)} "
        f"recorded historical events, including {int(ctx.hist_device_class_i_count or 0)} "
        f"Class I (most severe) events and "
        f"{int(ctx.hist_device_recall_count or 0)} recall events.\n\n"
        f"**Key Risk Factors (model explanation):** {factors_str}.\n\n"
        f"**Maintenance Recommendation:** Priority is '{ctx.maintenance_priority or 'N/A'}'. "
        f"Suggested actions: {actions_str}.\n\n"
        f"_Disclaimer: This is a decision-support prototype. All recommendations "
        f"must be reviewed by qualified maintenance or biomedical engineering personnel._"
    )


def _call_openai(context_block: str, question: str, model: str, api_key: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT + "\n\n" + context_block},
                {"role": "user", "content": question},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("OpenAI call failed: %s", exc)
        raise


def _call_gemini(context_block: str, question: str, model: str, api_key: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(model or "gemini-1.5-flash")
        prompt = f"{_SYSTEM_PROMPT}\n\n{context_block}\n\nUser Question: {question}"
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        log.warning("Gemini call failed: %s", exc)
        raise


@router.post("/copilot", response_model=CopilotResponse, tags=["Copilot"])
def copilot(request: CopilotRequest) -> CopilotResponse:
    """
    GenAI copilot grounded in real device context.

    Assembles trusted structured context (device info, historical event
    summary, real risk score, real SHAP factors, real maintenance recommendation)
    and calls the configured LLM (or deterministic fallback if no key configured).

    Always returns both the answer and the structured context used.
    """
    svc = get_model_service()
    device_id = request.device_id
    question = request.question

    # --- Assemble structured context ---
    device_data = svc.get_device_detail(device_id) or {}
    risk_row = svc.get_device_risk(device_id)
    rec = get_recommendation(device_id, svc)
    explanation = get_explanation(device_id, svc)

    # Historical event counts from feature row
    feature_row = svc.get_device_feature_row(device_id)
    def _feat(col: str) -> Optional[float]:
        if feature_row is not None and col in feature_row.index:
            try:
                return float(feature_row[col])
            except (TypeError, ValueError):
                return None
        return None

    # Top risk factors from SHAP (feature names only, no raw values)
    top_risk_factors: list[str] = []
    if explanation.available:
        for c in (explanation.top_positive + explanation.top_negative)[:5]:
            direction = "↑" if c.direction == "positive" else "↓"
            top_risk_factors.append(f"{c.feature} ({direction} risk, SHAP={c.shap_value:+.4f})")

    ctx = CopilotContext(
        device_id=device_id,
        device_name=str(device_data.get("device_name", "")) or None,
        device_classification=str(device_data.get("device_classification", "")) or None,
        risk_level=risk_row["risk_level"] if risk_row else None,
        risk_score=float(risk_row["risk_score"]) if risk_row else None,
        calibrated_probability=float(risk_row["calibrated_probability"]) if risk_row else None,
        maintenance_priority=rec.maintenance_priority if rec.available else None,
        recommended_actions=rec.recommended_actions if rec.available else [],
        top_risk_factors=top_risk_factors,
        hist_device_event_count=_feat("hist_device_event_count"),
        hist_device_class_i_count=_feat("hist_device_class_i_count"),
        hist_device_recall_count=_feat("hist_device_recall_count"),
        serving_event_date=str(risk_row["serving_event_date"]) if risk_row else None,
        model_version=risk_row.get("model_version") if risk_row else None,
    )

    context_block = _build_context_block(ctx)
    llm_used = False
    provider = "fallback"
    answer = ""

    # --- Try LLM call ---
    provider_cfg = (LLM_PROVIDER or "").strip().lower()
    if provider_cfg and LLM_API_KEY:
        try:
            if provider_cfg == "openai":
                answer = _call_openai(context_block, question, LLM_MODEL_NAME, LLM_API_KEY)
                llm_used = True
                provider = "openai"
            elif provider_cfg == "gemini":
                answer = _call_gemini(context_block, question, LLM_MODEL_NAME, LLM_API_KEY)
                llm_used = True
                provider = "gemini"
            else:
                log.warning("Copilot: unsupported LLM_PROVIDER '%s' — using fallback.", provider_cfg)
        except Exception as exc:
            log.warning("Copilot: LLM call failed (%s) — falling back to deterministic template.", exc)

    # --- Deterministic fallback ---
    if not llm_used:
        answer = _deterministic_answer(ctx, question)
        provider = "fallback"

    return CopilotResponse(
        device_id=device_id,
        question=question,
        answer=answer,
        context_used=ctx,
        llm_used=llm_used,
        provider=provider,
    )
