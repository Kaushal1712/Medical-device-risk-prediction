"""
backend/routes/assess.py
==========================
POST /assess — query-driven risk assessment workflow.

New endpoint that accepts device_information + problem_description and
returns a live ML risk prediction, historical evidence, and a disclaimer.

Preserved backward compatibility: the existing POST /predict (snapshot-based)
endpoint is NOT modified.

Key design invariants
---------------------
- device_id is OPTIONAL and is used ONLY for evidence retrieval and device
  metadata lookup.  It is NEVER passed to the ML model as a feature.
- The prediction target is described as:
  "likelihood that the reported safety event is Class I"
  NOT as future device failure prediction.
- Post-event fields (action, action_classification, determined_cause) are
  excluded from ALL model inputs AND from historical evidence returned to
  the user.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter

import src.retrieval.service as retrieval
from backend.schemas import (
    AssessHistoricalEvidence,
    AssessPrediction,
    AssessRequest,
    AssessResponse,
    HistoricalEvidenceEvent,
)
from backend.services import inference_service
from src.recommendations.engine import DeviceContext, MaintenanceEngine
from backend.schemas import ExplanationResponse, FeatureContributionItem, RecommendationResponse, PreventiveRiskResponse
import shap
import joblib
import numpy as np
from pathlib import Path

# Load raw RF and cols for query SHAP
_prod_dir = Path(__file__).resolve().parent.parent.parent / "models" / "production"
_raw_rf = None
_shap_cols = None
_explainer = None

log = logging.getLogger(__name__)
router = APIRouter()

_DISCLAIMER = (
    "This system is a decision-support prototype and does not replace "
    "qualified maintenance, biomedical engineering, regulatory, or clinical "
    "judgment. It is not a certified medical device and does not guarantee "
    "patient safety outcomes."
)

_LIMITATIONS = [
    "The model predicts the SEVERITY CLASSIFICATION of a reported safety event, "
    "not whether a device will fail in the future.",
    "Predictions without historical aggregate data are driven primarily by text "
    "features and may be less reliable.",
    "The model was trained on FDA, Health Canada, and TGA data from events "
    "up to 2018. Post-2018 device types or failure modes may not be well "
    "represented.",
    "This is a research prototype — do not use for clinical or regulatory decisions.",
]


def _to_evidence_event(raw: dict) -> HistoricalEvidenceEvent:
    """Convert a raw SQLite row dict to a HistoricalEvidenceEvent schema."""
    return HistoricalEvidenceEvent(
        event_id=str(raw.get("event_id", "")) or None,
        event_date=str(raw.get("event_date", "")) or None,
        event_type=raw.get("type"),
        reason=raw.get("reason"),
        device_name=raw.get("device_name"),
        device_classification=raw.get("device_classification"),
        mfr_name=raw.get("mfr_name"),
    )


@router.post("/assess", response_model=AssessResponse, tags=["Assessment"])
def assess_risk(req: AssessRequest) -> AssessResponse:
    """
    Query-driven risk assessment: classify the likely severity of a
    described safety event.

    **Primary inputs (required)**
    - `device_information` — free-text device description
    - `problem_description` — observed problem / safety concern

    **Optional context**
    - `device_id` — used ONLY for evidence retrieval and metadata lookup,
      never as a predictive ML feature
    - `device_classification`, `device_risk_class`, `device_implanted`,
      `country` — improve prediction quality if known

    **What the prediction means**

    The model estimates the *likelihood that the reported safety event would
    be classified as Class I* — the most serious FDA recall category (those
    most likely to cause serious harm or death).  It does NOT predict future
    device failure.

    **Output**
    - ML prediction (risk_level, risk_score, raw_probability)
    - Historical evidence (device events, similar events) — shown as context
      only, not used in the prediction
    - Limitations and disclaimer
    """
    # ── 1. ML Prediction (device_id NEVER passed here) ──────────────────────
    scoring = inference_service.predict(
        device_information=req.device_information,
        problem_description=req.problem_description,
        device_classification=req.device_classification,
        device_risk_class=req.device_risk_class,
        device_implanted=req.device_implanted,
        country=req.country,
    )

    prediction = AssessPrediction(
        risk_level=scoring.risk_level,
        risk_score=scoring.risk_score,
        raw_probability=round(scoring.raw_probability, 6),
        model_version=scoring.model_version,
    )

    # ── 2. Device metadata (optional, retrieval only) ────────────────────────
    device_info: Optional[dict] = None
    if req.device_id:
        device_info = retrieval.get_device_info(req.device_id)

    # ── 3. Historical evidence retrieval ─────────────────────────────────────
    device_events: list[HistoricalEvidenceEvent] = []
    device_facts: Optional[dict] = None

    if req.device_id:
        raw_events = retrieval.get_device_events(req.device_id, limit=5)
        device_events = [_to_evidence_event(e) for e in raw_events]
        facts = retrieval.get_historical_facts(req.device_id)
        if facts.get("total_events", 0) > 0:
            device_facts = facts

    # Similar events via FTS search on the problem description
    search_text = " ".join(
        filter(None, [req.problem_description, req.device_information])
    )
    raw_similar = retrieval.search_similar_events(
        search_text,
        limit=8,
        device_classification=req.device_classification,
    )
    similar_events = [_to_evidence_event(e) for e in raw_similar]

    historical_evidence = AssessHistoricalEvidence(
        device_events=device_events,
        similar_events=similar_events,
        device_facts=device_facts,
    )

    # ── 3b. Calculate Historical Preventive Risk ─────────────────────────────
    if device_facts and device_facts.get("total_events", 0) > 0:
        total = device_facts.get("total_events", 0)
        class_i = device_facts.get("class_i_events", 0)
        recalls = device_facts.get("recall_events", 0)
        
        if class_i >= 1 or recalls >= 1 or total >= 10:
            p_level = "HIGH"
        elif total >= 3:
            p_level = "MEDIUM"
        else:
            p_level = "LOW"
            
        p_note = f"Device profile has a history of {total} total events, {class_i} Class I events, and {recalls} recalls."
        preventive_risk = PreventiveRiskResponse(level=p_level, score_note=p_note)
    else:
        preventive_risk = PreventiveRiskResponse(level="UNKNOWN", score_note="Insufficient historical data (or no device_id provided) to calculate a preventive pattern score.")

    # ── 4. Generate query recommendation ─────────────────────────────────────
    engine = MaintenanceEngine()
    ctx = DeviceContext(
        device_id=req.device_id or "query",
        risk_level=scoring.risk_level,
        risk_score=scoring.risk_score,
        calibrated_probability=scoring.raw_probability,
        device_risk_class=req.device_risk_class,
        hist_device_class_i_count=device_facts.get("class_i_events", 0.0) if device_facts else 0.0,
        hist_device_event_count=device_facts.get("total_events", 0.0) if device_facts else 0.0,
        hist_device_recall_count=device_facts.get("recall_events", 0.0) if device_facts else 0.0,
        model_version=scoring.model_version,
    )
    rec_result = engine.recommend(ctx)
    recommendation = RecommendationResponse(**rec_result.to_dict())

    # ── 5. Generate query explanation (SHAP) ─────────────────────────────────
    global _raw_rf, _shap_cols, _explainer
    if _raw_rf is None:
        _raw_rf = joblib.load(_prod_dir / "model.pkl")
        import json
        with open(_prod_dir / "model_card.json") as f:
            _shap_cols = json.load(f)["feature_columns"]
        _explainer = shap.TreeExplainer(_raw_rf)

    X = scoring.feature_vector
    sv = _explainer.shap_values(X)
    
    if isinstance(sv, list):
        sv = sv[1][0]
    elif sv.ndim == 3:
        sv = sv[0, :, 1]
    else:
        sv = sv[0]
        
    expected = _explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(expected[1])
    else:
        expected = float(expected)
        
    contribs = []
    for col, val, s in zip(_shap_cols, X[0], sv):
        if abs(s) > 0.0:
            contribs.append({"feature": col, "value": float(val), "shap": float(s)})
            
    contribs.sort(key=lambda x: abs(x["shap"]), reverse=True)
    
    top_pos = []
    top_neg = []
    for c in contribs:
        if c["shap"] >= 0 and len(top_pos) < 5:
            top_pos.append(FeatureContributionItem(feature=c["feature"], value=c["value"], shap_value=c["shap"], direction="positive", rank=len(top_pos)+1))
        elif c["shap"] < 0 and len(top_neg) < 5:
            top_neg.append(FeatureContributionItem(feature=c["feature"], value=c["value"], shap_value=c["shap"], direction="negative", rank=len(top_neg)+1))
            
    explanation = ExplanationResponse(
        device_id=req.device_id or "query",
        model_version=scoring.model_version,
        available=True,
        base_value=expected,
        predicted_value=expected + float(np.sum(sv)),
        top_positive=top_pos,
        top_negative=top_neg,
    )

    # ── 6. Assemble response ─────────────────────────────────────────────────
    return AssessResponse(
        prediction=prediction,
        preventive_risk=preventive_risk,
        device_info=device_info,
        historical_evidence=historical_evidence,
        explanation=explanation,
        recommendation=recommendation,
        limitations=_LIMITATIONS,
        disclaimer=_DISCLAIMER,
    )
