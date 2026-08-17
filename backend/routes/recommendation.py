"""
backend/routes/recommendation.py
==================================
GET /recommendation/{device_id}  — maintenance priority + recommended actions.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import RecommendationResponse
from backend.services.model_service import get_model_service
from backend.services.recommendation_service import get_recommendation

router = APIRouter()


@router.get("/recommendation/{device_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def get_device_recommendation(device_id: str) -> RecommendationResponse:
    """
    Rule-based maintenance recommendation for a device.

    Returns:
    - Maintenance priority (Critical / High / Medium / Low)
    - Recommended actions
    - Rule inputs (risk level, criticality tier, historical event context)
    - The mandatory healthcare decision-support disclaimer

    Returns a structured 'unavailable' response (not a 404) if the device
    has no valid serving snapshot.

    Note: Device criticality is proxied from device_risk_class (FDA recall
    classification), as no explicit criticality field exists in the dataset.
    See docs/07_recommendations_rules.md for the full rule table.
    """
    svc = get_model_service()
    rec = get_recommendation(device_id, svc)

    return RecommendationResponse(
        device_id=rec.device_id,
        risk_level=rec.risk_level,
        criticality_tier=rec.criticality_tier,
        maintenance_priority=rec.maintenance_priority,
        recommended_actions=rec.recommended_actions,
        rule_inputs=rec.rule_inputs,
        disclaimer=rec.disclaimer,
        available=rec.available,
        unavailable_reason=rec.unavailable_reason,
    )
