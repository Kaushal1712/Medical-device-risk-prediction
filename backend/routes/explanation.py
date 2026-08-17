"""
backend/routes/explanation.py
==============================
GET /explanation/{device_id}  — real SHAP values for a device.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from backend.schemas import ExplanationResponse, FeatureContributionItem
from backend.services.explainability_service import get_explanation
from backend.services.model_service import get_model_service

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/explanation/{device_id}", response_model=ExplanationResponse, tags=["Explainability"])
def get_device_explanation(device_id: str) -> ExplanationResponse:
    """
    SHAP-backed explanation for a single device.

    Returns the top 5 positive and top 5 negative contributing features
    based on real SHAP values for the device's serving-snapshot feature row.

    Returns a structured 'insufficient data' response (not a 404) if the
    device has no valid scoreable snapshot or no feature row.

    Results are cached to artifacts/explanations/ to avoid recomputing
    SHAP values on every request.
    """
    svc = get_model_service()
    result = get_explanation(device_id, svc)

    top_positive = [
        FeatureContributionItem(
            feature=c.feature,
            value=c.value if c.value == c.value else None,  # NaN → None
            shap_value=c.shap_value,
            direction=c.direction,
            rank=c.rank,
        )
        for c in result.top_positive
    ]
    top_negative = [
        FeatureContributionItem(
            feature=c.feature,
            value=c.value if c.value == c.value else None,
            shap_value=c.shap_value,
            direction=c.direction,
            rank=c.rank,
        )
        for c in result.top_negative
    ]

    return ExplanationResponse(
        device_id=result.device_id,
        model_version=result.model_version,
        available=result.available,
        unavailable_reason=result.unavailable_reason,
        base_value=result.base_value,
        predicted_value=result.predicted_value,
        top_positive=top_positive,
        top_negative=top_negative,
    )
