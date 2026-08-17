"""
backend/routes/feature_importance.py
======================================
GET /feature-importance  — global feature importance ranked list.

Returns the pre-computed feature importance from the production model artifact
(models/production/feature_importance.json), loaded once at startup via
ModelService.  No per-request computation.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from backend.schemas import FeatureImportanceItem, FeatureImportanceResponse
from backend.services.model_service import get_model_service

log = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/feature-importance",
    response_model=FeatureImportanceResponse,
    tags=["Explainability"],
)
def get_feature_importance() -> FeatureImportanceResponse:
    """
    Global feature importance ranked list.

    Returns all features sorted by importance (descending), with their
    pre-computed importance score and 1-indexed rank.  Values are sourced
    from models/production/feature_importance.json — the artifact written
    during Stage 5 model training.

    Returns an empty features list (count=0) if no importance artifact was
    found at startup, rather than raising a 500 error.
    """
    svc = get_model_service()
    raw: list[dict] = svc.feature_importance  # already sorted importance desc

    features = [
        FeatureImportanceItem(
            feature=str(item["feature"]),
            importance=round(float(item["importance"]), 6),
            rank=i + 1,
        )
        for i, item in enumerate(raw)
    ]

    log.debug("feature-importance: returning %d features", len(features))

    return FeatureImportanceResponse(
        model_version=svc.model_version,
        count=len(features),
        features=features,
    )
