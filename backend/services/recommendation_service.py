"""
backend/services/recommendation_service.py
==========================================
Stage 7 — Thin wrapper around src.recommendations.engine.

Resolves device context from ModelService and delegates to MaintenanceEngine.
"""

from __future__ import annotations

import logging

from backend.services.model_service import ModelService
from src.recommendations.engine import DeviceContext, MaintenanceEngine, RecommendationResult

log = logging.getLogger(__name__)

_engine = MaintenanceEngine()  # stateless — instantiate once


def get_recommendation(device_id: str, model_service: ModelService) -> RecommendationResult:
    """
    Produce a maintenance recommendation for the given device.

    Returns a RecommendationResult with available=False if the device has
    no valid serving snapshot.
    """
    risk_row = model_service.get_device_risk(device_id)
    if risk_row is None:
        return RecommendationResult(
            device_id=device_id,
            risk_level="UNKNOWN",
            criticality_tier="UNKNOWN",
            maintenance_priority="Unknown",
            available=False,
            unavailable_reason="Device has no valid serving snapshot — prediction unavailable.",
        )

    # Resolve device attributes for criticality proxy
    device_detail = model_service.get_device_detail(device_id) or {}
    feature_row = model_service.get_device_feature_row(device_id)

    # Historical event counts (from feature row if available, else 0)
    def _feat(col: str) -> float:
        if feature_row is not None and col in feature_row.index:
            val = feature_row[col]
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    ctx = DeviceContext(
        device_id=device_id,
        risk_level=risk_row.get("risk_level", "LOW"),
        risk_score=float(risk_row.get("risk_score", 0.0)),
        calibrated_probability=float(risk_row.get("calibrated_probability", 0.0)),
        device_risk_class=device_detail.get("device_risk_class"),
        hist_device_event_count=_feat("hist_device_event_count"),
        hist_device_class_i_count=_feat("hist_device_class_i_count"),
        hist_device_recall_count=_feat("hist_device_recall_count"),
        serving_event_date=(
            str(risk_row["serving_event_date"])
            if risk_row.get("serving_event_date") is not None
            else None
        ),
        model_version=risk_row.get("model_version"),
    )

    return _engine.recommend(ctx)
