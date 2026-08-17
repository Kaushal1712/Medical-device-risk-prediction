"""
backend/services/explainability_service.py
==========================================
Stage 7 — Thin wrapper around src.explainability.explainer.

Resolves the device's feature row from ModelService, delegates to
DeviceExplainer, and returns an ExplanationResult (or unavailable result).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from backend.services.model_service import ModelService
from src.explainability.explainer import DeviceExplainer, ExplanationResult

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_PRODUCTION_DIR = _PROJECT_ROOT / "models" / "production"
_FEATURE_DIR    = _PROJECT_ROOT / "data" / "features"
_CACHE_DIR      = _PROJECT_ROOT / "artifacts" / "explanations"

# Module-level singleton explainer (lazy-initialised on first call)
_explainer: Optional[DeviceExplainer] = None


def _get_explainer() -> DeviceExplainer:
    global _explainer
    if _explainer is None:
        _explainer = DeviceExplainer(
            production_dir=_PRODUCTION_DIR,
            feature_dir=_FEATURE_DIR,
            cache_dir=_CACHE_DIR,
        )
    return _explainer


def get_explanation(device_id: str, model_service: ModelService) -> ExplanationResult:
    """
    Return an ExplanationResult for the given device.

    Parameters
    ----------
    device_id : str
    model_service : ModelService

    Returns
    -------
    ExplanationResult — available=True with SHAP values, or available=False
    with an unavailable_reason if the device cannot be explained.
    """
    # Check serving snapshot exists
    risk_row = model_service.get_device_risk(device_id)
    if risk_row is None:
        return ExplanationResult(
            device_id=device_id,
            model_version="unknown",
            available=False,
            unavailable_reason="Device has no valid serving snapshot — prediction unavailable.",
        )

    # Get feature row
    feature_row = model_service.get_device_feature_row(device_id)
    if feature_row is None:
        return ExplanationResult(
            device_id=device_id,
            model_version="unknown",
            available=False,
            unavailable_reason="No feature row found for this device in the feature store.",
        )

    explainer = _get_explainer()
    return explainer.explain_device_cached(device_id, feature_row)


def get_global_importance(model_service: ModelService) -> list[dict]:
    """Return the pre-computed global feature importance list."""
    return model_service.feature_importance
