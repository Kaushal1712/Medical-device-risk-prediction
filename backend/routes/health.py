"""
backend/routes/health.py
========================
GET /health
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from backend.schemas import HealthResponse
from backend.services.model_service import get_model_service
from src.config import HEALTHCARE_DISCLAIMER

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL_CARD = _PROJECT_ROOT / "models" / "production" / "model_card.json"


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """
    System health check.

    Returns model version, data manifest hash, training timestamp, and the
    mandatory healthcare disclaimer.
    """
    svc = get_model_service()
    card = svc.model_card

    trained_at = card.get("experiment_dir", "unknown")

    return HealthResponse(
        status="ok",
        model_version=svc.model_version,
        data_manifest_hash=svc.manifest_hash,
        trained_at=trained_at,
        disclaimer=HEALTHCARE_DISCLAIMER,
    )
