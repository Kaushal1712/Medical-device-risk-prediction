"""
backend/routes/risk_summary.py
================================
GET /risk-summary  — aggregate risk statistics from the real serving snapshot.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import (
    CategoryBreakdownItem,
    ManufacturerBreakdownItem,
    RiskLevelCount,
    RiskSummaryResponse,
)
from backend.services.model_service import get_model_service

router = APIRouter()


@router.get("/risk-summary", response_model=RiskSummaryResponse, tags=["Risk"])
def risk_summary() -> RiskSummaryResponse:
    """
    Aggregate risk statistics computed from the real scored population
    (Stage 3f serving-snapshot policy).

    Includes:
    - Total device counts (scored / unscored / per-level)
    - Risk score distribution stats
    - Top-15 breakdown by device category
    - Top-15 breakdown by manufacturer
    """
    svc = get_model_service()
    data = svc.compute_risk_summary()

    # Coerce to schema types
    risk_levels = {
        k: RiskLevelCount(count=v["count"], percent=v["percent"])
        for k, v in data["risk_levels"].items()
    }

    cat_breakdown = [
        CategoryBreakdownItem(
            category=item.get("category", item.get("manufacturer", "Unknown")),
            high=item["high"],
            medium=item["medium"],
            low=item["low"],
            total=item["total"],
        )
        for item in data["category_breakdown"]
    ]

    mfr_breakdown = [
        ManufacturerBreakdownItem(
            manufacturer=item.get("manufacturer", item.get("category", "Unknown")),
            high=item["high"],
            medium=item["medium"],
            low=item["low"],
            total=item["total"],
        )
        for item in data["manufacturer_breakdown"]
    ]

    return RiskSummaryResponse(
        total_devices_in_data=data["total_devices_in_data"],
        total_scored=data["total_scored"],
        total_unscored=data["total_unscored"],
        risk_levels=risk_levels,
        risk_score_stats=data["risk_score_stats"],
        category_breakdown=cat_breakdown,
        manufacturer_breakdown=mfr_breakdown,
    )
