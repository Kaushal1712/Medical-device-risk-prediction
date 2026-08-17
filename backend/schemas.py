"""
backend/schemas.py
==================
Stage 7 — Pydantic v2 request / response models for all 8 API endpoints.

All fields are strictly typed. No Any fields.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / reusable
# ---------------------------------------------------------------------------

class Pagination(BaseModel):
    page: int = Field(ge=1, description="1-indexed page number")
    page_size: int = Field(ge=1, le=200, description="Number of items per page")
    total_items: int
    total_pages: int


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_version: str
    data_manifest_hash: str
    trained_at: str
    disclaimer: str


# ---------------------------------------------------------------------------
# GET /devices  (list)
# ---------------------------------------------------------------------------

class DeviceListItem(BaseModel):
    device_id: str
    device_name: Optional[str] = None
    device_classification: Optional[str] = None
    device_risk_class: Optional[str] = None
    device_country: Optional[str] = None
    mfr_name: Optional[str] = None
    mfr_parent_company: Optional[str] = None
    risk_level: Optional[str] = None          # None = prediction unavailable
    risk_score: Optional[float] = None
    calibrated_probability: Optional[float] = None
    serving_event_date: Optional[str] = None  # ISO date string


class DeviceListResponse(BaseModel):
    items: list[DeviceListItem]
    pagination: Pagination


# ---------------------------------------------------------------------------
# GET /devices/{id}  (detail)
# ---------------------------------------------------------------------------

class DeviceDetail(BaseModel):
    # Core device fields from merged.parquet
    device_id: str
    device_name: Optional[str] = None
    device_description: Optional[str] = None
    device_classification: Optional[str] = None
    device_risk_class: Optional[str] = None
    device_implanted: Optional[str] = None
    device_country: Optional[str] = None
    device_number: Optional[str] = None
    device_distributed_to: Optional[str] = None
    manufacturer_id: Optional[str] = None
    mfr_name: Optional[str] = None
    mfr_parent_company: Optional[str] = None
    mfr_source: Optional[str] = None
    mfr_address: Optional[str] = None

    # Risk (from serving table or null)
    prediction_available: bool
    prediction_unavailable_reason: str = ""
    risk_level: Optional[str] = None
    risk_score: Optional[float] = None
    calibrated_probability: Optional[float] = None
    serving_event_date: Optional[str] = None
    model_version: Optional[str] = None

    # Maintenance summary (abbreviated — full detail at /recommendation/{id})
    maintenance_priority: Optional[str] = None


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    device_id: str = Field(min_length=1, description="The device ID to retrieve a risk prediction for")


class PredictResponse(BaseModel):
    device_id: str
    prediction_available: bool
    unavailable_reason: str = ""
    raw_probability: Optional[float] = None
    calibrated_probability: Optional[float] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    serving_event_date: Optional[str] = None
    model_version: Optional[str] = None
    note: str = (
        "This endpoint returns the device's designated latest valid serving snapshot "
        "per the Stage 3f serving-snapshot policy. Ad-hoc what-if prediction is not "
        "supported in this version."
    )


# ---------------------------------------------------------------------------
# GET /risk-summary
# ---------------------------------------------------------------------------

class RiskLevelCount(BaseModel):
    count: int
    percent: float


class CategoryBreakdownItem(BaseModel):
    category: str
    high: int
    medium: int
    low: int
    total: int


class ManufacturerBreakdownItem(BaseModel):
    manufacturer: str
    high: int
    medium: int
    low: int
    total: int


class RiskSummaryResponse(BaseModel):
    total_devices_in_data: int        # total unique devices in merged.parquet
    total_scored: int                 # devices with a valid serving snapshot
    total_unscored: int               # devices with no valid snapshot
    risk_levels: dict[str, RiskLevelCount]     # "HIGH", "MEDIUM", "LOW"
    risk_score_stats: dict[str, float]         # min, mean, median, max
    category_breakdown: list[CategoryBreakdownItem]
    manufacturer_breakdown: list[ManufacturerBreakdownItem]


# ---------------------------------------------------------------------------
# GET /explanation/{id}
# ---------------------------------------------------------------------------

class FeatureContributionItem(BaseModel):
    feature: str
    value: Optional[float] = None
    shap_value: float
    direction: str   # "positive" | "negative"
    rank: int


class ExplanationResponse(BaseModel):
    device_id: str
    model_version: str
    available: bool
    unavailable_reason: str = ""
    base_value: float = 0.0
    predicted_value: float = 0.0
    top_positive: list[FeatureContributionItem] = []
    top_negative: list[FeatureContributionItem] = []


# ---------------------------------------------------------------------------
# GET /recommendation/{id}
# ---------------------------------------------------------------------------

class RecommendationResponse(BaseModel):
    device_id: str
    risk_level: str
    criticality_tier: str
    maintenance_priority: str
    recommended_actions: list[str]
    rule_inputs: dict
    disclaimer: str
    available: bool
    unavailable_reason: str = ""


# ---------------------------------------------------------------------------
# POST /copilot
# ---------------------------------------------------------------------------

class CopilotRequest(BaseModel):
    device_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)


class CopilotContext(BaseModel):
    """Structured context passed to / returned from the LLM for transparency."""
    device_id: str
    device_name: Optional[str] = None
    device_classification: Optional[str] = None
    risk_level: Optional[str] = None
    risk_score: Optional[float] = None
    calibrated_probability: Optional[float] = None
    maintenance_priority: Optional[str] = None
    recommended_actions: list[str] = []
    top_risk_factors: list[str] = []
    hist_device_event_count: Optional[float] = None
    hist_device_class_i_count: Optional[float] = None
    hist_device_recall_count: Optional[float] = None
    serving_event_date: Optional[str] = None
    model_version: Optional[str] = None


class CopilotResponse(BaseModel):
    device_id: str
    question: str
    answer: str
    context_used: CopilotContext
    llm_used: bool              # True if LLM was called; False if deterministic fallback
    provider: str = ""          # e.g. "openai", "gemini", or "fallback"


# ---------------------------------------------------------------------------
# GET /feature-importance
# ---------------------------------------------------------------------------

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    rank: int


class FeatureImportanceResponse(BaseModel):
    model_version: str
    count: int
    features: list[FeatureImportanceItem]
