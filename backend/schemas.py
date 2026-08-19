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


# ---------------------------------------------------------------------------
# POST /assess  (new query-driven risk assessment workflow)
# ---------------------------------------------------------------------------

class AssessRequest(BaseModel):
    """
    Request schema for the query-driven risk assessment workflow.

    device_id is OPTIONAL — it is used only for evidence retrieval and device
    metadata lookup, never as a predictive ML feature.

    device_information + problem_description are the primary predictive inputs.
    """
    device_information: str = Field(
        min_length=3,
        max_length=2000,
        description=(
            "Description of the device (e.g., 'Implanted cardiac defibrillator, "
            "model X200'). Used for text feature extraction."
        ),
    )
    problem_description: str = Field(
        min_length=10,
        max_length=4000,
        description=(
            "Description of the observed problem or safety concern "
            "(e.g., 'Repeated electrical faults and abnormal battery behavior'). "
            "This is the primary predictive text input."
        ),
    )
    # Optional context (for enriched response, not prediction)
    device_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional device ID for retrieving historical evidence and device "
            "metadata. NOT used as a predictive ML feature."
        ),
    )
    # Optional device attributes (improve prediction quality if known)
    device_classification: Optional[str] = Field(
        default=None,
        description="FDA device classification category (e.g., 'Cardiovascular Devices').",
    )
    device_risk_class: Optional[str] = Field(
        default=None,
        description="FDA device risk class ('1', '2', '3', 'HDE', 'Not Classified').",
    )
    device_implanted: Optional[str] = Field(
        default=None,
        description="Whether the device is implanted ('YES' or 'NO').",
    )
    country: Optional[str] = Field(
        default=None,
        description="Country where the event occurred ('USA', 'CAN', 'AUS').",
    )


class AssessPrediction(BaseModel):
    """ML prediction result for the /assess endpoint."""
    risk_level: str = Field(description="'HIGH', 'MEDIUM', or 'LOW'")
    risk_score: float = Field(description="0–100 continuous score (higher = higher risk)")
    raw_probability: float = Field(
        description="Raw Random Forest probability (Class I event likelihood)"
    )
    model_version: str
    target_description: str = Field(
        default=(
            "Estimated likelihood that the reported safety event would be "
            "classified as Class I (the most serious FDA recall category)"
        )
    )
    note: str = Field(
        default=(
            "This prediction uses the problem_description and device_information "
            "as primary inputs. It does NOT use device_id as a predictive feature. "
            "Predictions without historical aggregate data (hist_* = 0) are "
            "driven primarily by text and device attribute features."
        )
    )


class HistoricalEvidenceEvent(BaseModel):
    """A single historical event returned as evidence."""
    event_id: Optional[str] = None
    event_date: Optional[str] = None
    event_type: Optional[str] = None
    reason: Optional[str] = None
    device_name: Optional[str] = None
    device_classification: Optional[str] = None
    mfr_name: Optional[str] = None


class PreventiveRiskResponse(BaseModel):
    level: str = Field(description="'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'")
    score_note: str = Field(description="Explanation of the historical aggregate rule used to determine this score.")

class AssessHistoricalEvidence(BaseModel):
    """Historical evidence retrieved from the serving database."""
    device_events: list[HistoricalEvidenceEvent] = Field(
        default_factory=list,
        description="Past events for this specific device (if device_id provided).",
    )
    similar_events: list[HistoricalEvidenceEvent] = Field(
        default_factory=list,
        description="Historically similar events retrieved by full-text search.",
    )
    device_facts: Optional[dict] = Field(
        default=None,
        description="Aggregated historical facts for this device.",
    )
    retrieval_note: str = Field(
        default=(
            "Historical events are shown as context only. They were NOT used "
            "as inputs to the ML prediction."
        )
    )


class AssessResponse(BaseModel):
    """
    Full response from the query-driven /assess endpoint.

    Contains the ML prediction, retrieved evidence, and a mandatory disclaimer.
    """
    prediction: AssessPrediction
    preventive_risk: PreventiveRiskResponse
    device_info: Optional[dict] = Field(
        default=None,
        description="Device metadata (if device_id was provided).",
    )
    historical_evidence: AssessHistoricalEvidence
    explanation: Optional[ExplanationResponse] = Field(
        default=None,
        description="SHAP explanation for the query prediction."
    )
    recommendation: Optional[RecommendationResponse] = Field(
        default=None,
        description="Maintenance priority recommendation for the query."
    )
    limitations: list[str] = Field(
        default_factory=lambda: [
            "The model predicts the SEVERITY CLASSIFICATION of a reported safety "
            "event, not whether a device will fail in the future.",
            "Predictions without historical aggregate data are driven primarily "
            "by text features and may be less reliable.",
            "The model was trained on FDA, Health Canada, and TGA data from before "
            "2018. Post-2018 device types or failure modes may not be well represented.",
            "This is a research prototype — do not use for clinical or regulatory decisions.",
        ]
    )
    disclaimer: str = Field(
        default=(
            "This system is a decision-support prototype and does not replace "
            "qualified maintenance, biomedical engineering, regulatory, or clinical "
            "judgment. It is not a certified medical device and does not guarantee "
            "patient safety outcomes."
        )
    )

