"""
backend/routes/predict.py
==========================
POST /predict  — retrieve a device's latest valid serving snapshot risk score.

Does NOT accept ad-hoc feature payloads. Does NOT recompute probabilities.
Returns the pre-materialized serving snapshot from the Stage 6 artifact.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import PredictRequest, PredictResponse
from backend.services.model_service import get_model_service

router = APIRouter()


@router.post("/predict", response_model=PredictResponse, tags=["Risk"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Retrieve the designated latest valid serving snapshot for a device.

    Returns the pre-materialized risk score from the Stage 6 artifact
    (artifacts/risk/device_risk_snapshot.parquet).

    Returns a structured "prediction unavailable" response (not a 404) if
    the device has no valid scoreable snapshot. Returns 422 for malformed input.

    Note: Ad-hoc what-if prediction (passing raw feature values) is not
    supported in this version. See docs/ for the future enhancement note.
    """
    svc = get_model_service()
    risk_row = svc.get_device_risk(request.device_id)

    if risk_row is None:
        return PredictResponse(
            device_id=request.device_id,
            prediction_available=False,
            unavailable_reason=(
                "This device has no valid serving snapshot. Either the device ID "
                "was not found in the scored population, or insufficient historical "
                "data was available before any usable cutoff date."
            ),
        )

    return PredictResponse(
        device_id=request.device_id,
        prediction_available=True,
        raw_probability=float(risk_row.get("raw_probability", 0.0)),
        calibrated_probability=float(risk_row.get("calibrated_probability", 0.0)),
        risk_score=float(risk_row.get("risk_score", 0.0)),
        risk_level=risk_row.get("risk_level"),
        serving_event_date=str(risk_row.get("serving_event_date", "")),
        model_version=risk_row.get("model_version"),
    )
