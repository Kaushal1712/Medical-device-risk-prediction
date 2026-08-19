"""
backend/services/inference_service.py
=======================================
Query-driven ML inference service for the new /assess workflow.

This service translates a user-supplied query
    (device_information + problem_description + optional metadata)
into a risk prediction using the production Random Forest model,
WITHOUT using device_id as a predictive feature.

Design decisions
----------------
1. The raw RandomForestClassifier (model.pkl) is used instead of the
   calibrated model (calibrated_model.pkl).  Reason: isotonic calibration
   was fit on the training data and collapses new query vectors (with
   hist_* = 0) to probability 0.  The raw RF provides meaningful signal
   discrimination from text + categorical features.

2. Risk banding uses the same thresholds derived from training, adapted
   for raw probability:
     HIGH   : raw_prob >= 0.40
     MEDIUM : raw_prob >= 0.15
     LOW    : raw_prob <  0.15

3. The risk_score is linear in raw_prob (0–100 scale) for UI display.

4. The QueryFeatureBuilder is loaded from the pkl artifact once and
   cached for the lifetime of the process.

5. device_id is NEVER passed to the feature builder.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PRODUCTION_DIR = _PROJECT_ROOT / "models" / "production"
_FEATURE_DIR = _PROJECT_ROOT / "data" / "features"

# Singleton holders
_query_builder = None
_raw_rf = None
_feature_col_index: Optional[dict] = None
_model_card: Optional[dict] = None

# Risk band thresholds for raw RF probability
# Canonical thresholds — single source of truth for /assess inference:
#   HIGH   : raw_prob >= 0.50  (risk_score >= 50)
#   MEDIUM : raw_prob >= 0.20  (risk_score >= 20)
#   LOW    : raw_prob <  0.20  (risk_score <  20)
# NOTE: 41.7% probability → risk_score 41.7 → MEDIUM (not HIGH)
_THRESHOLD_HIGH = 0.50    # raw_prob >= this → HIGH
_THRESHOLD_MEDIUM = 0.20  # raw_prob >= this → MEDIUM (else LOW)


@dataclass
class ScoringResult:
    """Result of a single query-based risk assessment."""
    risk_level: str          # "HIGH" | "MEDIUM" | "LOW"
    risk_score: float        # 0–100 continuous score
    raw_probability: float   # raw RF probability (Class I likelihood)
    model_version: str
    feature_vector: Optional[np.ndarray] = None  # stored for SHAP

    # Semantic note — must always accompany predictions
    target_description: str = (
        "Estimated likelihood that the reported safety event would be "
        "classified as Class I (the most serious FDA recall category)"
    )


def _load_artifacts() -> None:
    """Lazy-load the query builder and raw RF model (singleton, thread-safe enough for FastAPI)."""
    global _query_builder, _raw_rf, _feature_col_index, _model_card

    if _query_builder is not None:
        return

    log.info("InferenceService: loading artifacts...")

    # Load query feature builder (from data/features/ or models/production/)
    qfb_paths = [
        _FEATURE_DIR / "query_feature_builder.pkl",
        _PRODUCTION_DIR / "query_feature_builder.pkl",
    ]
    for qfb_path in qfb_paths:
        if qfb_path.exists():
            try:
                _query_builder = joblib.load(str(qfb_path))
                log.info("Loaded QueryFeatureBuilder from %s", qfb_path)
                break
            except Exception as exc:
                log.warning("Failed to load QueryFeatureBuilder from %s: %s", qfb_path, exc)

    if _query_builder is None:
        raise RuntimeError(
            "QueryFeatureBuilder not found or could not be loaded. "
            "Expected at data/features/query_feature_builder.pkl"
        )

    # Load raw RandomForestClassifier (not calibrated)
    raw_model_path = _PRODUCTION_DIR / "model.pkl"
    _raw_rf = joblib.load(str(raw_model_path))
    log.info("Loaded raw RF from %s", raw_model_path)

    # Build feature column index for reordering
    card_path = _PRODUCTION_DIR / "model_card.json"
    with open(card_path) as f:
        _model_card = json.load(f)

    card_cols = _model_card["feature_columns"]
    qfb_order = {col: i for i, col in enumerate(_query_builder.feature_columns)}
    _feature_col_index = {
        i_card: qfb_order[col]
        for i_card, col in enumerate(card_cols)
        if col in qfb_order
    }

    log.info(
        "InferenceService ready: %d features, model_version=%s",
        len(card_cols),
        _model_card.get("experiment_dir", "unknown"),
    )


def _reorder_to_model(query_row: np.ndarray) -> np.ndarray:
    """Reorder the query feature vector to match the model's expected column order."""
    card_cols = _model_card["feature_columns"]
    reordered = np.array(
        [[query_row[0, _feature_col_index.get(i, 0)] for i in range(len(card_cols))]],
        dtype=np.float32,
    )
    return reordered


def _band(prob: float) -> str:
    """Map raw RF probability to risk level."""
    if prob >= _THRESHOLD_HIGH:
        return "HIGH"
    if prob >= _THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "LOW"


def predict(
    device_information: str,
    problem_description: str,
    *,
    device_classification: Optional[str] = None,
    device_risk_class: Optional[str] = None,
    device_implanted: Optional[str] = None,
    device_country: Optional[str] = None,
    mfr_parent_company: Optional[str] = None,
    mfr_source: Optional[str] = None,
    country: Optional[str] = None,
    event_type: Optional[str] = None,
    hist_device_event_count: float = 0.0,
    hist_device_class_i_count: float = 0.0,
    hist_device_recall_count: float = 0.0,
    hist_mfr_event_count: float = 0.0,
    hist_mfr_class_i_count: float = 0.0,
    hist_mfr_recall_count: float = 0.0,
    hist_category_event_count: float = 0.0,
    hist_category_class_i_count: float = 0.0,
) -> ScoringResult:
    """
    Score a user query through the production Random Forest model.

    Parameters
    ----------
    device_information : str
        Free-text description of the device (name, type, characteristics).
        Combined with problem_description for text features.
    problem_description : str
        Observed problem / issue description from the user.
        This is the primary predictive text input.
    device_classification : str, optional
        FDA device classification category.
    device_risk_class : str, optional
        FDA risk class ('1', '2', '3', 'HDE', ...).
    device_implanted : str, optional
        Whether device is implanted ('YES' / 'NO').
    device_country : str, optional
        Device country of origin ('USA', 'CAN', 'AUS').
    mfr_parent_company : str, optional
        Manufacturer parent company.
    mfr_source : str, optional
        Regulatory data source.
    country : str, optional
        Event country.
    event_type : str, optional
        Event type (e.g., 'Recall').
    hist_* : float
        Historical aggregate statistics. Default 0 (unknown device/category).

    Returns
    -------
    ScoringResult with risk_level, risk_score, raw_probability, model_version.
    """
    _load_artifacts()

    # Build feature row (device_id is never passed here)
    feature_row = _query_builder.build(
        device_information=device_information,
        problem_description=problem_description,
        device_classification=device_classification,
        device_risk_class=device_risk_class,
        device_implanted=device_implanted,
        device_country=device_country,
        mfr_parent_company=mfr_parent_company,
        mfr_source=mfr_source,
        country=country,
        event_type=event_type,
        hist_device_event_count=hist_device_event_count,
        hist_device_class_i_count=hist_device_class_i_count,
        hist_device_recall_count=hist_device_recall_count,
        hist_mfr_event_count=hist_mfr_event_count,
        hist_mfr_class_i_count=hist_mfr_class_i_count,
        hist_mfr_recall_count=hist_mfr_recall_count,
        hist_category_event_count=hist_category_event_count,
        hist_category_class_i_count=hist_category_class_i_count,
    )

    # Reorder to model's expected column order
    X = _reorder_to_model(feature_row)

    # Score with raw RF (not isotonic-calibrated)
    raw_prob = float(_raw_rf.predict_proba(X)[0, 1])

    # Risk score is a 0-100 continuous display metric
    risk_score = round(raw_prob * 100, 1)

    model_version = _model_card.get("experiment_dir", "unknown")

    return ScoringResult(
        risk_level=_band(raw_prob),
        risk_score=risk_score,
        raw_probability=raw_prob,
        model_version=model_version,
        feature_vector=X,
    )


def get_model_version() -> str:
    """Return the model version string (loads artifacts if necessary)."""
    _load_artifacts()
    return _model_card.get("experiment_dir", "unknown")


def is_ready() -> bool:
    """Return True if the inference service has been initialized."""
    return _query_builder is not None and _raw_rf is not None
