"""
Central configuration for the Medical Device Failure Risk Prediction System.

All paths, thresholds, random seeds, and environment-dependent values are
defined here or read from .env. No magic numbers should be scattered
elsewhere in the codebase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present (git-ignored, never committed)
load_dotenv()

# =============================================================================
# Project root (repo root, two levels up from src/config.py)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# Data paths — configurable via .env, with sensible defaults
# =============================================================================
RAW_DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DATA_DIR = Path(os.environ.get("PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
FEATURES_DATA_DIR = Path(os.environ.get("FEATURES_DATA_DIR", PROJECT_ROOT / "data" / "features"))

# Raw CSV file paths
DEVICES_CSV = RAW_DATA_DIR / "devices.csv"
EVENTS_CSV = RAW_DATA_DIR / "events.csv"
MANUFACTURERS_CSV = RAW_DATA_DIR / "manufacturers.csv"

# Processed Parquet file paths
DEVICES_PARQUET = PROCESSED_DATA_DIR / "devices.parquet"
EVENTS_PARQUET = PROCESSED_DATA_DIR / "events.parquet"
MANUFACTURERS_PARQUET = PROCESSED_DATA_DIR / "manufacturers.parquet"
MERGED_PARQUET = PROCESSED_DATA_DIR / "merged.parquet"
MANIFEST_PATH = PROCESSED_DATA_DIR / "_manifest.json"

# Feature Parquet file paths
TRAIN_PARQUET = FEATURES_DATA_DIR / "train.parquet"
VALIDATION_PARQUET = FEATURES_DATA_DIR / "validation.parquet"
TEST_PARQUET = FEATURES_DATA_DIR / "test.parquet"

# =============================================================================
# Model paths
# =============================================================================
MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))
PRODUCTION_MODEL_DIR = Path(os.environ.get("PRODUCTION_MODEL_DIR", MODELS_DIR / "production"))
EXPERIMENT_MODEL_DIR = Path(os.environ.get("EXPERIMENT_MODEL_DIR", MODELS_DIR / "experiments"))

# =============================================================================
# Artifacts paths
# =============================================================================
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
EXPLANATIONS_DIR = ARTIFACTS_DIR / "explanations"
RISK_DIR = ARTIFACTS_DIR / "risk"

# =============================================================================
# Documentation paths
# =============================================================================
DOCS_DIR = PROJECT_ROOT / "docs"

# =============================================================================
# LLM / GenAI Copilot configuration
# =============================================================================
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "")

# =============================================================================
# Application settings
# =============================================================================
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))
APP_DEBUG = os.environ.get("APP_DEBUG", "false").lower() in ("true", "1", "yes")

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

# =============================================================================
# Pipeline version — bump this when pipeline logic changes to invalidate cache
# =============================================================================
PIPELINE_VERSION = "0.1.0"

# =============================================================================
# Healthcare disclaimer (used in README, /health endpoint, dashboard footer)
# =============================================================================
HEALTHCARE_DISCLAIMER = (
    "This system is a decision-support prototype and does not replace qualified "
    "maintenance, biomedical engineering, regulatory, or clinical judgment. "
    "It is not a certified medical device and does not guarantee patient safety outcomes."
)

# =============================================================================
# Stage 6 — Risk Scoring Engine
# =============================================================================

# Scoring version — bump when calibration or band definitions change
RISK_SCORE_VERSION = "1.0"

# Calibration method (sklearn 1.9.0: FrozenEstimator + CalibratedClassifierCV)
CALIBRATION_METHOD = "isotonic"

# Production scoring artifact paths
CALIBRATED_MODEL_PATH = PRODUCTION_MODEL_DIR / "calibrated_model.pkl"
CALIBRATION_REPORT_PATH = PRODUCTION_MODEL_DIR / "calibration_report.json"

# Serving table — one row per device, latest valid snapshot per Stage 3f policy
RISK_SNAPSHOT_PATH = RISK_DIR / "device_risk_snapshot.parquet"

# Risk band thresholds on the CALIBRATED probability scale [0, 1].
# Canonical thresholds — single source of truth for the serving/batch pipeline:
#   LOW:    risk_score <  20  (calibrated_probability <  0.20)
#   MEDIUM: risk_score >= 20 and < 50  (calibrated_probability >= 0.20 and < 0.50)
#   HIGH:   risk_score >= 50  (calibrated_probability >= 0.50)
# NOTE: These operational bands are independent of the model decision_threshold
# (used for is_class_i_predicted). The model, calibration, and raw probabilities
# are unchanged. See docs/06_risk_scoring_report.md.
RISK_THRESHOLD_HIGH: float = 0.50    # calibrated_prob >= this → HIGH
RISK_THRESHOLD_MEDIUM: float = 0.20  # calibrated_prob >= this → MEDIUM (else LOW)
