"""
src/risk/scorer.py
==================
Stage 6 — Risk Scoring Engine: Core scoring module.

Public API
----------
  RiskScorer       — loads model + calibration, scores feature rows
  ScoringResult    — structured output of a single scoring call
  probability_to_score(p)            -> float in [0, 100]
  score_to_band(p, t_medium, t_high) -> "LOW" | "MEDIUM" | "HIGH"

Run the full scoring pipeline via the companion scripts:
  python -m src.risk.calibrate           # fit calibration + derive thresholds
  python -m src.risk.build_serving_table # score all events + materialise serving table
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Columns that are metadata / target — never part of X
_METADATA_COLS: frozenset = frozenset(
    {"id", "device_id", "manufacturer_id", "event_date", "event_date_available", "is_class_i"}
)


# ---------------------------------------------------------------------------
# Pure helper functions (no model required — unit-testable in isolation)
# ---------------------------------------------------------------------------

def probability_to_score(p: float) -> float:
    """Convert a calibrated probability in [0, 1] to a 0-100 risk score.

    Formula: score = round(p * 100, 2)

    Rationale: linear mapping preserves the probability interpretation. No
    log-transform or percentile normalisation is applied — those would obscure
    the posterior meaning for downstream consumers.

    Parameters
    ----------
    p : float
        Calibrated probability in [0, 1].

    Returns
    -------
    float
        Risk score in [0.0, 100.0].

    Raises
    ------
    ValueError
        If p is NaN or outside [0, 1].
    """
    if isinstance(p, float) and math.isnan(p):
        raise ValueError(
            "probability_to_score: probability is NaN — cannot compute score."
        )
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"probability_to_score: probability {p!r} is outside [0, 1]. "
            "This should never occur with a fitted sklearn model."
        )
    return round(p * 100.0, 2)


def score_to_band(
    calibrated_prob: float,
    t_medium: float,
    t_high: float,
) -> str:
    """Map a calibrated probability to a categorical risk band.

    Boundaries are inclusive at the upper end:
      calibrated_prob >= t_high   -> "HIGH"
      calibrated_prob >= t_medium -> "MEDIUM"
      else                        -> "LOW"

    Parameters
    ----------
    calibrated_prob : float
        Calibrated class-1 probability in [0, 1].
    t_medium : float
        Lower boundary of the MEDIUM band.
    t_high : float
        Lower boundary of the HIGH band.

    Returns
    -------
    str
        "LOW", "MEDIUM", or "HIGH".

    Raises
    ------
    ValueError
        If t_medium >= t_high (invalid band configuration).
    """
    if t_medium >= t_high:
        raise ValueError(
            f"score_to_band: t_medium ({t_medium}) must be less than t_high ({t_high})."
        )
    if calibrated_prob >= t_high:
        return "HIGH"
    if calibrated_prob >= t_medium:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# ScoringResult — structured output of one scoring call
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """Result of scoring a single event/device row.

    Attributes
    ----------
    raw_probability : float
        Uncalibrated model output from predict_proba()[:, 1].
    calibrated_probability : float
        Isotonic-calibrated posterior probability.
    risk_score : float
        0-100 score derived from calibrated_probability x 100.
    risk_level : str
        "LOW", "MEDIUM", or "HIGH" per the derived band thresholds.
    is_class_i_predicted : bool
        True when raw_probability >= decision_threshold (0.8555).
    decision_threshold : float
        The raw-probability threshold sourced from model_card.json.
    model_version : str
        Timestamp string from model_card.json identifying the model run.
    warnings : list[str]
        Non-fatal issues encountered (e.g. all-NaN feature row).
    """

    raw_probability: float
    calibrated_probability: float
    risk_score: float
    risk_level: str
    is_class_i_predicted: bool
    decision_threshold: float
    model_version: str
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# RiskScorer
# ---------------------------------------------------------------------------

class RiskScorer:
    """Loads the Stage 5 production model and Stage 6 calibrated wrapper,
    and provides single-row and batch scoring methods.

    Usage
    -----
    Typical serving path (calibration already fitted and saved):

        scorer = RiskScorer()
        scorer.load_calibration()
        result = scorer.score(X_row, feature_cols)

    Calibration fitting path (run once by calibrate.py):

        scorer = RiskScorer()
        scorer.fit_calibration(X_train, y_train, save=True)
    """

    def __init__(self, model_dir=None):
        """Load the base model and model card from models/production/.

        Parameters
        ----------
        model_dir : Path or str, optional
            Override path to the production model directory.
            Defaults to models/production/ relative to the project root.

        Raises
        ------
        FileNotFoundError
            If model.pkl or model_card.json are not found.
        """
        if model_dir is None:
            _root = Path(__file__).resolve().parent.parent.parent
            model_dir = _root / "models" / "production"

        self.model_dir = Path(model_dir)
        model_path = self.model_dir / "model.pkl"
        card_path = self.model_dir / "model_card.json"

        if not model_path.exists():
            raise FileNotFoundError(f"RiskScorer: model not found at {model_path}")
        if not card_path.exists():
            raise FileNotFoundError(f"RiskScorer: model card not found at {card_path}")

        log.info("Loading base model from %s", model_path)
        self._base_model = joblib.load(model_path)
        self._card = json.loads(card_path.read_text(encoding="utf-8"))

        self.decision_threshold = float(self._card["decision_threshold"])
        self.feature_columns = self._card["feature_columns"]
        self.model_version = self._card.get(
            "trained_at", self._card.get("experiment_dir", "unknown")
        )

        # Calibrated model — populated by fit_calibration() or load_calibration()
        self._calibrated_model = None

        log.info(
            "RiskScorer ready — model=%s  features=%d  threshold=%.6f",
            self._card.get("model_name", "?"),
            len(self.feature_columns),
            self.decision_threshold,
        )

    # ------------------------------------------------------------------
    # Calibration fitting
    # ------------------------------------------------------------------

    def fit_calibration(self, X_train, y_train, save=True):
        """Fit isotonic calibration on the training split only.

        Uses sklearn 1.9.0 FrozenEstimator to wrap the already-fitted base
        model, then fits CalibratedClassifierCV with method="isotonic" and
        cv="prefit". This is leakage-safe: the base model weights are frozen
        and the isotonic regression sees only training-period data.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix (float32, shape [n_train, n_features]).
        y_train : np.ndarray
            Training labels (int, shape [n_train]).
        save : bool
            If True, saves calibrated_model.pkl to model_dir.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator

        log.info("Fitting isotonic calibration on %d training samples...", len(y_train))
        # sklearn 1.9.0: cv="prefit" is removed. With FrozenEstimator, use cv=None —
        # the base estimator weights are frozen; only the isotonic calibrator is fitted
        # on the provided X_train, y_train. This is the documented replacement.
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(self._base_model),
            method="isotonic",
            cv=None,
        )
        calibrated.fit(X_train, y_train)
        self._calibrated_model = calibrated
        log.info("Calibration fitted.")

        if save:
            out = self.model_dir / "calibrated_model.pkl"
            joblib.dump(calibrated, out)
            log.info("Saved calibrated model -> %s", out)

    def load_calibration(self):
        """Load a previously fitted calibrated_model.pkl from model_dir.

        Raises
        ------
        FileNotFoundError
            If calibrated_model.pkl is not present.
        """
        cal_path = self.model_dir / "calibrated_model.pkl"
        if not cal_path.exists():
            raise FileNotFoundError(
                f"RiskScorer: calibrated model not found at {cal_path}. "
                "Run: python -m src.risk.calibrate"
            )
        log.info("Loading calibrated model from %s", cal_path)
        self._calibrated_model = joblib.load(cal_path)
        log.info("Calibrated model loaded.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_proba(self, X):
        """Return class-1 probability from the base model."""
        return self._base_model.predict_proba(X)[:, 1]

    def _calibrated_proba(self, X):
        """Return calibrated class-1 probability. Requires fit/load_calibration."""
        if self._calibrated_model is None:
            raise RuntimeError(
                "RiskScorer: calibrated model not loaded. "
                "Call load_calibration() or fit_calibration() first."
            )
        return self._calibrated_model.predict_proba(X)[:, 1]

    def _validate_feature_array(self, X, feature_cols):
        """Raise if X has wrong number of columns."""
        expected = len(self.feature_columns)
        if X.shape[1] != expected:
            raise ValueError(
                f"RiskScorer: expected {expected} feature columns, "
                f"got {X.shape[1]}."
            )

    # ------------------------------------------------------------------
    # Public scoring interface
    # ------------------------------------------------------------------

    def score(self, X, feature_cols, t_medium=None, t_high=None):
        """Score a single row (shape [1, n_features] or [n_features]).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix with shape (1, n_features) or (n_features,).
        feature_cols : list[str]
            Column names corresponding to X's columns.
        t_medium, t_high : float, optional
            Risk band thresholds. If None, read from src.config.

        Returns
        -------
        ScoringResult
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._validate_feature_array(X, feature_cols)

        warnings = []
        if np.all(np.isnan(X.astype(float))):
            warnings.append("All feature values are NaN — score may be unreliable.")

        if t_medium is None or t_high is None:
            import src.config as cfg
            t_medium = t_medium if t_medium is not None else cfg.RISK_THRESHOLD_MEDIUM
            t_high = t_high if t_high is not None else cfg.RISK_THRESHOLD_HIGH

        raw_p = float(self._raw_proba(X)[0])
        cal_p = float(self._calibrated_proba(X)[0])

        return ScoringResult(
            raw_probability=raw_p,
            calibrated_probability=cal_p,
            risk_score=probability_to_score(cal_p),
            risk_level=score_to_band(cal_p, t_medium, t_high),
            is_class_i_predicted=raw_p >= self.decision_threshold,
            decision_threshold=self.decision_threshold,
            model_version=self.model_version,
            warnings=warnings,
        )

    def batch_score(self, df, t_medium=None, t_high=None):
        """Score all rows in a feature DataFrame.

        Drops metadata and target columns automatically. Preserves
        id, device_id, event_date for downstream joining.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame (may include metadata columns).
        t_medium, t_high : float, optional
            Risk band thresholds. If None, read from src.config.

        Returns
        -------
        pd.DataFrame
            Columns: id, device_id, event_date,
            raw_probability, calibrated_probability, risk_score,
            risk_level, is_class_i_predicted, decision_threshold,
            model_version.
        """
        if t_medium is None or t_high is None:
            import src.config as cfg
            t_medium = t_medium if t_medium is not None else cfg.RISK_THRESHOLD_MEDIUM
            t_high = t_high if t_high is not None else cfg.RISK_THRESHOLD_HIGH

        # Preserve metadata for output
        id_col = df["id"].values if "id" in df.columns else np.arange(len(df)).astype(str)
        device_id_col = df["device_id"].values if "device_id" in df.columns else None
        event_date_col = df["event_date"].values if "event_date" in df.columns else None

        # Build feature matrix — use the authoritative feature list from model card.
        # This is immune to extra columns added by the caller (e.g., _split).
        missing_cols = [c for c in self.feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"batch_score: DataFrame is missing {len(missing_cols)} model features: "
                f"{missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}"
            )
        X = df[self.feature_columns].values.astype(np.float32)
        feat_cols = self.feature_columns

        if X.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"batch_score: expected {len(self.feature_columns)} feature columns, "
                f"got {X.shape[1]}."
            )

        raw_proba = self._raw_proba(X)
        cal_proba = self._calibrated_proba(X)

        risk_scores = np.round(cal_proba * 100.0, 2)
        risk_levels = np.array([
            score_to_band(float(p), t_medium, t_high) for p in cal_proba
        ])
        is_pred = raw_proba >= self.decision_threshold

        return pd.DataFrame({
            "id": id_col,
            "device_id": device_id_col,
            "event_date": event_date_col,
            "raw_probability": raw_proba,
            "calibrated_probability": cal_proba,
            "risk_score": risk_scores,
            "risk_level": risk_levels,
            "is_class_i_predicted": is_pred,
            "decision_threshold": self.decision_threshold,
            "model_version": self.model_version,
        })
