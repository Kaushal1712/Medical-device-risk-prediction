"""
src/explainability/explainer.py
================================
Stage 7 — Explainability Engine.

Provides global feature importance (from pre-computed artifact) and local
per-device SHAP explanations using shap.TreeExplainer.

Public API
----------
  DeviceExplainer(production_dir, feature_dir, cache_dir)
      .global_importance()           -> list[dict]  (feature, importance, rank)
      .explain_device(feature_row)   -> ExplanationResult
      .explain_device_cached(device_id, feature_row) -> ExplanationResult

  ExplanationResult  — structured output for one device

Notes
-----
- Uses shap.TreeExplainer (fast, exact for tree models — no KernelExplainer).
- Background data: a subsample of the training set (100 rows, fixed seed).
- Cache: artifacts/explanations/{device_id}_{model_version}.json — avoids
  recomputing SHAP on every API call.
- Falls back to a structured "insufficient data" result if no feature row
  is available.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Columns that are metadata — never part of X
_METADATA_COLS: frozenset = frozenset(
    {"id", "device_id", "manufacturer_id", "event_date", "event_date_available", "is_class_i"}
)

# How many top features to return on each side
TOP_N = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    """A single feature's SHAP contribution for one prediction."""
    feature: str
    value: float          # actual feature value for this device
    shap_value: float     # SHAP contribution (positive → increases risk)
    direction: str        # "positive" | "negative"
    rank: int             # 1 = largest magnitude contribution


@dataclass
class ExplanationResult:
    """Structured SHAP explanation for a single device."""
    device_id: str
    model_version: str
    available: bool
    unavailable_reason: str = ""
    top_positive: list[FeatureContribution] = field(default_factory=list)
    top_negative: list[FeatureContribution] = field(default_factory=list)
    base_value: float = 0.0       # SHAP expected value (mean prediction)
    predicted_value: float = 0.0  # sum(shap values) + base_value

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "model_version": self.model_version,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "base_value": self.base_value,
            "predicted_value": self.predicted_value,
            "top_positive": [
                {
                    "feature": c.feature,
                    "value": float(c.value) if not np.isnan(float(c.value)) else None,
                    "shap_value": round(c.shap_value, 6),
                    "direction": c.direction,
                    "rank": c.rank,
                }
                for c in self.top_positive
            ],
            "top_negative": [
                {
                    "feature": c.feature,
                    "value": float(c.value) if not np.isnan(float(c.value)) else None,
                    "shap_value": round(c.shap_value, 6),
                    "direction": c.direction,
                    "rank": c.rank,
                }
                for c in self.top_negative
            ],
        }


# ---------------------------------------------------------------------------
# Main explainer class
# ---------------------------------------------------------------------------

class DeviceExplainer:
    """
    Loads the calibrated model and computes SHAP explanations.

    Parameters
    ----------
    production_dir : Path
        Directory containing calibrated_model.pkl, model_card.json,
        feature_importance.json.
    feature_dir : Path
        Directory containing train.parquet (used for SHAP background data).
    cache_dir : Path
        Directory for caching per-device explanation JSONs.
    background_n : int
        Number of training rows to use as SHAP background sample (default 100).
    random_seed : int
        Random seed for background sample selection.
    """

    def __init__(
        self,
        production_dir: Path,
        feature_dir: Path,
        cache_dir: Path,
        background_n: int = 100,
        random_seed: int = 42,
    ) -> None:
        self._production_dir = Path(production_dir)
        self._feature_dir = Path(feature_dir)
        self._cache_dir = Path(cache_dir)
        self._background_n = background_n
        self._random_seed = random_seed

        self._model_version: str = "unknown"
        self._feature_cols: list[str] = []
        self._explainer = None   # shap.TreeExplainer (lazy-loaded)
        self._global_importance: list[dict] | None = None

        self._load_metadata()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        """Read model_card.json and feature_importance.json."""
        card_path = self._production_dir / "model_card.json"
        if card_path.exists():
            with open(card_path) as f:
                card = json.load(f)
            self._model_version = card.get("experiment_dir", "unknown")
            self._feature_cols = card.get("feature_columns", [])
            log.info(
                "DeviceExplainer: model_version=%s  n_features=%d",
                self._model_version,
                len(self._feature_cols),
            )
        else:
            log.warning("DeviceExplainer: model_card.json not found at %s", card_path)

        fi_path = self._production_dir / "feature_importance.json"
        if fi_path.exists():
            with open(fi_path) as f:
                self._global_importance = json.load(f)
            log.info("DeviceExplainer: loaded %d global importance entries.", len(self._global_importance))
        else:
            log.warning("DeviceExplainer: feature_importance.json not found at %s", fi_path)

    def _get_explainer(self):
        """Lazy-load the shap.TreeExplainer (can take a few seconds on first call)."""
        if self._explainer is not None:
            return self._explainer

        try:
            import shap
        except ImportError as e:
            raise ImportError("shap is required for the explainability engine. Install it with: pip install shap") from e

        model_path = self._production_dir / "calibrated_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"calibrated_model.pkl not found at {model_path}")

        log.info("DeviceExplainer: loading calibrated model from %s ...", model_path)
        t0 = time.time()
        calibrated_model = joblib.load(model_path)
        log.info("DeviceExplainer: model loaded in %.1fs", time.time() - t0)

        # Extract the inner Random Forest from the CalibratedClassifierCV wrapper
        base_estimator = self._extract_base_forest(calibrated_model)

        # Build background data: subsample of training set
        background = self._build_background()

        log.info("DeviceExplainer: building shap.TreeExplainer ...")
        t0 = time.time()
        self._explainer = shap.TreeExplainer(
            base_estimator,
            data=background,
            feature_perturbation="interventional",
        )
        log.info("DeviceExplainer: TreeExplainer ready in %.1fs", time.time() - t0)
        return self._explainer

    def _extract_base_forest(self, calibrated_model):
        """
        Extract the underlying RandomForest from a CalibratedClassifierCV.

        sklearn 1.9 calibration chain:
          CalibratedClassifierCV
            → calibrated_classifiers_[0]  (_CalibratedClassifier)
              → .estimator  (FrozenEstimator)
                → .estimator  (RandomForestClassifier)  ← this is what SHAP needs
        """
        est = calibrated_model

        # Unwrap CalibratedClassifierCV
        if hasattr(est, "calibrated_classifiers_"):
            est = est.calibrated_classifiers_[0]

        # Unwrap _CalibratedClassifier
        if hasattr(est, "estimator"):
            est = est.estimator

        # Unwrap FrozenEstimator (sklearn 1.9+)
        try:
            from sklearn.frozen import FrozenEstimator
            if isinstance(est, FrozenEstimator):
                est = est.estimator
        except ImportError:
            pass  # sklearn version without FrozenEstimator — skip

        # Unwrap any remaining .estimator wrapper
        if hasattr(est, "estimator") and not hasattr(est, "estimators_"):
            est = est.estimator

        log.info(
            "DeviceExplainer: extracted base estimator type=%s",
            type(est).__name__,
        )
        return est


    def _build_background(self) -> np.ndarray:
        """Load training data and return a background sample as a numpy array."""
        train_path = self._feature_dir / "train.parquet"
        if not train_path.exists():
            raise FileNotFoundError(f"train.parquet not found at {train_path}")

        log.info("DeviceExplainer: loading training data for SHAP background ...")
        df = pd.read_parquet(train_path)
        X = self._to_feature_matrix(df)

        rng = np.random.default_rng(self._random_seed)
        n = min(self._background_n, len(X))
        idx = rng.choice(len(X), size=n, replace=False)
        background = X[idx]
        log.info("DeviceExplainer: background sample shape %s", background.shape)
        return background

    def _to_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Select only the feature columns (in the correct order from model_card.json)
        and convert to float32 numpy array.
        """
        if not self._feature_cols:
            raise RuntimeError("DeviceExplainer: feature_cols not loaded — check model_card.json")

        available = [c for c in self._feature_cols if c in df.columns]
        missing = [c for c in self._feature_cols if c not in df.columns]
        if missing:
            log.warning("DeviceExplainer: %d feature cols missing from DataFrame: %s", len(missing), missing)

        X = df[available].copy()
        # Fill any remaining NaN with 0.0 (same strategy used in training)
        X = X.fillna(0.0)
        return X.values.astype(np.float32)

    def _cache_path(self, device_id: str) -> Path:
        safe_id = str(device_id).replace("/", "_").replace("\\", "_")
        return self._cache_dir / f"{safe_id}_{self._model_version}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def global_importance(self) -> list[dict]:
        """
        Return the pre-computed global feature importance list.

        Returns
        -------
        list of dicts with keys: feature, importance, rank
        """
        if self._global_importance is None:
            return []
        # Add rank (already sorted by importance desc from Stage 5)
        return [
            {"feature": item["feature"], "importance": round(item["importance"], 6), "rank": i + 1}
            for i, item in enumerate(self._global_importance)
        ]

    def explain_device(self, device_id: str, feature_row: pd.Series | pd.DataFrame) -> ExplanationResult:
        """
        Compute SHAP values for one device's feature row.

        Parameters
        ----------
        device_id : str
        feature_row : pd.Series or single-row pd.DataFrame

        Returns
        -------
        ExplanationResult
        """
        if feature_row is None:
            return ExplanationResult(
                device_id=device_id,
                model_version=self._model_version,
                available=False,
                unavailable_reason="No feature row found for this device.",
            )

        # Normalise to a 1-row DataFrame
        if isinstance(feature_row, pd.Series):
            row_df = feature_row.to_frame().T
        else:
            row_df = feature_row.copy()

        try:
            X_row = self._to_feature_matrix(row_df)  # shape (1, n_features)
        except Exception as exc:
            log.warning("DeviceExplainer: failed to build feature matrix for %s: %s", device_id, exc)
            return ExplanationResult(
                device_id=device_id,
                model_version=self._model_version,
                available=False,
                unavailable_reason=f"Feature matrix build failed: {exc}",
            )

        try:
            explainer = self._get_explainer()
            shap_values = explainer.shap_values(X_row)
        except Exception as exc:
            log.error("DeviceExplainer: SHAP computation failed for %s: %s", device_id, exc)
            return ExplanationResult(
                device_id=device_id,
                model_version=self._model_version,
                available=False,
                unavailable_reason=f"SHAP computation failed: {exc}",
            )

        # Extract class-1 SHAP values for the first (only) row.
        # shap 0.52+ returns a single ndarray of shape (n_samples, n_features, n_classes)
        # for multi-output models; older shap returns list[ndarray(n_samples, n_features)].
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # New format: (n_samples, n_features, n_classes) — pick row 0, class 1
            sv = shap_values[0, :, 1]
        elif isinstance(shap_values, list):
            # Old format: list of (n_samples, n_features) per class — pick class 1, row 0
            sv = shap_values[1][0]
        else:
            # Fallback: single output, 2D (n_samples, n_features) or 1D
            sv = shap_values[0] if shap_values.ndim == 2 else shap_values

        # expected_value is an array of per-class base values for multi-output models
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
            base_value = float(ev[1])   # class 1
        else:
            base_value = float(ev) if not isinstance(ev, (list, np.ndarray)) else float(ev[0])
        predicted_value = float(base_value + np.sum(sv))

        # Available feature names (in matrix column order)
        available_feats = [c for c in self._feature_cols if c in row_df.columns]

        # Build contributions
        contributions: list[FeatureContribution] = []
        for i, (feat, sv_val) in enumerate(zip(available_feats, sv)):
            raw_val = float(row_df[feat].iloc[0]) if feat in row_df.columns else float("nan")
            contributions.append(FeatureContribution(
                feature=feat,
                value=raw_val,
                shap_value=float(sv_val),
                direction="positive" if sv_val >= 0 else "negative",
                rank=0,  # filled below
            ))

        # Sort by absolute SHAP value descending
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
        for rank, c in enumerate(contributions, start=1):
            c.rank = rank

        top_positive = [c for c in contributions if c.shap_value >= 0][:TOP_N]
        top_negative = [c for c in contributions if c.shap_value < 0][:TOP_N]

        return ExplanationResult(
            device_id=device_id,
            model_version=self._model_version,
            available=True,
            top_positive=top_positive,
            top_negative=top_negative,
            base_value=round(base_value, 6),
            predicted_value=round(predicted_value, 6),
        )

    def explain_device_cached(self, device_id: str, feature_row: pd.Series | pd.DataFrame) -> ExplanationResult:
        """
        Same as explain_device, but reads from / writes to the JSON cache.
        On cache hit, deserialises the stored result (fast — no SHAP recomputation).
        """
        cache_path = self._cache_path(device_id)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache hit
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    stored = json.load(f)
                result = self._dict_to_result(stored)
                log.debug("DeviceExplainer: cache hit for device_id=%s", device_id)
                return result
            except Exception as exc:
                log.warning(
                    "DeviceExplainer: cache read failed for %s (%s) — recomputing.",
                    device_id,
                    exc,
                )

        # Cache miss: compute
        result = self.explain_device(device_id, feature_row)

        # Write to cache (best-effort — don't fail if disk is full, etc.)
        try:
            with open(cache_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            log.debug("DeviceExplainer: cached explanation for device_id=%s", device_id)
        except Exception as exc:
            log.warning("DeviceExplainer: failed to write cache for %s: %s", device_id, exc)

        return result

    @staticmethod
    def _dict_to_result(d: dict) -> ExplanationResult:
        """Deserialise a cached explanation dict back to ExplanationResult."""
        def _fc(item: dict, direction: str) -> FeatureContribution:
            return FeatureContribution(
                feature=item["feature"],
                value=item.get("value") or float("nan"),
                shap_value=item["shap_value"],
                direction=direction,
                rank=item["rank"],
            )

        return ExplanationResult(
            device_id=d["device_id"],
            model_version=d["model_version"],
            available=d["available"],
            unavailable_reason=d.get("unavailable_reason", ""),
            top_positive=[_fc(c, "positive") for c in d.get("top_positive", [])],
            top_negative=[_fc(c, "negative") for c in d.get("top_negative", [])],
            base_value=d.get("base_value", 0.0),
            predicted_value=d.get("predicted_value", 0.0),
        )
