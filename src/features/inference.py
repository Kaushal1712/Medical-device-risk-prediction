"""
src/features/inference.py
==========================
Query-time feature construction for the medical device risk prediction workflow.

Translates a user-supplied query:
    device_information + problem_description + optional metadata
into the 86-feature numeric vector expected by the production Random Forest
model, WITHOUT using device_id as a predictive feature.

The user's 'problem_description' is the inference-time analogue of the
training-time 'reason' field.  It is projected through the same TF-IDF +
SVD pipeline that was fit on training data only.

Public API
----------
  QueryFeatureBuilder
      .build(device_information, problem_description, **kwargs)
          → np.ndarray of shape (1, 86), float32

Leakage guarantees
------------------
  - device_id is never passed into this function.
  - The text transformer is loaded from a pre-fit artifact (training data only).
  - Historical aggregate features default to 0 when not available.
  - Post-event fields (action, action_classification, determined_cause) are
    never used.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class QueryFeatureBuilder:
    """
    Converts a user query into the 86-feature vector required by the
    production model.

    Attributes set via __setstate__ (from the saved pkl):
      feature_columns : list[str]     — ordered list of model feature names
      encoding        : dict          — OHE categories, frequency maps
      text_transformer                — fitted ReportedIssueTextTransformer

    Parameters
    ----------
    feature_columns : list[str]
        The ordered feature list from model_card.json. These must exactly
        match the columns the model was trained on.
    encoding : dict
        Encoding metadata produced by the feature pipeline:
        - OHE columns: {"method": "one_hot", "categories": [...]}
        - Frequency columns: {"method": "frequency", "frequencies": {...}}
    text_transformer : ReportedIssueTextTransformer
        The fitted text pipeline (TF-IDF + SVD).
    """

    def __init__(
        self,
        feature_columns: list[str],
        encoding: dict,
        text_transformer,
    ) -> None:
        self.feature_columns = feature_columns
        self.encoding = encoding
        self.text_transformer = text_transformer

    # ------------------------------------------------------------------
    # Categorical encoding helpers (mirror training logic exactly)
    # ------------------------------------------------------------------

    def _encode_ohe(
        self, row: dict, col: str, categories: list[str], value: Optional[str]
    ) -> None:
        """Fill one-hot encoded columns for `col` into `row`."""
        clean_value = str(value) if value and str(value).lower() not in ("nan", "none", "") else "__MISSING__"
        if clean_value not in categories:
            clean_value = "__UNKNOWN__"
        for cat in categories:
            feature_name = f"{col}_{cat}"
            if feature_name in self.feature_columns:
                row[feature_name] = 1 if clean_value == cat else 0

    def _encode_freq(self, row: dict, col: str, frequencies: dict, value: Optional[str]) -> None:
        """Fill frequency-encoded column for `col` into `row`."""
        feature_name = f"{col}_freq"
        if feature_name not in self.feature_columns:
            return
        clean_value = str(value) if value and str(value).lower() not in ("nan", "none", "") else "__MISSING__"
        row[feature_name] = float(frequencies.get(clean_value, 0.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
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
        # Historical aggregates (default to 0 when not available)
        hist_device_event_count: float = 0.0,
        hist_device_class_i_count: float = 0.0,
        hist_device_recall_count: float = 0.0,
        hist_mfr_event_count: float = 0.0,
        hist_mfr_class_i_count: float = 0.0,
        hist_mfr_recall_count: float = 0.0,
        hist_category_event_count: float = 0.0,
        hist_category_class_i_count: float = 0.0,
    ) -> np.ndarray:
        """
        Build the feature vector for a query.

        Parameters
        ----------
        device_information : str
            Free-text description of the device (e.g., 'Implanted cardiac defibrillator').
            Used for text SVD features (combined with problem_description).
        problem_description : str
            Observed problem / issue description.
            This is the primary text input — the training analogue is 'reason'.
        device_classification : str, optional
            FDA device classification (e.g., 'Cardiovascular Devices').
        device_risk_class : str, optional
            Device risk class ('1', '2', '3', 'HDE', etc.)
        device_implanted : str, optional
            Implanted flag ('YES' or 'NO').
        device_country : str, optional
            Country of device ('USA', 'CAN', 'AUS').
        mfr_parent_company : str, optional
            Manufacturer parent company name.
        mfr_source : str, optional
            Regulatory source name (e.g., 'U.S. Food and Drug Administration').
        country : str, optional
            Event country ('USA', 'CAN', 'AUS').
        event_type : str, optional
            Event type ('Recall', etc.).
        hist_* : float
            Historical aggregate counts. Defaults to 0 (unknown device).

        Returns
        -------
        np.ndarray of shape (1, n_features), float32.
            Ready to pass to model.predict_proba().
        """
        row: dict = {}

        # ── Text features (problem_description + device_information) ────────
        combined_text = " ".join(
            filter(None, [problem_description.strip(), device_information.strip()])
        )
        try:
            svd_vector = self.text_transformer.transform([combined_text])[0]
            for name, val in zip(self.text_transformer.get_feature_names(), svd_vector):
                if name in self.feature_columns:
                    row[name] = float(val)
        except Exception as exc:
            log.warning("Text transform failed, zeroing SVD features: %s", exc)
            for name in self.text_transformer.get_feature_names():
                if name in self.feature_columns:
                    row[name] = 0.0

        # ── Device length proxies ────────────────────────────────────────────
        if "device_description_len" in self.feature_columns:
            row["device_description_len"] = min(len(device_information), 10000)
        if "device_name_len" in self.feature_columns:
            row["device_name_len"] = min(len(device_information.split()[0]) if device_information.strip() else 0, 1000)

        # ── Categorical features ─────────────────────────────────────────────
        enc = self.encoding

        if "device_classification" in enc:
            cats = enc["device_classification"]["categories"]
            self._encode_ohe(row, "device_classification", cats, device_classification)
            # Missing indicator
            missing_col = "device_classification_missing"
            if missing_col in self.feature_columns:
                is_missing = device_classification is None or str(device_classification).lower() in ("nan", "none", "")
                row[missing_col] = 1 if is_missing else 0

        if "device_risk_class" in enc:
            cats = enc["device_risk_class"]["categories"]
            self._encode_ohe(row, "device_risk_class", cats, device_risk_class)
            missing_col = "device_risk_class_missing"
            if missing_col in self.feature_columns:
                is_missing = device_risk_class is None or str(device_risk_class).lower() in ("nan", "none", "")
                row[missing_col] = 1 if is_missing else 0

        if "device_implanted" in enc:
            cats = enc["device_implanted"]["categories"]
            self._encode_ohe(row, "device_implanted", cats, device_implanted)
            missing_col = "device_implanted_missing"
            if missing_col in self.feature_columns:
                is_missing = device_implanted is None or str(device_implanted).lower() in ("nan", "none", "")
                row[missing_col] = 1 if is_missing else 0

        if "device_country" in enc:
            cats = enc["device_country"]["categories"]
            self._encode_ohe(row, "device_country", cats, device_country)

        if "mfr_parent_company" in enc:
            freqs = enc["mfr_parent_company"].get("frequencies", {})
            self._encode_freq(row, "mfr_parent_company", freqs, mfr_parent_company)

        if "mfr_source" in enc:
            cats = enc["mfr_source"]["categories"]
            self._encode_ohe(row, "mfr_source", cats, mfr_source)

        if "country" in enc:
            cats = enc["country"]["categories"]
            self._encode_ohe(row, "country", cats, country)

        if "type" in enc:
            cats = enc["type"]["categories"]
            self._encode_ohe(row, "type", cats, event_type)

        # ── Historical aggregate features ────────────────────────────────────
        hist_map = {
            "hist_device_event_count": hist_device_event_count,
            "hist_device_class_i_count": hist_device_class_i_count,
            "hist_device_recall_count": hist_device_recall_count,
            "hist_mfr_event_count": hist_mfr_event_count,
            "hist_mfr_class_i_count": hist_mfr_class_i_count,
            "hist_mfr_recall_count": hist_mfr_recall_count,
            "hist_category_event_count": hist_category_event_count,
            "hist_category_class_i_count": hist_category_class_i_count,
        }
        for col, val in hist_map.items():
            if col in self.feature_columns:
                row[col] = float(val)

        # ── Derived rate features ────────────────────────────────────────────
        if "hist_mfr_severity_rate" in self.feature_columns:
            if hist_mfr_event_count > 0:
                row["hist_mfr_severity_rate"] = hist_mfr_class_i_count / hist_mfr_event_count
                row["hist_mfr_severity_rate_available"] = 1
            else:
                row["hist_mfr_severity_rate"] = 0.0
                row["hist_mfr_severity_rate_available"] = 0

        if "hist_category_severity_rate" in self.feature_columns:
            if hist_category_event_count > 0:
                row["hist_category_severity_rate"] = hist_category_class_i_count / hist_category_event_count
                row["hist_category_severity_rate_available"] = 1
            else:
                row["hist_category_severity_rate"] = 0.0
                row["hist_category_severity_rate_available"] = 0

        # ── Assemble into ordered numpy array ───────────────────────────────
        feature_vector = np.array(
            [float(row.get(col, 0.0)) for col in self.feature_columns],
            dtype=np.float32,
        )

        # Sanity: verify no NaN in output
        nan_count = np.isnan(feature_vector).sum()
        if nan_count > 0:
            log.warning(
                "QueryFeatureBuilder.build: %d NaN values in output — replacing with 0",
                nan_count,
            )
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)

        log.debug(
            "QueryFeatureBuilder.build: feature vector shape=%s, non-zero=%d",
            feature_vector.shape,
            (feature_vector != 0).sum(),
        )

        return feature_vector.reshape(1, -1)

    def __repr__(self) -> str:
        return (
            f"QueryFeatureBuilder("
            f"n_features={len(self.feature_columns)}, "
            f"text_transformer={self.text_transformer!r})"
        )
