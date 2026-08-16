"""
tests/models/test_model.py
==========================
Stage 5 test suite for the ML Risk Engine.

Test classes:
  TestDataLoading          — feature parquets load correctly
  TestLeakageInModel       — no prohibited columns reach X arrays
  TestMetricsComputation   — compute_metrics_at_threshold returns correct keys/values
  TestProductionModel      — production model artifact loads and predicts (requires training)
  TestTestMetrics          — test_metrics.json validates against baseline (requires training)

Run all:   pytest tests/models/test_model.py -v
Run fast:  pytest tests/models/test_model.py -v -m "not requires_trained_model"
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Project paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = ROOT / "data" / "features"
PRODUCTION_DIR = ROOT / "models" / "production"
METADATA_COLS = {"id", "device_id", "manufacturer_id", "event_date", "event_date_available"}
TARGET_COL = "is_class_i"

# Prohibited feature names (must never appear in X)
PROHIBITED_FEATURES = frozenset({
    "action", "action_summary", "action_classification", "action_level",
    "determined_cause", "status", "date_terminated", "date_updated",
    "target_audience", "reason",
    "slug", "uid", "uid_hash", "url", "authorities_link",
    "documents", "icij_notes", "data_notes", "number",
    "device_slug", "device_number", "device_distributed_to",
    "device_quantity_in_commerce",
    "mfr_slug", "mfr_comment", "mfr_representative", "mfr_address",
    "created_at", "updated_at", "device_created_at", "device_updated_at",
    "mfr_created_at", "mfr_updated_at",
    "date", "date_initiated_by_firm", "date_posted", "create_date",
    "event_date_source",
})

SPLIT_NAMES = ["train", "validation", "test", "holdout_2018"]
EXPECTED_N_FEATURES = 62  # as verified in Stage 4 report


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_split(name: str):
    path = FEATURES_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS and c != TARGET_COL]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(int)
    return X, y, feature_cols, df


# ─────────────────────────────────────────────────────────────────────────────
# TestDataLoading
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoading:
    """Feature parquets exist, have expected shapes, and contain target."""

    @pytest.mark.parametrize("split", SPLIT_NAMES)
    def test_parquet_exists(self, split):
        path = FEATURES_DIR / f"{split}.parquet"
        assert path.exists(), f"Missing feature file: {path}"

    def test_train_rows(self):
        X, y, _, df = _load_split("train")
        # Stage 4 verified 38,247 rows
        assert len(df) == 38_247, f"Expected 38247 train rows, got {len(df)}"

    def test_validation_rows(self):
        X, y, _, df = _load_split("validation")
        assert len(df) == 4_273, f"Expected 4273 val rows, got {len(df)}"

    def test_test_rows(self):
        X, y, _, df = _load_split("test")
        assert len(df) == 8_918, f"Expected 8918 test rows, got {len(df)}"

    def test_holdout_rows(self):
        X, y, _, df = _load_split("holdout_2018")
        assert len(df) == 1_361, f"Expected 1361 holdout rows, got {len(df)}"

    def test_target_column_present(self):
        _, _, _, df = _load_split("train")
        assert TARGET_COL in df.columns, f"'{TARGET_COL}' not in train columns"

    def test_target_is_binary(self):
        _, y, _, _ = _load_split("train")
        unique_vals = set(np.unique(y))
        assert unique_vals.issubset({0, 1}), f"Target is not binary: {unique_vals}"

    def test_positive_rate_in_range(self):
        _, y, _, _ = _load_split("train")
        pos_rate = y.mean()
        # Stage 4 report: 8.34% in train
        assert 0.07 < pos_rate < 0.10, f"Train positive rate {pos_rate:.4f} outside expected range"

    def test_feature_count(self):
        X, _, feature_cols, _ = _load_split("train")
        assert len(feature_cols) == EXPECTED_N_FEATURES, (
            f"Expected {EXPECTED_N_FEATURES} features, got {len(feature_cols)}"
        )

    def test_metadata_not_in_X(self):
        X, _, feature_cols, _ = _load_split("train")
        overlap = set(feature_cols) & METADATA_COLS
        assert len(overlap) == 0, f"Metadata columns in feature matrix: {overlap}"

    def test_no_all_nan_columns(self):
        X, _, feature_cols, _ = _load_split("train")
        all_nan_mask = np.all(np.isnan(X), axis=0)
        if all_nan_mask.any():
            bad_cols = [feature_cols[i] for i in np.where(all_nan_mask)[0]]
            pytest.fail(f"Columns are all-NaN in train X: {bad_cols}")

    def test_consistent_feature_columns_across_splits(self):
        _, _, train_cols, _ = _load_split("train")
        for split in ["validation", "test", "holdout_2018"]:
            _, _, other_cols, _ = _load_split(split)
            assert train_cols == other_cols, (
                f"Feature columns differ between train and {split}: "
                f"extra={set(other_cols) - set(train_cols)}, "
                f"missing={set(train_cols) - set(other_cols)}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TestLeakageInModel
# ─────────────────────────────────────────────────────────────────────────────

class TestLeakageInModel:
    """No prohibited features or target column reach the model's X array."""

    @pytest.mark.parametrize("split", SPLIT_NAMES)
    def test_target_not_in_feature_columns(self, split):
        X, _, feature_cols, _ = _load_split(split)
        assert TARGET_COL not in feature_cols, (
            f"'{TARGET_COL}' appears in feature columns for split '{split}'"
        )

    @pytest.mark.parametrize("split", SPLIT_NAMES)
    def test_no_prohibited_features_in_X(self, split):
        X, _, feature_cols, _ = _load_split(split)
        leaky = set(feature_cols) & PROHIBITED_FEATURES
        assert len(leaky) == 0, (
            f"Prohibited features found in '{split}' X: {leaky}"
        )

    def test_action_classification_not_in_train(self):
        _, _, _, df = _load_split("train")
        assert "action_classification" not in df.columns, (
            "action_classification (the source of the target) found in train parquet"
        )

    def test_reason_not_in_features(self):
        _, _, feature_cols, _ = _load_split("train")
        assert "reason" not in feature_cols, (
            "'reason' (borderline leaky text field) found in feature columns"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestMetricsComputation
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsComputation:
    """Unit-test the evaluate.py metrics function in isolation."""

    def _get_compute_fn(self):
        import sys
        sys.path.insert(0, str(ROOT))
        from src.models.evaluate import compute_metrics_at_threshold
        return compute_metrics_at_threshold

    def test_returns_required_keys(self):
        fn = self._get_compute_fn()
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        result = fn(y_true, y_proba, 0.5, "test_split")
        required = {
            "split", "n_samples", "roc_auc", "pr_auc",
            "precision", "recall", "f1",
            "tp", "fp", "tn", "fn", "threshold",
        }
        missing = required - set(result.keys())
        assert len(missing) == 0, f"Missing keys in metrics: {missing}"

    def test_perfect_classifier(self):
        fn = self._get_compute_fn()
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.01, 0.02, 0.98, 0.99])
        result = fn(y_true, y_proba, 0.5, "perfect")
        assert result["roc_auc"] == pytest.approx(1.0)
        assert result["pr_auc"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_random_classifier_pr_auc_near_positive_rate(self):
        fn = self._get_compute_fn()
        rng = np.random.default_rng(42)
        n = 10_000
        pos_rate = 0.076
        y_true = (rng.random(n) < pos_rate).astype(int)
        y_proba = rng.random(n)  # random scores
        result = fn(y_true, y_proba, 0.5, "random")
        # For a random classifier, PR-AUC ≈ positive rate
        assert abs(result["pr_auc"] - pos_rate) < 0.02, (
            f"Random classifier PR-AUC={result['pr_auc']:.4f}, expected ~{pos_rate:.4f}"
        )

    def test_n_samples_correct(self):
        fn = self._get_compute_fn()
        y_true = np.array([0, 1, 0, 1, 0])
        y_proba = np.array([0.2, 0.8, 0.3, 0.7, 0.1])
        result = fn(y_true, y_proba, 0.5, "check")
        assert result["n_samples"] == 5

    def test_confusion_matrix_sums(self):
        fn = self._get_compute_fn()
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.3, 0.2])
        result = fn(y_true, y_proba, 0.5, "cm")
        assert result["tp"] + result["fp"] + result["tn"] + result["fn"] == 6


# ─────────────────────────────────────────────────────────────────────────────
# TestProductionModel  (requires trained model)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.requires_trained_model
class TestProductionModel:
    """Production artifacts exist and the model can predict."""

    def test_model_pkl_exists(self):
        assert (PRODUCTION_DIR / "model.pkl").exists(), (
            "models/production/model.pkl not found. Run: python -m src.models.evaluate"
        )

    def test_model_card_exists(self):
        assert (PRODUCTION_DIR / "model_card.json").exists()

    def test_test_metrics_exists(self):
        assert (PRODUCTION_DIR / "test_metrics.json").exists()

    def test_model_loads(self):
        import joblib
        model = joblib.load(PRODUCTION_DIR / "model.pkl")
        assert hasattr(model, "predict_proba"), "Loaded model has no predict_proba"

    def test_model_predict_shape(self):
        import joblib
        model = joblib.load(PRODUCTION_DIR / "model.pkl")
        preprocessor_path = PRODUCTION_DIR / "preprocessor.pkl"
        preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None

        X, y, _, _ = _load_split("test")
        if preprocessor is not None:
            X = preprocessor.transform(X)
        probas = model.predict_proba(X)[:, 1]
        assert probas.shape == (len(y),), (
            f"predict_proba shape {probas.shape} != expected ({len(y)},)"
        )

    def test_probas_in_unit_interval(self):
        import joblib
        model = joblib.load(PRODUCTION_DIR / "model.pkl")
        preprocessor_path = PRODUCTION_DIR / "preprocessor.pkl"
        preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None

        X, _, _, _ = _load_split("test")
        if preprocessor is not None:
            X = preprocessor.transform(X)
        probas = model.predict_proba(X)[:, 1]
        assert np.all(probas >= 0.0) and np.all(probas <= 1.0), (
            "Some predicted probabilities are outside [0, 1]"
        )

    def test_feature_importance_exists_for_tree_models(self):
        import joblib
        model = joblib.load(PRODUCTION_DIR / "model.pkl")
        # Only tree models (RF, XGBoost) have feature_importances_
        if hasattr(model, "feature_importances_"):
            assert (PRODUCTION_DIR / "feature_importance.json").exists(), (
                "feature_importance.json missing for a tree-based model"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TestTestMetrics  (requires trained model)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.requires_trained_model
class TestTestMetrics:
    """Validate test_metrics.json content: best model beats baseline on test."""

    def _load_metrics(self):
        path = PRODUCTION_DIR / "test_metrics.json"
        assert path.exists(), "test_metrics.json not found"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_test_metrics_has_required_keys(self):
        m = self._load_metrics()
        for key in ["validation", "test", "decision_threshold"]:
            assert key in m, f"Missing key '{key}' in test_metrics.json"

    def test_pr_auc_above_baseline(self):
        """Best model PR-AUC on test must beat the positive rate (random baseline)."""
        m = self._load_metrics()
        test_pr_auc = m["test"]["pr_auc"]
        # Positive rate on test is ~5.52%; a useful model should beat this clearly
        MINIMUM_USEFUL_PR_AUC = 0.08
        assert test_pr_auc > MINIMUM_USEFUL_PR_AUC, (
            f"Test PR-AUC={test_pr_auc:.4f} is too close to random baseline (~0.055). "
            f"Model must exceed {MINIMUM_USEFUL_PR_AUC}"
        )

    def test_roc_auc_above_0_5(self):
        m = self._load_metrics()
        roc_auc = m["test"]["roc_auc"]
        assert roc_auc > 0.5, (
            f"Test ROC-AUC={roc_auc:.4f} <= 0.5 (worse than random flip)"
        )

    def test_recall_nonzero(self):
        """Model must be catching at least some Class I events on the test set."""
        m = self._load_metrics()
        recall = m["test"]["recall"]
        assert recall > 0.0, (
            f"Test recall={recall:.4f}: model predicts zero positives. "
            "Check threshold — it may be too high."
        )

    def test_threshold_in_unit_interval(self):
        m = self._load_metrics()
        threshold = m["decision_threshold"]
        assert 0.0 < threshold < 1.0, (
            f"Decision threshold={threshold} outside (0, 1)"
        )

    def test_test_n_samples_correct(self):
        m = self._load_metrics()
        n = m["test"]["n_samples"]
        # Stage 4: test has 8,918 rows
        assert n == 8_918, f"test n_samples={n}, expected 8918"

    def test_holdout_pr_auc_reasonable(self):
        """Holdout 2018 PR-AUC should not drastically diverge from test."""
        m = self._load_metrics()
        if m.get("holdout_2018") is None:
            pytest.skip("No holdout metrics available")
        test_pr_auc = m["test"]["pr_auc"]
        holdout_pr_auc = m["holdout_2018"]["pr_auc"]
        # Allow up to 50% relative drop — temporal drift is expected
        assert holdout_pr_auc > test_pr_auc * 0.5, (
            f"Holdout PR-AUC={holdout_pr_auc:.4f} is drastically lower than "
            f"test PR-AUC={test_pr_auc:.4f}. Possible data/distribution issue."
        )
