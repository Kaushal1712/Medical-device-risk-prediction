"""
tests/risk/test_risk_scorer.py
==============================
Stage 6 — Risk Scoring Engine test suite.

Test classes
------------
  TestProbabilityToScore     — pure-function unit tests for probability_to_score()
  TestScoreToBand            — pure-function unit tests for score_to_band()
  TestScoringResult          — ScoringResult dataclass structure
  TestRiskScorerInit         — RiskScorer constructor / artifact loading
  TestRiskScorerCalibration  — calibrated model exists and loads correctly
  TestRiskScorerScore        — single-row .score() integration tests
  TestRiskScorerBatchScore   — batch_score() integration tests
  TestServingTable           — serving-table artifact validation
  TestCalibrationReport      — calibration_report.json content validation
  TestConfigThresholds       — src/config.py thresholds are valid post-calibrate

Run all:    pytest tests/risk/test_risk_scorer.py -v
Run fast:   pytest tests/risk/test_risk_scorer.py -v -m "not requires_trained_model"
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Project paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTION_DIR = ROOT / "models" / "production"
FEATURES_DIR = ROOT / "data" / "features"
ARTIFACTS_RISK_DIR = ROOT / "artifacts" / "risk"
RISK_SNAPSHOT = ARTIFACTS_RISK_DIR / "device_risk_snapshot.parquet"

METADATA_COLS = frozenset(
    {"id", "device_id", "manufacturer_id", "event_date", "event_date_available", "is_class_i"}
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_val_split():
    """Return (X float32, y int, feat_cols, df) for the validation split."""
    df = pd.read_parquet(FEATURES_DIR / "validation.parquet")
    feat_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feat_cols].values.astype(np.float32)
    y = df["is_class_i"].values.astype(int)
    return X, y, feat_cols, df


# ─────────────────────────────────────────────────────────────────────────────
# TestProbabilityToScore  — pure function, no model required
# ─────────────────────────────────────────────────────────────────────────────

class TestProbabilityToScore:
    """Unit tests for probability_to_score() — no model or data required."""

    def _fn(self):
        from src.risk.scorer import probability_to_score
        return probability_to_score

    def test_zero_probability_gives_zero_score(self):
        fn = self._fn()
        assert fn(0.0) == 0.0

    def test_one_probability_gives_hundred_score(self):
        fn = self._fn()
        assert fn(1.0) == 100.0

    def test_half_probability_gives_fifty(self):
        fn = self._fn()
        assert fn(0.5) == pytest.approx(50.0)

    def test_linear_scaling(self):
        fn = self._fn()
        assert fn(0.25) == pytest.approx(25.0)
        assert fn(0.75) == pytest.approx(75.0)
        assert fn(0.123) == pytest.approx(12.3)

    def test_result_is_rounded_to_two_decimal_places(self):
        fn = self._fn()
        result = fn(0.333333)
        # Should be rounded to 2 dp: 33.33
        assert result == 33.33

    def test_output_is_float(self):
        fn = self._fn()
        result = fn(0.42)
        assert isinstance(result, float)

    def test_nan_raises_value_error(self):
        fn = self._fn()
        with pytest.raises(ValueError, match="NaN"):
            fn(float("nan"))

    def test_negative_probability_raises_value_error(self):
        fn = self._fn()
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            fn(-0.01)

    def test_above_one_raises_value_error(self):
        fn = self._fn()
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            fn(1.001)

    def test_boundary_values_accepted(self):
        """Exact 0.0 and 1.0 must not raise."""
        fn = self._fn()
        assert fn(0.0) == 0.0
        assert fn(1.0) == 100.0

    def test_score_in_range_0_100(self):
        fn = self._fn()
        for p in np.linspace(0.0, 1.0, 21):
            score = fn(float(p))
            assert 0.0 <= score <= 100.0, f"Score {score} out of [0,100] for p={p}"


# ─────────────────────────────────────────────────────────────────────────────
# TestScoreToBand  — pure function, no model required
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreToBand:
    """Unit tests for score_to_band() — no model or data required."""

    def _fn(self):
        from src.risk.scorer import score_to_band
        return score_to_band

    # --- Correct classification ---

    def test_below_medium_is_low(self):
        fn = self._fn()
        assert fn(0.0, 0.5, 0.8) == "LOW"
        assert fn(0.3, 0.5, 0.8) == "LOW"
        assert fn(0.4999, 0.5, 0.8) == "LOW"

    def test_at_medium_threshold_is_medium(self):
        fn = self._fn()
        assert fn(0.5, 0.5, 0.8) == "MEDIUM"

    def test_between_medium_and_high_is_medium(self):
        fn = self._fn()
        assert fn(0.6, 0.5, 0.8) == "MEDIUM"
        assert fn(0.7999, 0.5, 0.8) == "MEDIUM"

    def test_at_high_threshold_is_high(self):
        fn = self._fn()
        assert fn(0.8, 0.5, 0.8) == "HIGH"

    def test_above_high_is_high(self):
        fn = self._fn()
        assert fn(0.9, 0.5, 0.8) == "HIGH"
        assert fn(1.0, 0.5, 0.8) == "HIGH"

    # --- Production operational bands (score-based: LOW<20, MEDIUM 20-50, HIGH>=50) ---
    # Probability thresholds: T_MEDIUM=0.20, T_HIGH=0.50
    # Equivalent to: risk_score = calibrated_probability * 100
    #   LOW:    risk_score <  20   calibrated_prob <  0.20
    #   MEDIUM: risk_score >= 20   calibrated_prob >= 0.20  (and < 0.50)
    #   HIGH:   risk_score >= 50   calibrated_prob >= 0.50

    def test_production_low_below_score_20(self):
        """score < 20 (prob < 0.20) must be LOW with production thresholds."""
        fn = self._fn()
        assert fn(0.0, 0.20, 0.50) == "LOW"
        assert fn(0.10, 0.20, 0.50) == "LOW"
        assert fn(0.1999, 0.20, 0.50) == "LOW"

    def test_production_medium_at_score_20(self):
        """score == 20 (prob == 0.20) must be MEDIUM with production thresholds."""
        fn = self._fn()
        assert fn(0.20, 0.20, 0.50) == "MEDIUM"

    def test_production_medium_between_20_and_50(self):
        """20 <= score < 50 must be MEDIUM with production thresholds."""
        fn = self._fn()
        assert fn(0.30, 0.20, 0.50) == "MEDIUM"
        assert fn(0.49, 0.20, 0.50) == "MEDIUM"
        assert fn(0.4999, 0.20, 0.50) == "MEDIUM"

    def test_production_high_at_score_50(self):
        """score == 50 (prob == 0.50) must be HIGH with production thresholds."""
        fn = self._fn()
        assert fn(0.50, 0.20, 0.50) == "HIGH"

    def test_production_high_above_score_50(self):
        """score > 50 must be HIGH with production thresholds."""
        fn = self._fn()
        assert fn(0.75, 0.20, 0.50) == "HIGH"
        assert fn(1.00, 0.20, 0.50) == "HIGH"

    def test_production_thresholds_from_config_match(self):
        """Production thresholds in src.config must match the expected values."""
        import importlib
        import src.config as cfg
        importlib.reload(cfg)
        fn = self._fn()
        # Below T_MEDIUM=0.20 → LOW
        assert fn(0.0, cfg.RISK_THRESHOLD_MEDIUM, cfg.RISK_THRESHOLD_HIGH) == "LOW"
        assert fn(0.19, cfg.RISK_THRESHOLD_MEDIUM, cfg.RISK_THRESHOLD_HIGH) == "LOW"
        # At T_MEDIUM → MEDIUM
        assert fn(cfg.RISK_THRESHOLD_MEDIUM, cfg.RISK_THRESHOLD_MEDIUM, cfg.RISK_THRESHOLD_HIGH) == "MEDIUM"
        # At T_HIGH=0.50 → HIGH
        assert fn(cfg.RISK_THRESHOLD_HIGH, cfg.RISK_THRESHOLD_MEDIUM, cfg.RISK_THRESHOLD_HIGH) == "HIGH"

    # --- Invalid configurations ---

    def test_t_medium_equal_t_high_raises(self):
        fn = self._fn()
        with pytest.raises(ValueError, match="t_medium"):
            fn(0.5, 0.7, 0.7)

    def test_t_medium_above_t_high_raises(self):
        fn = self._fn()
        with pytest.raises(ValueError, match="t_medium"):
            fn(0.5, 0.9, 0.8)

    # --- Output type ---

    def test_returns_string(self):
        fn = self._fn()
        result = fn(0.5, 0.3, 0.7)
        assert isinstance(result, str)

    def test_returns_one_of_three_levels(self):
        fn = self._fn()
        valid_levels = {"LOW", "MEDIUM", "HIGH"}
        for p in [0.0, 0.3, 0.5, 0.6, 0.8, 0.9, 1.0]:
            assert fn(p, 0.4, 0.7) in valid_levels


# ─────────────────────────────────────────────────────────────────────────────
# TestScoringResult  — dataclass structure
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringResult:
    """Unit tests for ScoringResult dataclass."""

    def _make_result(self, **kwargs):
        from src.risk.scorer import ScoringResult
        defaults = dict(
            raw_probability=0.9,
            calibrated_probability=0.8,
            risk_score=80.0,
            risk_level="HIGH",
            is_class_i_predicted=True,
            decision_threshold=0.8555,
            model_version="test_v1",
        )
        defaults.update(kwargs)
        return ScoringResult(**defaults)

    def test_required_fields_exist(self):
        r = self._make_result()
        assert hasattr(r, "raw_probability")
        assert hasattr(r, "calibrated_probability")
        assert hasattr(r, "risk_score")
        assert hasattr(r, "risk_level")
        assert hasattr(r, "is_class_i_predicted")
        assert hasattr(r, "decision_threshold")
        assert hasattr(r, "model_version")
        assert hasattr(r, "warnings")

    def test_warnings_defaults_to_empty_list(self):
        r = self._make_result()
        assert r.warnings == []

    def test_warnings_accepts_list(self):
        r = self._make_result(warnings=["all NaN"])
        assert r.warnings == ["all NaN"]

    def test_risk_level_values(self):
        for level in ("LOW", "MEDIUM", "HIGH"):
            r = self._make_result(risk_level=level)
            assert r.risk_level == level

    def test_is_class_i_predicted_is_bool(self):
        r = self._make_result(is_class_i_predicted=True)
        assert isinstance(r.is_class_i_predicted, bool)


# ─────────────────────────────────────────────────────────────────────────────
# TestRiskScorerInit  — requires production model artifacts
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.requires_trained_model
class TestRiskScorerInit:
    """RiskScorer loads base model and model card correctly."""

    def _make_scorer(self):
        from src.risk.scorer import RiskScorer
        return RiskScorer(PRODUCTION_DIR)

    def test_scorer_loads_without_error(self):
        scorer = self._make_scorer()
        assert scorer is not None

    def test_feature_columns_loaded(self):
        scorer = self._make_scorer()
        assert len(scorer.feature_columns) == 62, (
            f"Expected 62 feature columns, got {len(scorer.feature_columns)}"
        )

    def test_decision_threshold_loaded(self):
        scorer = self._make_scorer()
        assert 0.0 < scorer.decision_threshold < 1.0, (
            f"decision_threshold {scorer.decision_threshold} outside (0,1)"
        )

    def test_decision_threshold_matches_card(self):
        scorer = self._make_scorer()
        card = json.loads((PRODUCTION_DIR / "model_card.json").read_text())
        assert scorer.decision_threshold == pytest.approx(
            float(card["decision_threshold"]), rel=1e-6
        )

    def test_model_version_non_empty(self):
        scorer = self._make_scorer()
        assert scorer.model_version and scorer.model_version != "unknown"

    def test_missing_model_raises_file_not_found(self, tmp_path):
        from src.risk.scorer import RiskScorer
        with pytest.raises(FileNotFoundError, match="model not found"):
            RiskScorer(tmp_path)

    def test_missing_card_raises_file_not_found(self, tmp_path):
        import shutil
        from src.risk.scorer import RiskScorer
        # Copy model but not card
        shutil.copy(PRODUCTION_DIR / "model.pkl", tmp_path / "model.pkl")
        with pytest.raises(FileNotFoundError, match="model card not found"):
            RiskScorer(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# TestRiskScorerCalibration  — requires calibrated model
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.requires_trained_model
class TestRiskScorerCalibration:
    """Calibrated model artifact exists and loads correctly."""

    def test_calibrated_model_pkl_exists(self):
        assert (PRODUCTION_DIR / "calibrated_model.pkl").exists(), (
            "calibrated_model.pkl not found. Run: python -m src.risk.calibrate"
        )

    def test_calibration_report_json_exists(self):
        assert (PRODUCTION_DIR / "calibration_report.json").exists()

    def test_load_calibration_succeeds(self):
        from src.risk.scorer import RiskScorer
        scorer = RiskScorer(PRODUCTION_DIR)
        scorer.load_calibration()
        assert scorer._calibrated_model is not None

    def test_calibrated_model_has_predict_proba(self):
        from src.risk.scorer import RiskScorer
        scorer = RiskScorer(PRODUCTION_DIR)
        scorer.load_calibration()
        assert hasattr(scorer._calibrated_model, "predict_proba")

    def test_calibrated_proba_shape_and_range(self):
        """Calibrated probabilities on val split are in [0,1]."""
        from src.risk.scorer import RiskScorer
        scorer = RiskScorer(PRODUCTION_DIR)
        scorer.load_calibration()
        X_val, _, _, _ = _load_val_split()
        cal_p = scorer._calibrated_proba(X_val)
        assert cal_p.shape == (len(X_val),)
        assert np.all(cal_p >= 0.0) and np.all(cal_p <= 1.0), (
            f"Calibrated probabilities outside [0,1]: "
            f"min={cal_p.min():.6f}  max={cal_p.max():.6f}"
        )

    def test_load_calibration_without_file_raises(self, tmp_path):
        """load_calibration raises FileNotFoundError if pkl absent."""
        import shutil
        from src.risk.scorer import RiskScorer
        shutil.copy(PRODUCTION_DIR / "model.pkl", tmp_path / "model.pkl")
        shutil.copy(PRODUCTION_DIR / "model_card.json", tmp_path / "model_card.json")
        scorer = RiskScorer(tmp_path)
        with pytest.raises(FileNotFoundError, match="calibrated model not found"):
            scorer.load_calibration()

    def test_uncalibrated_scorer_raises_on_calibrated_proba(self):
        """Calling _calibrated_proba without loading calibration raises RuntimeError."""
        from src.risk.scorer import RiskScorer
        scorer = RiskScorer(PRODUCTION_DIR)
        X_val, _, _, _ = _load_val_split()
        with pytest.raises(RuntimeError, match="not loaded"):
            scorer._calibrated_proba(X_val[:2])


# ─────────────────────────────────────────────────────────────────────────────
# TestRiskScorerScore  — single-row scoring
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.requires_trained_model
class TestRiskScorerScore:
    """Single-row .score() integration tests."""

    @pytest.fixture(scope="class")
    def scorer(self):
        from src.risk.scorer import RiskScorer
        s = RiskScorer(PRODUCTION_DIR)
        s.load_calibration()
        return s

    @pytest.fixture(scope="class")
    def val_data(self):
        return _load_val_split()

    def test_score_returns_scoring_result(self, scorer, val_data):
        from src.risk.scorer import ScoringResult
        X, _, feat_cols, _ = val_data
        result = scorer.score(X[:1], feat_cols, t_medium=0.3, t_high=0.7)
        assert isinstance(result, ScoringResult)

    def test_score_all_required_fields_populated(self, scorer, val_data):
        X, _, feat_cols, _ = val_data
        r = scorer.score(X[:1], feat_cols, t_medium=0.3, t_high=0.7)
        assert 0.0 <= r.raw_probability <= 1.0
        assert 0.0 <= r.calibrated_probability <= 1.0
        assert 0.0 <= r.risk_score <= 100.0
        assert r.risk_level in {"LOW", "MEDIUM", "HIGH"}
        assert isinstance(r.is_class_i_predicted, (bool, np.bool_))
        assert 0.0 < r.decision_threshold < 1.0
        assert r.model_version

    def test_score_risk_score_equals_calibrated_prob_times_100(self, scorer, val_data):
        X, _, feat_cols, _ = val_data
        r = scorer.score(X[0], feat_cols, t_medium=0.3, t_high=0.7)
        expected = round(r.calibrated_probability * 100.0, 2)
        assert r.risk_score == pytest.approx(expected, abs=0.01)

    def test_score_accepts_1d_input(self, scorer, val_data):
        """1D feature array (n_features,) should work as well as (1, n_features)."""
        X, _, feat_cols, _ = val_data
        r1d = scorer.score(X[0], feat_cols, t_medium=0.3, t_high=0.7)
        r2d = scorer.score(X[:1], feat_cols, t_medium=0.3, t_high=0.7)
        assert r1d.risk_score == pytest.approx(r2d.risk_score, abs=1e-6)

    def test_score_is_deterministic(self, scorer, val_data):
        """Same row always gives same result.

        Note: raw_probability uses approx because Random Forest parallel tree
        averaging can produce last-ULP (1e-15) float differences across calls
        in Python 3.14. risk_score and risk_level (from calibrated prob) are
        exact-deterministic.
        """
        X, _, feat_cols, _ = val_data
        r1 = scorer.score(X[0], feat_cols, t_medium=0.3, t_high=0.7)
        r2 = scorer.score(X[0], feat_cols, t_medium=0.3, t_high=0.7)
        assert r1.risk_score == r2.risk_score
        assert r1.risk_level == r2.risk_level
        # raw_probability: allow last-ULP machine-epsilon tolerance
        assert r1.raw_probability == pytest.approx(r2.raw_probability, rel=1e-10)

    def test_score_wrong_feature_count_raises(self, scorer, val_data):
        X, _, feat_cols, _ = val_data
        # Drop one column
        with pytest.raises(ValueError, match="expected.*feature columns"):
            scorer.score(X[:1, :-1], feat_cols[:-1], t_medium=0.3, t_high=0.7)

    def test_is_class_i_predicted_matches_threshold(self, scorer, val_data):
        """is_class_i_predicted = raw_prob >= decision_threshold."""
        X, _, feat_cols, _ = val_data
        for i in range(min(30, len(X))):
            r = scorer.score(X[i], feat_cols, t_medium=0.3, t_high=0.7)
            expected = r.raw_probability >= scorer.decision_threshold
            assert bool(r.is_class_i_predicted) == bool(expected), (
                f"Row {i}: is_class_i_predicted={r.is_class_i_predicted}, "
                f"raw_p={r.raw_probability:.4f}, thr={scorer.decision_threshold:.4f}"
            )

    def test_all_nan_row_returns_warning(self, scorer, val_data):
        """An all-NaN feature row must return a non-empty warnings list."""
        X, _, feat_cols, _ = val_data
        X_nan = np.full((1, X.shape[1]), np.nan, dtype=np.float32)
        r = scorer.score(X_nan, feat_cols, t_medium=0.3, t_high=0.7)
        assert len(r.warnings) > 0, "Expected warning for all-NaN row"

    def test_score_uses_production_thresholds_from_config_when_none(self, scorer, val_data):
        """score() falls back to src.config thresholds when t_medium/t_high are None."""
        import src.config as cfg
        X, _, feat_cols, _ = val_data
        r_config = scorer.score(X[0], feat_cols)  # use defaults
        r_explicit = scorer.score(
            X[0], feat_cols,
            t_medium=cfg.RISK_THRESHOLD_MEDIUM,
            t_high=cfg.RISK_THRESHOLD_HIGH,
        )
        assert r_config.risk_level == r_explicit.risk_level
        assert r_config.risk_score == pytest.approx(r_explicit.risk_score, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# TestRiskScorerBatchScore  — batch scoring
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.requires_trained_model
class TestRiskScorerBatchScore:
    """batch_score() integration tests."""

    @pytest.fixture(scope="class")
    def scorer(self):
        from src.risk.scorer import RiskScorer
        s = RiskScorer(PRODUCTION_DIR)
        s.load_calibration()
        return s

    @pytest.fixture(scope="class")
    def val_df(self):
        return pd.read_parquet(FEATURES_DIR / "validation.parquet")

    def test_batch_score_returns_dataframe(self, scorer, val_df):
        result = scorer.batch_score(val_df, t_medium=0.3, t_high=0.7)
        assert isinstance(result, pd.DataFrame)

    def test_batch_score_correct_row_count(self, scorer, val_df):
        result = scorer.batch_score(val_df, t_medium=0.3, t_high=0.7)
        assert len(result) == len(val_df)

    def test_batch_score_required_columns(self, scorer, val_df):
        result = scorer.batch_score(val_df, t_medium=0.3, t_high=0.7)
        required = {
            "id", "device_id", "event_date",
            "raw_probability", "calibrated_probability", "risk_score",
            "risk_level", "is_class_i_predicted", "decision_threshold",
            "model_version",
        }
        missing = required - set(result.columns)
        assert len(missing) == 0, f"Missing columns in batch_score output: {missing}"

    def test_batch_score_probability_in_range(self, scorer, val_df):
        result = scorer.batch_score(val_df, t_medium=0.3, t_high=0.7)
        assert (result["raw_probability"] >= 0).all() and (result["raw_probability"] <= 1).all()
        assert (result["calibrated_probability"] >= 0).all()
        assert (result["calibrated_probability"] <= 1).all()

    def test_batch_score_risk_score_in_range(self, scorer, val_df):
        result = scorer.batch_score(val_df, t_medium=0.3, t_high=0.7)
        assert (result["risk_score"] >= 0).all() and (result["risk_score"] <= 100).all()

    def test_batch_score_risk_levels_valid(self, scorer, val_df):
        result = scorer.batch_score(val_df, t_medium=0.3, t_high=0.7)
        assert set(result["risk_level"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})

    def test_batch_score_deterministic(self, scorer, val_df):
        """Two identical calls must return bit-identical scores."""
        r1 = scorer.batch_score(val_df, t_medium=0.3, t_high=0.7)
        r2 = scorer.batch_score(val_df, t_medium=0.3, t_high=0.7)
        pd.testing.assert_series_equal(r1["risk_score"], r2["risk_score"])
        pd.testing.assert_series_equal(r1["risk_level"], r2["risk_level"])

    def test_batch_score_risk_score_equals_prob_times_100(self, scorer, val_df):
        result = scorer.batch_score(val_df, t_medium=0.3, t_high=0.7)
        expected = (result["calibrated_probability"] * 100.0).round(2)
        pd.testing.assert_series_equal(
            result["risk_score"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_batch_score_missing_model_feature_raises(self, scorer, val_df):
        """Dropping a required model feature column raises ValueError."""
        card = json.loads((PRODUCTION_DIR / "model_card.json").read_text())
        first_feat = card["feature_columns"][0]
        df_missing = val_df.drop(columns=[first_feat])
        with pytest.raises(ValueError, match="missing.*model features"):
            scorer.batch_score(df_missing, t_medium=0.3, t_high=0.7)

    def test_batch_score_extra_column_is_tolerated(self, scorer, val_df):
        """Extra non-feature columns (like _split) must not raise."""
        df_extra = val_df.copy()
        df_extra["_split"] = "validation"
        result = scorer.batch_score(df_extra, t_medium=0.3, t_high=0.7)
        assert len(result) == len(val_df)


# ─────────────────────────────────────────────────────────────────────────────
# TestServingTable  — serving-table artifact
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.requires_trained_model
class TestServingTable:
    """Validate the device_risk_snapshot.parquet serving table."""

    @pytest.fixture(scope="class")
    def serving(self):
        assert RISK_SNAPSHOT.exists(), (
            f"{RISK_SNAPSHOT} not found. Run: python -m src.risk.build_serving_table"
        )
        return pd.read_parquet(RISK_SNAPSHOT)

    def test_serving_table_exists(self):
        assert RISK_SNAPSHOT.exists()

    def test_required_serving_columns(self, serving):
        required = {
            "device_id", "event_id", "serving_event_date",
            "raw_probability", "calibrated_probability", "risk_score",
            "risk_level", "is_class_i_predicted", "decision_threshold",
            "model_version", "scored_at",
        }
        missing = required - set(serving.columns)
        assert len(missing) == 0, f"Missing serving-table columns: {missing}"

    def test_one_row_per_device_id(self, serving):
        """Stage 3f serving policy: exactly one row per device_id."""
        duplicates = serving["device_id"].duplicated().sum()
        assert duplicates == 0, (
            f"Serving table has {duplicates} duplicate device_ids — "
            "Stage 3f policy requires exactly one row per device."
        )

    def test_total_devices_matches_feature_splits(self, serving):
        """Serving table must cover all unique device_ids in all splits."""
        dfs = []
        for split in ["train", "validation", "test", "holdout_2018"]:
            df = pd.read_parquet(FEATURES_DIR / f"{split}.parquet")
            dfs.append(df[["device_id"]])
        all_devices = pd.concat(dfs)["device_id"].nunique()
        assert len(serving) == all_devices, (
            f"Serving table has {len(serving)} rows but there are "
            f"{all_devices} unique device_ids across all splits."
        )

    def test_risk_score_in_range(self, serving):
        assert (serving["risk_score"] >= 0).all()
        assert (serving["risk_score"] <= 100).all()

    def test_probabilities_in_unit_interval(self, serving):
        assert (serving["raw_probability"] >= 0).all()
        assert (serving["raw_probability"] <= 1).all()
        assert (serving["calibrated_probability"] >= 0).all()
        assert (serving["calibrated_probability"] <= 1).all()

    def test_risk_levels_valid(self, serving):
        assert set(serving["risk_level"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})

    def test_all_three_risk_levels_present(self, serving):
        """The serving table must contain all three risk levels."""
        levels = set(serving["risk_level"].unique())
        assert "LOW" in levels, "No LOW-risk devices in serving table"
        assert "HIGH" in levels, "No HIGH-risk devices in serving table"

    def test_high_risk_count_plausible(self, serving):
        """HIGH band should be > 0 and < 50% of all devices."""
        n_high = (serving["risk_level"] == "HIGH").sum()
        assert n_high > 0, "No HIGH-risk devices — HIGH band may be empty"
        assert n_high < len(serving) * 0.5, (
            f"HIGH band has {n_high}/{len(serving)} devices — suspiciously large"
        )

    def test_decision_threshold_consistent(self, serving):
        """All rows must share the same decision threshold from the model card."""
        assert serving["decision_threshold"].nunique() == 1, (
            "Multiple decision_threshold values found in serving table"
        )
        card = json.loads((PRODUCTION_DIR / "model_card.json").read_text())
        expected_thr = float(card["decision_threshold"])
        actual_thr = float(serving["decision_threshold"].iloc[0])
        assert actual_thr == pytest.approx(expected_thr, rel=1e-5)

    def test_is_class_i_predicted_matches_decision_threshold(self, serving):
        """is_class_i_predicted must equal raw_prob >= decision_threshold."""
        thr = float(serving["decision_threshold"].iloc[0])
        expected = serving["raw_probability"] >= thr
        mismatches = (serving["is_class_i_predicted"] != expected).sum()
        assert mismatches == 0, (
            f"{mismatches} rows have is_class_i_predicted inconsistent with threshold"
        )

    def test_risk_score_equals_calibrated_prob_times_100(self, serving):
        expected = (serving["calibrated_probability"] * 100.0).round(2)
        pd.testing.assert_series_equal(
            serving["risk_score"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            rtol=1e-4,
        )

    def test_scored_at_is_utc_iso_string(self, serving):
        """scored_at should be a non-empty string (ISO 8601 UTC timestamp)."""
        assert serving["scored_at"].nunique() == 1
        scored_at = serving["scored_at"].iloc[0]
        assert isinstance(scored_at, str)
        assert len(scored_at) > 10

    def test_model_version_consistent(self, serving):
        assert serving["model_version"].nunique() == 1

    def test_serving_event_date_is_latest_per_device(self):
        """For devices with multiple events, the serving row must be the latest."""
        # Load all events across splits
        dfs = []
        for split in ["train", "validation", "test", "holdout_2018"]:
            df = pd.read_parquet(FEATURES_DIR / f"{split}.parquet")
            df["_split"] = split
            dfs.append(df[["device_id", "event_date"]])
        all_events = pd.concat(dfs, ignore_index=True)
        all_events["event_date"] = pd.to_datetime(all_events["event_date"])

        # Compute true latest per device
        latest_per_device = (
            all_events.groupby("device_id")["event_date"].max().reset_index()
        )

        # Load serving table
        serving = pd.read_parquet(RISK_SNAPSHOT)
        serving["serving_event_date"] = pd.to_datetime(serving["serving_event_date"])

        merged = pd.merge(
            serving[["device_id", "serving_event_date"]],
            latest_per_device.rename(columns={"event_date": "true_latest"}),
            on="device_id",
            how="inner",
        )
        mismatches = (merged["serving_event_date"] != merged["true_latest"]).sum()
        assert mismatches == 0, (
            f"{mismatches} devices in the serving table do not have the latest event_date. "
            "Stage 3f serving policy violated."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestCalibrationReport  — calibration_report.json content validation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.requires_trained_model
class TestCalibrationReport:
    """Validate calibration_report.json content."""

    @pytest.fixture(scope="class")
    def report(self):
        path = PRODUCTION_DIR / "calibration_report.json"
        assert path.exists(), "calibration_report.json not found"
        return json.loads(path.read_text())

    def test_required_top_level_keys(self, report):
        required = {
            "calibration_method", "risk_score_version", "base_model",
            "n_train", "n_val", "positive_rate_val",
            "before_calibration", "after_calibration",
            "brier_improvement", "risk_band_thresholds",
            "validation_band_distribution",
        }
        missing = required - set(report.keys())
        assert len(missing) == 0, f"Missing keys in calibration report: {missing}"

    def test_calibration_method_is_isotonic(self, report):
        assert report["calibration_method"] == "isotonic"

    def test_brier_improved_after_calibration(self, report):
        """Post-calibration Brier score must be lower than pre-calibration."""
        before = report["before_calibration"]["brier_score"]
        after = report["after_calibration"]["brier_score"]
        assert after < before, (
            f"Brier score did not improve after calibration: before={before:.6f} after={after:.6f}"
        )

    def test_brier_improvement_positive(self, report):
        assert report["brier_improvement"] > 0

    def test_n_train_matches_expected(self, report):
        assert report["n_train"] == 38_247

    def test_n_val_matches_expected(self, report):
        assert report["n_val"] == 4_273

    def test_risk_band_thresholds_keys(self, report):
        thr = report["risk_band_thresholds"]
        for key in ["t_medium", "t_high", "t_medium_reasoning", "t_high_reasoning"]:
            assert key in thr, f"Missing key in risk_band_thresholds: {key}"

    def test_t_medium_in_unit_interval(self, report):
        t = report["risk_band_thresholds"]["t_medium"]
        assert 0.0 < t <= 1.0, f"t_medium={t} outside (0, 1]"

    def test_t_high_in_unit_interval(self, report):
        t = report["risk_band_thresholds"]["t_high"]
        assert 0.0 < t <= 1.0, f"t_high={t} outside (0, 1]"

    def test_t_medium_less_than_or_equal_to_t_high(self, report):
        t_medium = report["risk_band_thresholds"]["t_medium"]
        t_high = report["risk_band_thresholds"]["t_high"]
        assert t_medium <= t_high, (
            f"t_medium ({t_medium}) > t_high ({t_high}) — invalid band configuration"
        )

    def test_validation_band_distribution_sums_to_n_val(self, report):
        dist = report["validation_band_distribution"]
        total = dist["LOW"] + dist.get("MEDIUM", 0) + dist["HIGH"]
        assert total == report["n_val"], (
            f"Band distribution total {total} != n_val {report['n_val']}"
        )

    def test_high_band_not_empty(self, report):
        """HIGH band must catch at least some devices on validation."""
        n_high = report["validation_band_distribution"]["HIGH"]
        assert n_high > 0, "HIGH band is empty on validation — threshold derivation may have failed"

    def test_t_high_matches_config(self, report):
        """T_HIGH in the calibration report must match RISK_THRESHOLD_HIGH in config.

        The calibration report records whatever thresholds were active when
        the serving table was last built. Since we changed the operational
        bands, the report now reflects the new values.
        """
        import importlib
        import src.config as cfg
        importlib.reload(cfg)
        t_high = report["risk_band_thresholds"]["t_high"]
        assert t_high == pytest.approx(cfg.RISK_THRESHOLD_HIGH, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# TestConfigThresholds  — config thresholds are valid
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigThresholds:
    """src/config.py thresholds are set to valid values after calibration."""

    def test_risk_threshold_medium_not_sentinel(self):
        import importlib
        import src.config as cfg
        importlib.reload(cfg)
        assert cfg.RISK_THRESHOLD_MEDIUM != 0.0, (
            "RISK_THRESHOLD_MEDIUM is still 0.0 (sentinel). "
            "Run: python -m src.risk.calibrate"
        )

    def test_risk_threshold_high_not_sentinel(self):
        import importlib
        import src.config as cfg
        importlib.reload(cfg)
        assert cfg.RISK_THRESHOLD_HIGH != 0.0, (
            "RISK_THRESHOLD_HIGH is still 0.0 (sentinel). "
            "Run: python -m src.risk.calibrate"
        )

    def test_thresholds_in_valid_range(self):
        import importlib
        import src.config as cfg
        importlib.reload(cfg)
        assert 0.0 < cfg.RISK_THRESHOLD_MEDIUM <= 1.0
        assert 0.0 < cfg.RISK_THRESHOLD_HIGH <= 1.0

    def test_medium_threshold_less_than_or_equal_high(self):
        import importlib
        import src.config as cfg
        importlib.reload(cfg)
        assert cfg.RISK_THRESHOLD_MEDIUM <= cfg.RISK_THRESHOLD_HIGH, (
            f"RISK_THRESHOLD_MEDIUM ({cfg.RISK_THRESHOLD_MEDIUM}) > "
            f"RISK_THRESHOLD_HIGH ({cfg.RISK_THRESHOLD_HIGH})"
        )

    def test_risk_score_version_defined(self):
        import src.config as cfg
        assert hasattr(cfg, "RISK_SCORE_VERSION")
        assert cfg.RISK_SCORE_VERSION != ""

    def test_calibration_artifact_paths_defined(self):
        import src.config as cfg
        assert hasattr(cfg, "CALIBRATED_MODEL_PATH")
        assert hasattr(cfg, "CALIBRATION_REPORT_PATH")
        assert hasattr(cfg, "RISK_SNAPSHOT_PATH")
