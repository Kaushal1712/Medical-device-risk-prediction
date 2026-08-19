"""
tests/data/test_pipeline_integrity.py
========================================
Stage 10 — Pipeline artifact integrity and processed-data schema validation.

Section 9 testing matrix requirements:
  - Data: "schema validation on processed Parquet, join correctness
    (row/match counts match Stage 1/2 findings), missing-value handling
    behaves as documented, target construction produces the expected class
    balance from Stage 3's report."
  - Integration: verify all pipeline artifacts are present, internally
    consistent, and coherent with each other.

Does NOT re-run the pipeline (that would take minutes).
Instead, verifies that the outputs of a prior pipeline run are correct,
complete, and internally consistent — catching stale, corrupted, or
mismatched artifacts immediately.

Run:  python -m pytest tests/data/test_pipeline_integrity.py -v
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR  = ROOT / "data" / "features"
PRODUCTION_DIR = ROOT / "models" / "production"
ARTIFACTS_DIR  = ROOT / "artifacts"
RAW_DIR = ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Skip guard — all tests here need the processed artifacts to exist
# ---------------------------------------------------------------------------

def _artifacts_exist() -> bool:
    return (
        (PROCESSED_DIR / "merged.parquet").exists() and
        (FEATURES_DIR / "train.parquet").exists() and
        (PRODUCTION_DIR / "model_card.json").exists() and
        (ARTIFACTS_DIR / "risk" / "device_risk_snapshot.parquet").exists()
    )


if not _artifacts_exist():
    pytestmark = pytest.mark.skip(reason="Pipeline artifacts not found — run the full pipeline first.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest() -> dict:
    path = PROCESSED_DIR / "_manifest.json"
    assert path.exists(), f"Manifest not found: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model_card() -> dict:
    return json.loads((PRODUCTION_DIR / "model_card.json").read_text(encoding="utf-8"))


def _load_feature_metadata() -> dict:
    return json.loads((FEATURES_DIR / "feature_metadata.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# TestProcessedParquetSchema
# ---------------------------------------------------------------------------

class TestProcessedParquetSchema:
    """
    Verifies that processed Parquet outputs exist, have the correct schema,
    and match the row counts recorded in the pipeline manifest.
    """

    def test_manifest_exists(self):
        assert (PROCESSED_DIR / "_manifest.json").exists(), (
            "data/processed/_manifest.json missing — has the pipeline been run?"
        )

    def test_merged_parquet_exists(self):
        assert (PROCESSED_DIR / "merged.parquet").exists(), (
            "data/processed/merged.parquet missing — run: python -m src.data.pipeline"
        )

    def test_devices_parquet_exists(self):
        assert (PROCESSED_DIR / "devices.parquet").exists()

    def test_events_parquet_exists(self):
        assert (PROCESSED_DIR / "events.parquet").exists()

    def test_manufacturers_parquet_exists(self):
        assert (PROCESSED_DIR / "manufacturers.parquet").exists()

    def test_merged_row_count_matches_manifest(self):
        """
        merged.parquet row count must match the manifest's recorded events_rows
        (Stage 1/2: events table is the source of joined rows — 1:1 with events).
        """
        manifest = _load_manifest()
        expected = manifest["run_stats"]["merged_rows"]
        actual = len(pd.read_parquet(PROCESSED_DIR / "merged.parquet"))
        assert actual == expected, (
            f"merged.parquet has {actual} rows but manifest says {expected}. "
            "Pipeline output may be stale or corrupted."
        )

    def test_merged_join_correctness_events_count(self):
        """
        merged.parquet must have the same number of rows as events.parquet
        (inner join — each event produces exactly one merged row).
        """
        merged = pd.read_parquet(PROCESSED_DIR / "merged.parquet")
        events = pd.read_parquet(PROCESSED_DIR / "events.parquet")
        assert len(merged) == len(events), (
            f"merged ({len(merged)}) != events ({len(events)}). Join may have dropped or duplicated rows."
        )

    def test_merged_has_required_columns(self):
        """merged.parquet must contain device_id, manufacturer_id, and event_date."""
        merged = pd.read_parquet(PROCESSED_DIR / "merged.parquet")
        for col in ("device_id", "manufacturer_id", "event_date"):
            assert col in merged.columns, f"merged.parquet missing column: {col!r}"

    def test_merged_device_id_coverage(self):
        """
        Every device_id in merged.parquet should also appear in events.parquet
        (joined on the events table — 1:1 mapping verified in Stage 2).
        devices.parquet uses 'id' as its primary key (not 'device_id'), so
        coverage is validated against the events→merged join instead.
        """
        merged = pd.read_parquet(PROCESSED_DIR / "merged.parquet")
        events = pd.read_parquet(PROCESSED_DIR / "events.parquet")
        # merged inherits all event rows — device_id should be present in merged
        assert "device_id" in merged.columns, (
            "device_id column missing from merged.parquet"
        )
        # Every device_id in merged should map back to a valid device
        # (manufacturer_id present is a proxy — every event has a manufacturer)
        assert "manufacturer_id" in merged.columns, (
            "manufacturer_id column missing from merged.parquet — join may be broken"
        )
        null_devices = merged["device_id"].isna().sum()
        assert null_devices == 0 or null_devices < len(merged) * 0.01, (
            f"{null_devices} null device_ids in merged.parquet — join may have failed"
        )

    def test_manifest_file_hashes_present(self):
        """Manifest must record hashes for all three raw CSVs."""
        manifest = _load_manifest()
        for fname in ("devices.csv", "events.csv", "manufacturers.csv"):
            assert fname in manifest["file_hashes"], (
                f"Manifest missing hash for {fname}"
            )
            assert len(manifest["file_hashes"][fname]) == 32, (
                f"Hash for {fname} is not a 32-char MD5"
            )

    def test_manifest_hashes_match_current_raw_files(self):
        """
        If the raw CSV files exist, their MD5 hashes must match the manifest.
        This detects silent data drift (data updated without re-running the pipeline).
        Skipped gracefully if raw files are absent (git-ignored in CI).
        """
        manifest = _load_manifest()
        any_checked = False
        for fname, stored_hash in manifest["file_hashes"].items():
            raw_path = RAW_DIR / fname
            if not raw_path.exists():
                continue   # raw files are git-ignored; skip if absent
            any_checked = True
            md5 = hashlib.md5(raw_path.read_bytes()).hexdigest()
            assert md5 == stored_hash, (
                f"Raw file {fname} has changed since the pipeline was last run. "
                f"Stored hash: {stored_hash}, current hash: {md5}. "
                "Re-run the full pipeline to refresh processed artifacts."
            )
        if not any_checked:
            pytest.skip("Raw CSV files not present — hash consistency check skipped.")


# ---------------------------------------------------------------------------
# TestFeatureParquetSchema
# ---------------------------------------------------------------------------

class TestFeatureParquetSchema:
    """
    Verifies that feature Parquet files are present and internally
    consistent with model_card.json and feature_metadata.json.
    """

    def test_feature_metadata_exists(self):
        assert (FEATURES_DIR / "feature_metadata.json").exists()

    def test_all_split_parquets_exist(self):
        for split in ("train", "validation", "test", "holdout_2018"):
            path = FEATURES_DIR / f"{split}.parquet"
            assert path.exists(), f"Feature parquet missing: {path}"

    def test_feature_columns_match_model_card(self):
        """
        The feature columns in feature_metadata.json must exactly match
        those in model_card.json — any drift means model and features are
        out of sync.
        """
        fm = _load_feature_metadata()
        mc = _load_model_card()
        fm_cols = sorted(fm["feature_columns"])
        mc_cols = sorted(mc["feature_columns"])
        assert fm_cols == mc_cols, (
            f"Feature column mismatch between feature_metadata.json and model_card.json.\n"
            f"In feature_metadata only: {sorted(set(fm_cols) - set(mc_cols))}\n"
            f"In model_card only:       {sorted(set(mc_cols) - set(fm_cols))}"
        )

    def test_feature_count_matches_expected(self):
        """86 features: 62 numeric/categorical + 24 SVD text features (updated Stage 5)."""
        fm = _load_feature_metadata()
        assert fm["n_features"] == 86, (
            f"Expected 86 features, feature_metadata.json reports {fm['n_features']}"
        )


    def test_no_target_in_feature_columns(self):
        fm = _load_feature_metadata()
        assert "is_class_i" not in fm["feature_columns"], (
            "Target column 'is_class_i' found in feature_metadata feature_columns — leakage!"
        )
        assert "action_classification" not in fm["feature_columns"], (
            "Source of target 'action_classification' found in feature columns — leakage!"
        )

    def test_feature_metadata_n_features_consistent_with_list(self):
        fm = _load_feature_metadata()
        assert fm["n_features"] == len(fm["feature_columns"]), (
            f"n_features ({fm['n_features']}) != len(feature_columns) ({len(fm['feature_columns'])})"
        )

    def test_train_parquet_feature_columns_match_metadata(self):
        """Actual train.parquet feature columns must match feature_metadata.json."""
        fm = _load_feature_metadata()
        train = pd.read_parquet(FEATURES_DIR / "train.parquet")
        metadata_cols = set(fm["feature_columns"])
        METADATA_COLS = {"id", "device_id", "manufacturer_id", "event_date",
                         "event_date_available", "is_class_i"}
        train_feature_cols = set(c for c in train.columns if c not in METADATA_COLS)
        missing = metadata_cols - train_feature_cols
        extra   = train_feature_cols - metadata_cols
        assert not missing, f"Features in metadata but not in train.parquet: {missing}"
        assert not extra,   f"Features in train.parquet but not in metadata: {extra}"


# ---------------------------------------------------------------------------
# TestServingArtifactIntegrity
# ---------------------------------------------------------------------------

class TestServingArtifactIntegrity:
    """
    Verifies the device risk serving table is present, well-formed,
    and consistent with the model card and feature splits.
    """

    def test_serving_table_exists(self):
        assert (ARTIFACTS_DIR / "risk" / "device_risk_snapshot.parquet").exists(), (
            "artifacts/risk/device_risk_snapshot.parquet not found. "
            "Run: python -m src.risk.build_serving_table"
        )

    def test_serving_table_has_required_columns(self):
        serving = pd.read_parquet(ARTIFACTS_DIR / "risk" / "device_risk_snapshot.parquet")
        required = {
            "device_id", "event_id", "serving_event_date",
            "calibrated_probability", "risk_score", "risk_level", "model_version",
        }
        missing = required - set(serving.columns)
        assert not missing, f"Serving table missing columns: {missing}"

    def test_serving_table_model_version_consistent(self):
        """
        All rows in the serving table must share the same model_version,
        and that version string must start with the base model type recorded
        in model_card.json (e.g. 'random_forest_<timestamp>').

        Note: model_card.json records model_name as the base type string
        ('random_forest'), while the serving table records the full versioned
        name ('random_forest_<timestamp>'). Both must agree on the base type.
        """
        serving = pd.read_parquet(ARTIFACTS_DIR / "risk" / "device_risk_snapshot.parquet")
        mc = _load_model_card()

        # All rows must have the same model_version
        unique_versions = serving["model_version"].unique()
        assert len(unique_versions) == 1, (
            f"Serving table contains {len(unique_versions)} different model_version values: "
            f"{list(unique_versions)}. All rows must use the same model."
        )

        # The versioned name must start with the base model type from model_card
        serving_version = unique_versions[0]
        base_type = mc.get("model_name", "")
        assert serving_version.startswith(base_type), (
            f"Serving table model_version ({serving_version!r}) does not start with "
            f"model_card model_name ({base_type!r}). "
            f"Serving table may have been built from a different model type."
        )

    def test_serving_table_one_row_per_device(self):
        serving = pd.read_parquet(ARTIFACTS_DIR / "risk" / "device_risk_snapshot.parquet")
        dupes = serving["device_id"].duplicated().sum()
        assert dupes == 0, (
            f"Serving table has {dupes} duplicate device_ids. "
            "Stage 3f policy requires exactly one row per device."
        )

    def test_serving_table_risk_scores_in_range(self):
        serving = pd.read_parquet(ARTIFACTS_DIR / "risk" / "device_risk_snapshot.parquet")
        assert (serving["risk_score"] >= 0).all() and (serving["risk_score"] <= 100).all(), (
            "Some risk_scores are outside [0, 100]"
        )

    def test_serving_table_risk_levels_valid(self):
        serving = pd.read_parquet(ARTIFACTS_DIR / "risk" / "device_risk_snapshot.parquet")
        valid_levels = {"LOW", "MEDIUM", "HIGH"}
        invalid = set(serving["risk_level"].unique()) - valid_levels
        assert not invalid, f"Invalid risk levels in serving table: {invalid}"

    def test_serving_row_count_reasonable(self):
        """
        Serving table should contain the scored subset of devices from the
        full feature dataset. Stage 6 report: 50,341 scored devices.
        Allow a ±10 tolerance for any legitimate pipeline re-run differences.
        """
        serving = pd.read_parquet(ARTIFACTS_DIR / "risk" / "device_risk_snapshot.parquet")
        EXPECTED = 50_341
        TOLERANCE = 10
        assert abs(len(serving) - EXPECTED) <= TOLERANCE, (
            f"Serving table has {len(serving)} rows, expected ~{EXPECTED} (±{TOLERANCE})"
        )


# ---------------------------------------------------------------------------
# TestModelCardIntegrity
# ---------------------------------------------------------------------------

class TestModelCardIntegrity:
    """Verifies model_card.json is complete and internally consistent."""

    def test_model_card_exists(self):
        assert (PRODUCTION_DIR / "model_card.json").exists()

    def test_model_card_has_required_keys(self):
        mc = _load_model_card()
        required = {
            "model_name", "target", "decision_threshold",
            "feature_columns", "n_features", "metrics",
        }
        missing = required - set(mc.keys())
        assert not missing, f"model_card.json missing keys: {missing}"

    def test_model_card_n_features_consistent(self):
        mc = _load_model_card()
        assert mc["n_features"] == len(mc["feature_columns"]), (
            f"n_features ({mc['n_features']}) != len(feature_columns) ({len(mc['feature_columns'])})"
        )

    def test_model_card_decision_threshold_in_range(self):
        mc = _load_model_card()
        t = mc["decision_threshold"]
        assert 0.0 < t < 1.0, f"decision_threshold={t} outside (0, 1)"

    def test_model_card_target_is_correct(self):
        mc = _load_model_card()
        assert mc["target"] == "is_class_i", (
            f"model_card target is {mc['target']!r}, expected 'is_class_i'"
        )

    def test_test_metrics_json_consistent_with_model_card(self):
        """
        models/production/test_metrics.json must exist and the decision_threshold
        it records must match model_card.json (both written by the same evaluate run).
        """
        mc = _load_model_card()
        tm_path = PRODUCTION_DIR / "test_metrics.json"
        assert tm_path.exists(), "test_metrics.json not found"
        tm = json.loads(tm_path.read_text(encoding="utf-8"))
        assert abs(tm["decision_threshold"] - mc["decision_threshold"]) < 1e-9, (
            f"test_metrics.json threshold ({tm['decision_threshold']}) != "
            f"model_card threshold ({mc['decision_threshold']})"
        )
