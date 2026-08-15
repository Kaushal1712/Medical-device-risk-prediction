"""
Stage 4 — Leakage Validation Tests

Verifies that the feature-engineering pipeline produces outputs
free from temporal leakage, target leakage, and prohibited-feature
contamination.

Run:  python -m pytest tests/features/test_leakage.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

FEATURES_DIR = ROOT / "data" / "features"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def meta():
    with open(FEATURES_DIR / "feature_metadata.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def train():
    return pd.read_parquet(FEATURES_DIR / "train.parquet")


@pytest.fixture(scope="module")
def validation():
    return pd.read_parquet(FEATURES_DIR / "validation.parquet")


@pytest.fixture(scope="module")
def test_set():
    return pd.read_parquet(FEATURES_DIR / "test.parquet")


@pytest.fixture(scope="module")
def holdout():
    return pd.read_parquet(FEATURES_DIR / "holdout_2018.parquet")


@pytest.fixture(scope="module")
def merged():
    return pd.read_parquet(PROCESSED_DIR / "merged.parquet")


# ── 1. Target excluded from features ─────────────────────────────────────────

class TestTargetExclusion:
    def test_target_not_in_feature_columns(self, meta):
        """is_class_i must not appear in the feature column list."""
        assert meta["target"] == "is_class_i"
        assert "is_class_i" not in meta["feature_columns"]

    def test_action_classification_not_in_features(self, meta):
        """action_classification (source of target) must not be a feature."""
        assert "action_classification" not in meta["feature_columns"]
        assert "action_classification" in meta["prohibited_features"]


# ── 2. Prohibited post-event fields excluded ──────────────────────────────────

class TestProhibitedFields:
    PROHIBITED_POST_EVENT = [
        "action", "action_summary", "determined_cause", "status",
        "date_terminated", "action_classification", "action_level",
        "reason", "target_audience",
    ]

    @pytest.mark.parametrize("col", PROHIBITED_POST_EVENT)
    def test_prohibited_field_not_in_features(self, meta, col):
        """Post-event / target-derived fields must not be features."""
        assert col not in meta["feature_columns"], (
            f"Prohibited column '{col}' found in feature_columns!"
        )

    @pytest.mark.parametrize("col", PROHIBITED_POST_EVENT)
    def test_prohibited_field_not_in_train_data(self, train, col):
        """Post-event fields must not exist in the train DataFrame at all."""
        assert col not in train.columns, (
            f"Prohibited column '{col}' found in train.parquet columns!"
        )


# ── 3. Historical features use strict event_date < T ─────────────────────────

class TestHistoricalLeakage:
    def test_hist_device_count_is_zero_for_first_events(self, merged, train):
        """
        Devices whose first-ever event is in the train set should have
        hist_device_event_count == 0 for that first event.
        """
        # Find the earliest event per device in the full dataset
        dated = merged[merged["event_date"].notna()].copy()
        first_event = dated.groupby("device_id")["event_date"].min().reset_index()
        first_event.columns = ["device_id", "first_date"]

        # Merge with train
        check = train.merge(first_event, on="device_id", how="inner")

        # Events where event_date == first_date should have 0 device history
        is_first = check["event_date"] == check["first_date"]
        first_events = check[is_first]

        if len(first_events) > 0:
            assert (first_events["hist_device_event_count"] == 0).all(), (
                "Some first-ever device events have hist_device_event_count > 0 — temporal leakage!"
            )

    def test_hist_mfr_count_nonnegative(self, train):
        """Historical manufacturer counts must be non-negative."""
        assert (train["hist_mfr_event_count"] >= 0).all()
        assert (train["hist_mfr_class_i_count"] >= 0).all()
        assert (train["hist_mfr_recall_count"] >= 0).all()

    def test_hist_class_i_leq_event_count(self, train):
        """Class I counts cannot exceed total event counts."""
        assert (train["hist_device_class_i_count"] <= train["hist_device_event_count"]).all()
        assert (train["hist_mfr_class_i_count"] <= train["hist_mfr_event_count"]).all()


# ── 4. Same-day events do not contribute ──────────────────────────────────────

class TestSameDayExclusion:
    def test_same_day_device_events_get_same_history(self, merged):
        """
        If a device has multiple events on the same date, they should all have
        the same hist_device_event_count (computed from the historical features
        pipeline using the 'min' approach).
        """
        from src.features.build_features import _cumcount_before

        # Create a small test scenario
        test_data = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "device_id": [100, 100, 100, 200, 200],
            "event_date": pd.to_datetime([
                "2010-01-01", "2010-01-01", "2010-06-01",
                "2010-03-01", "2010-03-01",
            ]),
            "action_classification": ["Class I", "Class II", "Class II", "Class I", "Class III"],
            "type": ["Recall", "Recall", "Recall", "Safety alert", "Recall"],
        })
        test_data = test_data.sort_values("event_date").reset_index(drop=True)

        result = _cumcount_before(test_data, "device_id", {
            "hist_device_event_count": "__COUNT__",
        })

        # Events 1 and 2 are on the same day (2010-01-01) for device 100
        # They should BOTH have hist_device_event_count == 0 (no prior events)
        device_100_jan = result[
            (result["device_id"] == 100) &
            (result["event_date"] == pd.Timestamp("2010-01-01"))
        ]
        assert (device_100_jan["hist_device_event_count"] == 0).all(), (
            "Same-day events should have identical (and zero) history for first day"
        )

        # Event 3 is on 2010-06-01 for device 100 — should count 2 prior events
        device_100_jun = result[
            (result["device_id"] == 100) &
            (result["event_date"] == pd.Timestamp("2010-06-01"))
        ]
        assert (device_100_jun["hist_device_event_count"] == 2).all()

        # Device 200: both events on same day — should both have 0
        device_200 = result[result["device_id"] == 200]
        assert (device_200["hist_device_event_count"] == 0).all()


# ── 5. No future information in historical features ──────────────────────────

class TestNoFutureLeakage:
    def test_no_future_events_in_train_history(self, train, merged):
        """
        For each train event, verify that the historical manufacturer event count
        does not exceed the number of events that actually occurred before that
        event's date in the full dataset.
        """
        dated = merged[merged["event_date"].notna()].copy()

        # Sample 200 train events for performance
        sample = train.sample(min(200, len(train)), random_state=42)

        for _, row in sample.iterrows():
            mfr_id = row["manufacturer_id"]
            event_date = row["event_date"]
            hist_count = row["hist_mfr_event_count"]

            # Count actual events strictly before this date for this manufacturer
            actual = len(dated[
                (dated["manufacturer_id"] == mfr_id) &
                (dated["event_date"] < event_date)
            ])

            assert hist_count <= actual, (
                f"Event {row['id']}: hist_mfr_event_count={hist_count} > "
                f"actual prior events={actual} — future leakage!"
            )


# ── 6. Temporal split boundaries respected ────────────────────────────────────

class TestTemporalSplit:
    def test_train_before_2015(self, train):
        """All train events must have event_date <= 2014-12-31."""
        assert train["event_date"].max() <= pd.Timestamp("2014-12-31")

    def test_validation_in_2015(self, validation):
        """All validation events must be in 2015."""
        assert validation["event_date"].min() >= pd.Timestamp("2015-01-01")
        assert validation["event_date"].max() <= pd.Timestamp("2015-12-31")

    def test_test_in_2016_2017(self, test_set):
        """All test events must be in 2016-2017."""
        assert test_set["event_date"].min() >= pd.Timestamp("2016-01-01")
        assert test_set["event_date"].max() <= pd.Timestamp("2017-12-31")

    def test_holdout_in_2018(self, holdout):
        """All holdout events must be in 2018."""
        assert holdout["event_date"].min() >= pd.Timestamp("2018-01-01")
        assert holdout["event_date"].max() <= pd.Timestamp("2018-12-31")

    def test_no_overlap_between_splits(self, train, validation, test_set, holdout):
        """No event ID should appear in more than one split."""
        train_ids = set(train["id"].values)
        val_ids = set(validation["id"].values)
        test_ids = set(test_set["id"].values)
        holdout_ids = set(holdout["id"].values)

        assert len(train_ids & val_ids) == 0, "Train/validation overlap!"
        assert len(train_ids & test_ids) == 0, "Train/test overlap!"
        assert len(val_ids & test_ids) == 0, "Validation/test overlap!"
        assert len(train_ids & holdout_ids) == 0, "Train/holdout overlap!"


# ── 7. Row count traceability ─────────────────────────────────────────────────

class TestRowCounts:
    def test_total_labeled_events(self, train, validation, test_set, holdout):
        """Total across splits should match expected labeled count (minus undated)."""
        total = len(train) + len(validation) + len(test_set) + len(holdout)
        # 52946 labeled - 147 undated - any outside 2019+ = should be around 52799
        assert total > 50000, f"Unexpectedly few events: {total}"
        assert total <= 52946, f"More events than labeled total: {total}"

    def test_positive_rate_reasonable(self, train, validation, test_set):
        """Positive rate should be between 3% and 15% in each split."""
        for name, df in [("train", train), ("val", validation), ("test", test_set)]:
            rate = df["is_class_i"].mean()
            assert 0.03 < rate < 0.15, (
                f"{name} positive rate {rate:.3f} outside expected range"
            )
