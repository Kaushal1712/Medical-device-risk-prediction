"""
Feature Engineering — Stage 4

Builds the feature matrix from processed Parquet files for
the approved severity target (is_class_i: Class I vs Class II/III).

Implements:
- Tier 1: Safe static device/manufacturer features
- Tier 2: Event type + temporal historical aggregates (strict event_date < T)
- Tier 3 architecture: Reason text excluded from primary model but hookable
- Temporal train/val/test split
- Explicit missing-value handling
- Leakage-safe categorical encoding

Run:  python -m src.features.build_features

Outputs:
    data/features/train.parquet
    data/features/validation.parquet
    data/features/test.parquet
    data/features/holdout_2018.parquet  (optional)
    data/features/feature_metadata.json

Does NOT train any ML model.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    MERGED_PARQUET,
    FEATURES_DATA_DIR,
    RANDOM_SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Target definition (approved Stage 3)
TARGET_COL = "action_classification"
TARGET_POSITIVE = "Class I"
TARGET_NEGATIVES = {"Class II", "Class III"}
TARGET_EXCLUDE_VALUES = {"Unclassified Correction", "Voluntary recall"}
TARGET_BINARY_NAME = "is_class_i"

# Temporal split boundaries
TRAIN_END = pd.Timestamp("2014-12-31")
VAL_START = pd.Timestamp("2015-01-01")
VAL_END = pd.Timestamp("2015-12-31")
TEST_START = pd.Timestamp("2016-01-01")
TEST_END = pd.Timestamp("2017-12-31")
HOLDOUT_START = pd.Timestamp("2018-01-01")
HOLDOUT_END = pd.Timestamp("2018-12-31")

# Prohibited columns — post-event or target-derived (MUST NEVER appear in features)
PROHIBITED_FEATURES = frozenset({
    "action", "action_summary", "action_classification", "action_level",
    "determined_cause", "status", "date_terminated", "date_updated",
    "target_audience",
    # Tier 3 (borderline leakage) — excluded from primary model
    "reason",
    # Non-feature metadata / identifiers
    "id", "slug", "uid", "uid_hash", "url", "authorities_link",
    "documents", "icij_notes", "data_notes", "number",
    "device_slug", "device_number", "device_distributed_to",
    "device_quantity_in_commerce",
    "mfr_slug", "mfr_comment", "mfr_representative", "mfr_address",
    # Database timestamps
    "created_at", "updated_at", "device_created_at", "device_updated_at",
    "mfr_created_at", "mfr_updated_at",
    # Raw date columns (already coalesced into event_date)
    "date", "date_initiated_by_firm", "date_posted", "create_date",
    # Auxiliary columns
    "event_date_source",
})

# Tier 1: Safe static features (device + manufacturer attributes)
TIER1_CATEGORICAL = [
    "device_classification",
    "device_risk_class",
    "device_implanted",
    "device_country",
    "mfr_parent_company",
    "mfr_source",
    "country",  # event country
]

# Tier 2a: Event-level features known at initiation
TIER2_EVENT_CATEGORICAL = [
    "type",  # event type (Recall, FSN, Safety Alert, etc.)
]

# Low-cardinality categoricals get one-hot encoded
LOW_CARDINALITY_THRESHOLD = 30

# Metadata columns kept for traceability (dropped before model input)
METADATA_COLS = frozenset({
    "id", "device_id", "manufacturer_id", "event_date", "event_date_available",
})


# =============================================================================
# Target creation
# =============================================================================

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target: is_class_i.
    Filters to labeled events only.
    """
    log.info("Creating target variable …")
    n_before = len(df)

    mask = df[TARGET_COL].notna()
    mask &= ~df[TARGET_COL].isin(TARGET_EXCLUDE_VALUES)
    df = df[mask].copy()

    log.info("  Filtered: %d → %d labeled events (dropped %d)",
             n_before, len(df), n_before - len(df))

    df[TARGET_BINARY_NAME] = (df[TARGET_COL] == TARGET_POSITIVE).astype(int)

    pos = df[TARGET_BINARY_NAME].sum()
    neg = len(df) - pos
    log.info("  Target: %d positive (%.1f%%) | %d negative (%.1f%%)",
             pos, pos / len(df) * 100, neg, neg / len(df) * 100)

    return df


# =============================================================================
# Temporal split
# =============================================================================

def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split into train/val/test/holdout by event_date.
    Events without dates are excluded (cannot be temporally placed).
    """
    log.info("Applying temporal split …")

    has_date = df["event_date"].notna()
    no_date = (~has_date).sum()
    if no_date > 0:
        log.warning("  %d labeled events have no event_date — excluded", no_date)
    df_dated = df[has_date].copy()

    splits = {
        "train": df_dated[df_dated["event_date"] <= TRAIN_END].copy(),
        "validation": df_dated[
            (df_dated["event_date"] >= VAL_START) & (df_dated["event_date"] <= VAL_END)
        ].copy(),
        "test": df_dated[
            (df_dated["event_date"] >= TEST_START) & (df_dated["event_date"] <= TEST_END)
        ].copy(),
        "holdout_2018": df_dated[
            (df_dated["event_date"] >= HOLDOUT_START) & (df_dated["event_date"] <= HOLDOUT_END)
        ].copy(),
    }

    for name, split_df in splits.items():
        pos = split_df[TARGET_BINARY_NAME].sum()
        neg = len(split_df) - pos
        log.info("  %s: %d events (pos=%d [%.1f%%], neg=%d)",
                 name, len(split_df), pos, pos / max(1, len(split_df)) * 100, neg)

    return splits


# =============================================================================
# Tier 2b: Temporal historical features — VECTORIZED
# =============================================================================

def _cumcount_before(
    events_sorted: pd.DataFrame,
    group_col: str,
    count_cols: dict[str, str],
) -> pd.DataFrame:
    """
    For each event, compute cumulative counts of prior events (same group,
    strictly earlier date) using a vectorized merge_asof approach.

    events_sorted: must be sorted by event_date.
    group_col: column to group by (e.g. 'device_id', 'manufacturer_id').
    count_cols: dict of {output_col_name: source_col_or_flag}.
        Special values:
        - '__COUNT__': count all events
        - 'col_name==value': count events where col_name == value
        - 'col_name~=pattern': count events where col_name contains pattern (case-insensitive)
    """
    es = events_sorted.copy()

    # Assign row number within each group (sorted by date)
    es["_grp_row"] = es.groupby(group_col).cumcount()

    # For each count_col, compute a boolean flag, then cumulative sum
    for out_col, spec in count_cols.items():
        if spec == "__COUNT__":
            es[f"_flag_{out_col}"] = 1
        elif "==" in spec:
            col_name, value = spec.split("==", 1)
            es[f"_flag_{out_col}"] = (es[col_name] == value).fillna(False).astype(int)
        elif "~=" in spec:
            col_name, pattern = spec.split("~=", 1)
            es[f"_flag_{out_col}"] = (
                es[col_name].str.contains(pattern, case=False, na=False).astype(int)
            )

        # Cumulative sum within group, then shift by 1 to exclude current row
        es[f"_cum_{out_col}"] = es.groupby(group_col)[f"_flag_{out_col}"].cumsum()
        # Shift: the count for the current row should be cumsum BEFORE this row
        es[f"_cum_{out_col}"] = (
            es[f"_cum_{out_col}"] - es[f"_flag_{out_col}"]
        )

    # Now handle same-day exclusion:
    # If multiple events have the same (group_col, event_date), the cumsum
    # above already includes same-day predecessors (sorted order is arbitrary
    # within the same date). We need to subtract same-day-and-earlier-in-sort-order
    # contributions. The simplest correct approach: for each group+date combo,
    # the historical count should be the cumsum UP TO (but not including)
    # the first event on that date.

    # Group by (group_col, event_date) and take the min cumsum in each group
    # That min is the count of events strictly before this date
    for out_col in count_cols:
        col = f"_cum_{out_col}"
        min_cum = es.groupby([group_col, "event_date"])[col].transform("min")
        es[out_col] = min_cum

    # Clean up temp columns
    drop_cols = [c for c in es.columns if c.startswith("_")]
    es = es.drop(columns=drop_cols)

    return es


def build_historical_features(df: pd.DataFrame, all_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical aggregate features for each event using ONLY events
    that occurred STRICTLY BEFORE the current event's event_date.

    Uses the full merged table (including unlabeled events) as the history source.
    Same-day events are EXCLUDED to prevent leakage.
    """
    log.info("Building temporal historical features (vectorized) …")

    # Prepare historical source: all events with dates
    hist = all_merged[all_merged["event_date"].notna()].copy()
    hist = hist[["id", "device_id", "manufacturer_id", "event_date",
                 "action_classification", "type", "device_classification"]].copy()
    hist = hist.sort_values("event_date").reset_index(drop=True)

    log.info("  Historical source: %d dated events", len(hist))

    # ── Device-level historical counts ──
    log.info("  Computing device-level historical counts …")
    hist = _cumcount_before(hist, "device_id", {
        "hist_device_event_count": "__COUNT__",
        "hist_device_class_i_count": "action_classification==Class I",
        "hist_device_recall_count": "type~=Recall",
    })

    # ── Manufacturer-level historical counts ──
    log.info("  Computing manufacturer-level historical counts …")
    hist = _cumcount_before(hist, "manufacturer_id", {
        "hist_mfr_event_count": "__COUNT__",
        "hist_mfr_class_i_count": "action_classification==Class I",
        "hist_mfr_recall_count": "type~=Recall",
    })

    # ── Category-level historical counts ──
    log.info("  Computing category-level historical counts …")
    # Need to handle NaN device_classification — fill with sentinel for groupby
    hist["_cat_group"] = hist["device_classification"].fillna("__NO_CATEGORY__")
    hist = _cumcount_before(hist, "_cat_group", {
        "hist_category_event_count": "__COUNT__",
        "hist_category_class_i_count": "action_classification==Class I",
    })
    # Zero out category history for events without device_classification
    no_cat = hist["device_classification"].isna()
    hist.loc[no_cat, "hist_category_event_count"] = 0
    hist.loc[no_cat, "hist_category_class_i_count"] = 0
    # (_cat_group already dropped by _cumcount_before's cleanup of _-prefixed columns)

    # ── Derived rate features ──
    hist["hist_mfr_severity_rate"] = np.where(
        hist["hist_mfr_event_count"] > 0,
        hist["hist_mfr_class_i_count"] / hist["hist_mfr_event_count"],
        np.nan,
    )
    hist["hist_category_severity_rate"] = np.where(
        hist["hist_category_event_count"] > 0,
        hist["hist_category_class_i_count"] / hist["hist_category_event_count"],
        np.nan,
    )

    # ── Merge back to labeled events ──
    hist_features = [c for c in hist.columns if c.startswith("hist_")]
    hist_for_join = hist[["id"] + hist_features].copy()

    n_before = len(df)
    df = df.merge(hist_for_join, on="id", how="left")
    assert len(df) == n_before, f"Row count changed after history merge: {n_before} → {len(df)}"

    log.info("  Historical features added: %d columns", len(hist_features))
    return df


# =============================================================================
# Categorical encoding
# =============================================================================

def encode_categoricals(
    train: pd.DataFrame,
    other_splits: dict[str, pd.DataFrame],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """
    Encode categorical features.
    - Low cardinality: one-hot encoding
    - High cardinality: frequency encoding (from train set only)

    Missing → '__MISSING__', unseen at test time → '__UNKNOWN__'.
    """
    log.info("Encoding categoricals …")
    encoding_meta = {}

    for col in cat_cols:
        if col not in train.columns:
            log.warning("  Column %s not found, skipping", col)
            continue

        # Fill missing with explicit marker
        train[col] = train[col].fillna("__MISSING__").astype(str)
        for split_df in other_splits.values():
            split_df[col] = split_df[col].fillna("__MISSING__").astype(str)

        # Build train vocabulary
        train_vocab = set(train[col].unique())
        nunique = len(train_vocab)

        # Map unseen categories in non-train splits
        for split_df in other_splits.values():
            unseen = ~split_df[col].isin(train_vocab)
            if unseen.any():
                split_df.loc[unseen, col] = "__UNKNOWN__"
        # Ensure __UNKNOWN__ is in vocab for consistent columns
        train_vocab.add("__UNKNOWN__")

        if nunique <= LOW_CARDINALITY_THRESHOLD:
            # One-hot encoding
            all_cats = sorted(train_vocab)

            for split_name, split_df in [("train", train)] + list(other_splits.items()):
                for cat in all_cats:
                    dummy_col = f"{col}_{cat}"
                    split_df[dummy_col] = (split_df[col] == cat).astype(int)
                split_df.drop(columns=[col], inplace=True)

            encoding_meta[col] = {
                "method": "one_hot",
                "n_unique_train": nunique,
                "categories": all_cats,
            }
            log.info("  %s: one-hot (%d categories)", col, nunique)
        else:
            # Frequency encoding
            freq = train[col].value_counts(normalize=True).to_dict()
            freq_col = f"{col}_freq"
            train[freq_col] = train[col].map(freq).astype(float)
            for split_df in other_splits.values():
                split_df[freq_col] = split_df[col].map(freq).fillna(0.0).astype(float)

            train.drop(columns=[col], inplace=True)
            for split_df in other_splits.values():
                split_df.drop(columns=[col], inplace=True)

            encoding_meta[col] = {
                "method": "frequency",
                "n_unique_train": nunique,
            }
            log.info("  %s: frequency encoded (%d categories)", col, nunique)

    return train, other_splits, encoding_meta


# =============================================================================
# Feature selection and cleaning
# =============================================================================

def select_and_clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop prohibited columns, keep only approved features + target + metadata.
    """
    log.info("Selecting features …")
    keep_cols = set()

    # Target
    keep_cols.add(TARGET_BINARY_NAME)

    # Metadata (kept for traceability, dropped before model input)
    keep_cols.update(METADATA_COLS)

    # Tier 1 categoricals
    keep_cols.update(TIER1_CATEGORICAL)

    # Tier 2a event categoricals
    keep_cols.update(TIER2_EVENT_CATEGORICAL)

    # Tier 2b historical features
    hist_cols = [c for c in df.columns if c.startswith("hist_")]
    keep_cols.update(hist_cols)

    # Derived numerical features from text length
    if "device_description" in df.columns:
        df["device_description_len"] = df["device_description"].str.len().fillna(0).astype(int)
        keep_cols.add("device_description_len")
    if "device_name" in df.columns:
        df["device_name_len"] = df["device_name"].str.len().fillna(0).astype(int)
        keep_cols.add("device_name_len")

    # Verify no prohibited columns in FEATURE set (metadata cols are not features)
    feature_only = keep_cols - METADATA_COLS - {TARGET_BINARY_NAME}
    for col in PROHIBITED_FEATURES:
        if col in feature_only:
            raise ValueError(f"LEAKAGE: Prohibited column '{col}' in feature set!")

    available = [c for c in df.columns if c in keep_cols]
    df = df[available].copy()

    log.info("  Selected %d columns (incl. target + metadata)", len(df.columns))
    return df


# =============================================================================
# Missing value handling
# =============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit missing-value handling:
    - Categoricals: '__MISSING__' (handled in encode_categoricals)
    - Historical counts: NaN → 0 (no history = zero)
    - Historical rates: NaN → 0 with missing indicator
    - Device attribute missingness indicators
    """
    # Missing indicators for high-missingness device attributes
    for col in ["device_classification", "device_risk_class", "device_implanted"]:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # Historical count features: NaN → 0
    hist_count_cols = [c for c in df.columns if c.startswith("hist_") and "rate" not in c]
    for col in hist_count_cols:
        df[col] = df[col].fillna(0).astype(int)

    # Historical rate features: NaN → 0 with indicator
    hist_rate_cols = [c for c in df.columns if "rate" in c and c.startswith("hist_")]
    for col in hist_rate_cols:
        df[f"{col}_available"] = df[col].notna().astype(int)
        df[col] = df[col].fillna(0.0)

    return df


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    log.info("=" * 70)
    log.info("STAGE 4 — Feature Engineering")
    log.info("=" * 70)

    start_time = time.time()

    # ── Load ──
    log.info("Loading processed data …")
    merged = pd.read_parquet(MERGED_PARQUET)
    log.info("  merged.parquet: %d rows × %d cols", *merged.shape)

    # ── Target ──
    labeled = create_target(merged)

    # ── Historical features (uses full merged as history source) ──
    labeled = build_historical_features(labeled, merged)

    # ── Feature selection ──
    labeled = select_and_clean_features(labeled)

    # ── Temporal split ──
    splits = temporal_split(labeled)

    # ── Missing values ──
    for name in splits:
        splits[name] = handle_missing_values(splits[name])

    # ── Categorical encoding ──
    cat_cols = [c for c in TIER1_CATEGORICAL + TIER2_EVENT_CATEGORICAL
                if c in splits["train"].columns]
    train = splits.pop("train")
    train, splits, encoding_meta = encode_categoricals(train, splits, cat_cols)

    # Put train back
    all_splits = {"train": train}
    all_splits.update(splits)

    # ── Feature inventory ──
    feature_cols = sorted([
        c for c in all_splits["train"].columns
        if c != TARGET_BINARY_NAME and c not in METADATA_COLS
    ])

    log.info("=" * 70)
    log.info("FEATURE SUMMARY — %d features", len(feature_cols))
    log.info("=" * 70)
    for fc in feature_cols:
        log.info("  %s", fc)

    # ── Persist ──
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, split_df in all_splits.items():
        path = FEATURES_DATA_DIR / f"{name}.parquet"
        split_df.to_parquet(path, index=False, engine="pyarrow")
        size_mb = path.stat().st_size / (1024 * 1024)
        log.info("  Saved %s: %d rows × %d cols (%.1f MB)",
                 path.name, len(split_df), len(split_df.columns), size_mb)

    # ── Feature metadata ──
    metadata = {
        "pipeline_version": "4.0.0",
        "target": TARGET_BINARY_NAME,
        "target_positive": TARGET_POSITIVE,
        "target_negatives": sorted(list(TARGET_NEGATIVES)),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "metadata_columns": sorted(list(METADATA_COLS)),
        "encoding": encoding_meta,
        "splits": {
            name: {
                "rows": len(split_df),
                "positive": int(split_df[TARGET_BINARY_NAME].sum()),
                "negative": int(len(split_df) - split_df[TARGET_BINARY_NAME].sum()),
                "positive_rate_pct": round(split_df[TARGET_BINARY_NAME].mean() * 100, 2),
            }
            for name, split_df in all_splits.items()
        },
        "temporal_boundaries": {
            "train": f"<= {TRAIN_END.strftime('%Y-%m-%d')}",
            "validation": f"{VAL_START.strftime('%Y-%m-%d')} to {VAL_END.strftime('%Y-%m-%d')}",
            "test": f"{TEST_START.strftime('%Y-%m-%d')} to {TEST_END.strftime('%Y-%m-%d')}",
            "holdout_2018": f"{HOLDOUT_START.strftime('%Y-%m-%d')} to {HOLDOUT_END.strftime('%Y-%m-%d')}",
        },
        "prohibited_features": sorted(list(PROHIBITED_FEATURES)),
    }
    meta_path = FEATURES_DATA_DIR / "feature_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("  Metadata → %s", meta_path)

    elapsed = time.time() - start_time
    log.info("=" * 70)
    log.info("STAGE 4 COMPLETE in %.1f seconds", elapsed)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
