"""
Data Engineering Pipeline — Stage 2

Loads the three raw CSVs, cleans them using strategies justified by Stage 1,
validates the confirmed join keys, constructs the coalesced event date,
and persists cleaned Parquet files.

Run:  python -m src.data.pipeline

Outputs:
    data/processed/devices.parquet
    data/processed/events.parquet
    data/processed/manufacturers.parquet
    data/processed/merged.parquet
    data/processed/_manifest.json

Does NOT define a prediction target, perform feature engineering, or train any model.
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DEVICES_CSV,
    EVENTS_CSV,
    MANUFACTURERS_CSV,
    PROCESSED_DATA_DIR,
    MANIFEST_PATH,
    PIPELINE_VERSION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# Constants — plausible date range boundaries (from Stage 1 analysis)
# =============================================================================
PLAUSIBLE_DATE_MIN = pd.Timestamp("1990-01-01")
PLAUSIBLE_DATE_MAX = pd.Timestamp("2025-12-31")


# =============================================================================
# Manifest / cache-invalidation helpers
# =============================================================================

def _md5_file(path: Path) -> str:
    """Compute MD5 hash of a file's contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_manifest(csv_paths: list[Path]) -> bool:
    """
    Return True if all outputs exist and the manifest hashes + pipeline version
    match the current raw files, meaning we can skip reprocessing.
    """
    if not MANIFEST_PATH.exists():
        log.info("No manifest found — will process from scratch.")
        return False

    manifest = json.loads(MANIFEST_PATH.read_text())

    # Check pipeline version
    if manifest.get("pipeline_version") != PIPELINE_VERSION:
        log.info(
            "Pipeline version changed (%s → %s) — reprocessing.",
            manifest.get("pipeline_version"),
            PIPELINE_VERSION,
        )
        return False

    # Check each raw file hash
    for path in csv_paths:
        stored_hash = manifest.get("file_hashes", {}).get(path.name)
        if stored_hash is None:
            log.info("Manifest missing hash for %s — reprocessing.", path.name)
            return False
        current_hash = _md5_file(path)
        if current_hash != stored_hash:
            log.info("Hash mismatch for %s — reprocessing.", path.name)
            return False

    # Check output files exist
    expected_outputs = ["devices.parquet", "events.parquet", "manufacturers.parquet", "merged.parquet"]
    for fname in expected_outputs:
        if not (PROCESSED_DATA_DIR / fname).exists():
            log.info("Output file %s missing — reprocessing.", fname)
            return False

    log.info("Manifest valid, all outputs exist — skipping reprocessing.")
    return True


def _write_manifest(csv_paths: list[Path], run_stats: dict) -> None:
    """Write the manifest JSON with file hashes, pipeline version, and run stats."""
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_hashes": {p.name: _md5_file(p) for p in csv_paths},
        "run_stats": run_stats,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Manifest written to %s", MANIFEST_PATH)


# =============================================================================
# Loading
# =============================================================================

def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV, failing loudly if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required raw data file missing: {path}. "
            f"Place the CSV files in {path.parent}/ and re-run."
        )
    log.info("Loading %s …", path.name)
    df = pd.read_csv(path, low_memory=False)
    log.info("  → %d rows × %d columns", *df.shape)
    return df


# =============================================================================
# Cleaning — manufacturers
# =============================================================================

def clean_manufacturers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean manufacturers.csv.
    
    Stage 1 findings:
    - 31,827 rows, 10 columns, 0 duplicates
    - PK: id (fully unique)
    - High missingness: representative (97.68%), comment (91.17%), address (81.25%)
    - parent_company: 18.73% missing, 3,501 unique values
    - name: 0.12% missing (37 rows)
    - source: 1 null
    
    Strategy:
    - Trim whitespace on string columns
    - Parse created_at/updated_at as datetime (database timestamps only)
    - Do NOT fill missing values — high missingness is the reality of the data
    """
    log.info("Cleaning manufacturers …")
    n_in = len(df)

    # Trim whitespace on string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype("string").str.strip()
        # Replace empty strings with NA after stripping
        df[col] = df[col].replace("", pd.NA)

    # Parse database timestamps
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", format="mixed")
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", format="mixed")

    log.info("  Manufacturers cleaned: %d rows in → %d rows out (no rows dropped)", n_in, len(df))
    return df


# =============================================================================
# Cleaning — devices
# =============================================================================

def clean_devices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean devices.csv.
    
    Stage 1 findings:
    - 118,249 rows, 15 columns, 0 duplicates
    - PK: id (fully unique)
    - FK: manufacturer_id → manufacturers.id
    - High missingness: quantity_in_commerce (73%), risk_class (72.14%),
      distributed_to (71.56%), implanted (70.49%), classification (69.89%),
      code (69.37%), number (46.05%), description (24.22%)
    - risk_class has both numeric (1,2,3) and Roman (II) and text (Unclassified, HDE)
    - classification: 16 FDA device categories
    - implanted: YES/NO binary
    - country: 46 countries, ISO-3 codes
    
    Strategy:
    - Trim whitespace on string columns
    - Normalize risk_class: merge Roman numeral duplicates (II→2)
    - Parse created_at/updated_at as datetime
    - Do NOT fill missing values
    """
    log.info("Cleaning devices …")
    n_in = len(df)

    # Trim whitespace on string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # Normalize risk_class: Stage 1 found both "II" (2 rows) alongside "2" (24,693 rows)
    # Map the Roman numeral variant to its numeric equivalent
    risk_class_map = {"II": "2"}  # Only this duplicate was observed in Stage 1
    df["risk_class"] = df["risk_class"].replace(risk_class_map)
    log.info("  risk_class normalized: mapped 'II' → '2'")

    # Normalize implanted to uppercase (already YES/NO, but ensure consistency)
    df["implanted"] = df["implanted"].str.upper()

    # Parse database timestamps
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", format="mixed")
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", format="mixed")

    log.info("  Devices cleaned: %d rows in → %d rows out (no rows dropped)", n_in, len(df))
    return df


# =============================================================================
# Cleaning — events
# =============================================================================

def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean events.csv.
    
    Stage 1 findings:
    - 124,969 rows, 30 columns, 0 duplicates
    - PK: id (fully unique)
    - FK: device_id → devices.id (100% match)
    - type column: 7 event types, all safety/failure-related
    - action_classification: 11 values with duplicated naming (Class 1/I, Class 2/II, Class 3/III)
    - Date columns are country-specific and mutually exclusive:
      * date and date_initiated_by_firm never co-occur
      * 6 outlier dates in `date`, 5 in `date_posted` (11 total)
      * SWE (3,469 events) has no dates at all
    - Free-text: reason (52.6%), action (50%), action_summary (23.3%)
    
    Strategy:
    - Trim whitespace on string columns
    - Normalize action_classification: standardize Class/Roman numeral variants
    - Parse all date columns individually
    - Construct coalesced event_date: date_initiated_by_firm > date > date_posted > create_date
    - Flag implausible dates (year < 1990 or > 2025) — set to NaT with logging
    - Add event_date_available boolean flag
    - Add event_date_source column tracking which raw column provided the date
    - Do NOT drop undateable events — preserve them with flag
    - Do NOT fill missing values
    """
    log.info("Cleaning events …")
    n_in = len(df)

    # Trim whitespace on string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # ── Normalize action_classification ──
    # Stage 1 found duplicated naming conventions across countries:
    # Class 1 / I / Class I → all represent severity class 1
    # Class 2 / II / Class II → severity class 2
    # Class 3 / III / Class III → severity class 3
    action_class_map = {
        "Class 1": "Class I",
        "I": "Class I",
        "Class 2": "Class II",
        "II": "Class II",
        "Class 3": "Class III",
        "III": "Class III",
    }
    original_counts = df["action_classification"].value_counts(dropna=False)
    df["action_classification"] = df["action_classification"].replace(action_class_map)
    new_counts = df["action_classification"].value_counts(dropna=False)
    log.info("  action_classification normalized:")
    for val in ["Class I", "Class II", "Class III"]:
        if val in new_counts.index:
            log.info("    %s: %d values", val, new_counts[val])

    # ── Parse date columns ──
    date_parse_cols = [
        "date",
        "date_initiated_by_firm",
        "date_posted",
        "date_terminated",
        "date_updated",
        "create_date",
    ]

    parsed_dates = {}
    for col in date_parse_cols:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            valid_count = parsed.notna().sum()
            log.info("  Parsed %s: %d valid dates", col, valid_count)
            parsed_dates[col] = parsed

    # ── Identify and handle implausible date outliers ──
    total_outliers = 0
    for col, parsed in parsed_dates.items():
        outlier_mask = parsed.notna() & (
            (parsed < PLAUSIBLE_DATE_MIN) | (parsed > PLAUSIBLE_DATE_MAX)
        )
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            # Log each outlier
            outlier_dates = parsed[outlier_mask]
            log.warning(
                "  %s: %d implausible date outlier(s) found — setting to NaT:",
                col,
                n_outliers,
            )
            for idx in outlier_dates.index:
                log.warning("    Row %d: %s → NaT", idx, outlier_dates[idx])
            # Set outliers to NaT
            parsed_dates[col] = parsed.where(~outlier_mask)
            total_outliers += n_outliers

    log.info(
        "  Total implausible date outliers handled: %d (set to NaT, rows preserved)",
        total_outliers,
    )

    # ── Store cleaned individual date columns ──
    for col, parsed in parsed_dates.items():
        df[col] = parsed

    # ── Construct coalesced event_date ──
    # Priority: date_initiated_by_firm > date > date_posted > create_date
    # (Stage 1 confirmed date and date_initiated_by_firm are mutually exclusive,
    #  so coalescing is straightforward — no conflict resolution needed)
    df["event_date"] = (
        parsed_dates["date_initiated_by_firm"]
        .fillna(parsed_dates["date"])
        .fillna(parsed_dates["date_posted"])
        .fillna(parsed_dates["create_date"])
    )

    # ── Track which raw column provided the coalesced date ──
    # This is important for traceability
    event_date_source = pd.Series(pd.NA, index=df.index, dtype="string")
    # Apply in reverse priority order so higher-priority sources overwrite
    for col in ["create_date", "date_posted", "date", "date_initiated_by_firm"]:
        mask = parsed_dates[col].notna()
        event_date_source = event_date_source.where(~mask, col)
    df["event_date_source"] = event_date_source

    # ── Add date-availability flag ──
    df["event_date_available"] = df["event_date"].notna()

    dated_count = df["event_date_available"].sum()
    undated_count = (~df["event_date_available"]).sum()
    log.info(
        "  Coalesced event_date: %d dated (%.1f%%) | %d undateable (%.1f%%) — undateable rows PRESERVED",
        dated_count,
        dated_count / len(df) * 100,
        undated_count,
        undated_count / len(df) * 100,
    )

    # Source breakdown
    source_counts = df["event_date_source"].value_counts(dropna=False)
    log.info("  event_date_source breakdown:")
    for src, cnt in source_counts.items():
        log.info("    %s: %d", src if pd.notna(src) else "NO_DATE", cnt)

    # Parse database timestamps
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", format="mixed")
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", format="mixed")

    # Normalize timezone-naive (Stage 1 showed these are all UTC-stamped database load dates)
    for col in ["created_at", "updated_at"]:
        if df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)

    log.info("  Events cleaned: %d rows in → %d rows out (no rows dropped)", n_in, len(df))
    return df


# =============================================================================
# Join validation
# =============================================================================

def validate_and_join(
    devices: pd.DataFrame,
    events: pd.DataFrame,
    manufacturers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the confirmed join keys and build the merged DataFrame.

    Join path (from Stage 1):
        manufacturers.id → devices.manufacturer_id
        devices.id → events.device_id

    Final merged: events LEFT JOIN devices LEFT JOIN manufacturers
    (left join to preserve all events, even if a device or manufacturer is missing)
    """
    log.info("=" * 60)
    log.info("JOIN VALIDATION")
    log.info("=" * 60)

    # ── Step 1: Validate devices → manufacturers join ──
    log.info("--- Join: devices.manufacturer_id → manufacturers.id ---")
    log.info("  devices rows:        %d", len(devices))
    log.info("  manufacturers rows:  %d", len(manufacturers))

    # How many device manufacturer_ids are in manufacturers?
    device_mfr_ids = set(devices["manufacturer_id"].unique())
    mfr_ids = set(manufacturers["id"].unique())
    matched_mfr = device_mfr_ids & mfr_ids
    unmatched_device_mfr = device_mfr_ids - mfr_ids
    orphan_mfr = mfr_ids - device_mfr_ids

    log.info("  Unique manufacturer_ids in devices:  %d", len(device_mfr_ids))
    log.info("  Unique ids in manufacturers:         %d", len(mfr_ids))
    log.info("  Matched:                             %d (%.2f%%)",
             len(matched_mfr), len(matched_mfr) / len(device_mfr_ids) * 100)
    log.info("  Unmatched device manufacturer_ids:   %d", len(unmatched_device_mfr))
    log.info("  Orphan manufacturers (no devices):   %d", len(orphan_mfr))

    if unmatched_device_mfr:
        log.warning("  %d device rows reference manufacturer_ids not in manufacturers table!",
                     devices["manufacturer_id"].isin(unmatched_device_mfr).sum())

    # ── Step 2: Validate events → devices join ──
    log.info("--- Join: events.device_id → devices.id ---")
    log.info("  events rows:   %d", len(events))
    log.info("  devices rows:  %d", len(devices))

    event_device_ids = set(events["device_id"].unique())
    device_ids = set(devices["id"].unique())
    matched_dev = event_device_ids & device_ids
    unmatched_event_dev = event_device_ids - device_ids
    orphan_dev = device_ids - event_device_ids

    log.info("  Unique device_ids in events:  %d", len(event_device_ids))
    log.info("  Unique ids in devices:        %d", len(device_ids))
    log.info("  Matched:                      %d (%.2f%%)",
             len(matched_dev), len(matched_dev) / len(event_device_ids) * 100)
    log.info("  Unmatched event device_ids:   %d", len(unmatched_event_dev))
    log.info("  Orphan devices (no events):   %d", len(orphan_dev))

    if unmatched_event_dev:
        log.warning("  %d event rows reference device_ids not in devices table!",
                     events["device_id"].isin(unmatched_event_dev).sum())

    # ── Step 3: Build merged DataFrame ──
    # Strategy: events LEFT JOIN devices ON events.device_id = devices.id
    #           then LEFT JOIN manufacturers ON devices.manufacturer_id = manufacturers.id
    # Using suffixes to avoid column name collisions
    log.info("--- Building merged DataFrame ---")

    # Prefix device columns (except id and manufacturer_id) to avoid collision
    devices_for_join = devices.rename(
        columns={
            c: f"device_{c}"
            for c in devices.columns
            if c not in ("id", "manufacturer_id")
        }
    )
    devices_for_join = devices_for_join.rename(columns={"id": "device_id_from_devices"})

    # Prefix manufacturer columns (except id) to avoid collision
    mfr_for_join = manufacturers.rename(
        columns={c: f"mfr_{c}" for c in manufacturers.columns if c != "id"}
    )
    mfr_for_join = mfr_for_join.rename(columns={"id": "mfr_id_from_manufacturers"})

    n_before = len(events)

    # Join events → devices
    merged = events.merge(
        devices_for_join,
        left_on="device_id",
        right_on="device_id_from_devices",
        how="left",
        validate="m:1",  # Many events per device
    )
    # Drop the redundant join key column
    merged = merged.drop(columns=["device_id_from_devices"])

    n_after_device_join = len(merged)
    log.info("  events × devices: %d → %d rows (should be unchanged for left join)",
             n_before, n_after_device_join)
    assert n_after_device_join == n_before, (
        f"Row count changed after left join! {n_before} → {n_after_device_join}"
    )

    # Join merged → manufacturers
    merged = merged.merge(
        mfr_for_join,
        left_on="manufacturer_id",
        right_on="mfr_id_from_manufacturers",
        how="left",
        validate="m:1",  # Many devices per manufacturer
    )
    merged = merged.drop(columns=["mfr_id_from_manufacturers"])

    n_after_mfr_join = len(merged)
    log.info("  (events×devices) × manufacturers: %d → %d rows",
             n_after_device_join, n_after_mfr_join)
    assert n_after_mfr_join == n_after_device_join, (
        f"Row count changed after manufacturer join! {n_after_device_join} → {n_after_mfr_join}"
    )

    # Verify no device info was lost
    device_info_null = merged["device_name"].isna().sum()
    mfr_info_null = merged["mfr_name"].isna().sum()
    log.info("  Rows missing device info after join: %d", device_info_null)
    log.info("  Rows missing manufacturer info after join: %d", mfr_info_null)

    log.info("  Final merged shape: %d rows × %d columns", *merged.shape)
    log.info("=" * 60)

    return merged


# =============================================================================
# Persistence
# =============================================================================

def persist_parquet(df: pd.DataFrame, path: Path, name: str) -> None:
    """Write a DataFrame to Parquet, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    size_mb = path.stat().st_size / (1024 * 1024)
    log.info("  Saved %s: %d rows × %d cols (%.1f MB)", name, len(df), len(df.columns), size_mb)


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    log.info("=" * 70)
    log.info("STAGE 2 — Data Engineering Pipeline")
    log.info("=" * 70)

    csv_paths = [DEVICES_CSV, EVENTS_CSV, MANUFACTURERS_CSV]

    # Verify all files exist
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required raw data file missing: {path}")

    # Check manifest for cache validity
    if _check_manifest(csv_paths):
        log.info("Pipeline outputs are up to date. Nothing to do.")
        return

    start_time = time.time()

    # ── Load ──
    manufacturers = _load_csv(MANUFACTURERS_CSV)
    devices = _load_csv(DEVICES_CSV)
    events = _load_csv(EVENTS_CSV)

    # ── Clean ──
    manufacturers = clean_manufacturers(manufacturers)
    devices = clean_devices(devices)
    events = clean_events(events)

    # ── Validate and join ──
    merged = validate_and_join(devices, events, manufacturers)

    # ── Persist ──
    log.info("Persisting Parquet files …")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    persist_parquet(devices, PROCESSED_DATA_DIR / "devices.parquet", "devices.parquet")
    persist_parquet(events, PROCESSED_DATA_DIR / "events.parquet", "events.parquet")
    persist_parquet(manufacturers, PROCESSED_DATA_DIR / "manufacturers.parquet", "manufacturers.parquet")
    persist_parquet(merged, PROCESSED_DATA_DIR / "merged.parquet", "merged.parquet")

    # ── Write manifest ──
    elapsed = time.time() - start_time
    run_stats = {
        "elapsed_seconds": round(elapsed, 2),
        "manufacturers_rows": len(manufacturers),
        "devices_rows": len(devices),
        "events_rows": len(events),
        "merged_rows": len(merged),
        "merged_columns": len(merged.columns),
        "events_dated": int(events["event_date_available"].sum()),
        "events_undated": int((~events["event_date_available"]).sum()),
    }
    _write_manifest(csv_paths, run_stats)

    log.info("=" * 70)
    log.info("STAGE 2 COMPLETE — Pipeline finished in %.1f seconds", elapsed)
    log.info("=" * 70)

    # ── Summary ──
    log.info("OUTPUT SUMMARY:")
    log.info("  data/processed/manufacturers.parquet : %d rows", len(manufacturers))
    log.info("  data/processed/devices.parquet       : %d rows", len(devices))
    log.info("  data/processed/events.parquet        : %d rows", len(events))
    log.info("  data/processed/merged.parquet        : %d rows × %d cols", *merged.shape)


if __name__ == "__main__":
    main()
