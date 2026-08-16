"""
src/risk/build_serving_table.py
================================
Stage 6 — Build the production serving table.

Scores every event row across all four feature splits (train, validation,
test, holdout_2018) using the calibrated model, then applies the Stage 3f
serving-snapshot policy: retain only the row with the latest event_date
per device_id.

Outputs artifacts/risk/device_risk_snapshot.parquet — the single source
of truth read by the backend API and dashboard.

Run:
    python -m src.risk.build_serving_table

Prerequisites:
    python -m src.risk.calibrate   (calibrated_model.pkl must exist)
"""

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    FEATURES_DATA_DIR,
    PRODUCTION_MODEL_DIR,
    RISK_SNAPSHOT_PATH,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
)
from src.risk.scorer import RiskScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SPLIT_NAMES = ["train", "validation", "test", "holdout_2018"]


def main() -> None:
    t0 = time.time()
    log.info("=" * 70)
    log.info("STAGE 6 -- Build Serving Table")
    log.info("=" * 70)

    # ── Validate thresholds ────────────────────────────────────────────────
    if RISK_THRESHOLD_MEDIUM == 0.0 or RISK_THRESHOLD_HIGH == 0.0:
        log.error(
            "RISK_THRESHOLD_MEDIUM or RISK_THRESHOLD_HIGH is 0.0 in src/config.py. "
            "Run `python -m src.risk.calibrate` first to derive valid thresholds."
        )
        sys.exit(1)

    log.info(
        "Thresholds: MEDIUM=%.6f  HIGH=%.6f",
        RISK_THRESHOLD_MEDIUM,
        RISK_THRESHOLD_HIGH,
    )

    # ── Load scorer ────────────────────────────────────────────────────────
    scorer = RiskScorer(PRODUCTION_MODEL_DIR)
    scorer.load_calibration()

    # ── Load and concatenate all splits ───────────────────────────────────
    log.info("\nLoading feature splits...")
    dfs = []
    for split in SPLIT_NAMES:
        path = FEATURES_DATA_DIR / f"{split}.parquet"
        df = pd.read_parquet(path)
        df["_split"] = split
        dfs.append(df)
        log.info("  %s: %d rows", split, len(df))

    all_events = pd.concat(dfs, ignore_index=True)
    log.info("Total events loaded: %d  |  Unique device_ids: %d",
             len(all_events), all_events["device_id"].nunique())

    # ── Batch score all rows ───────────────────────────────────────────────
    log.info("\nScoring all %d rows...", len(all_events))
    scored = scorer.batch_score(
        all_events,
        t_medium=RISK_THRESHOLD_MEDIUM,
        t_high=RISK_THRESHOLD_HIGH,
    )
    log.info("Scoring complete.")

    # ── Apply Stage 3f serving-snapshot policy ─────────────────────────────
    # For each device_id: retain only the row with the latest event_date.
    # This is the most information-rich snapshot and respects Stage 3f.
    log.info("\nApplying Stage 3f serving-snapshot policy (latest event_date per device)...")
    scored["event_date"] = pd.to_datetime(scored["event_date"])
    # Sort then drop_duplicates preserves the last (latest) row per device_id
    scored_sorted = scored.sort_values("event_date")
    serving = (
        scored_sorted
        .drop_duplicates(subset=["device_id"], keep="last")
        .reset_index(drop=True)
    )
    log.info(
        "Serving table: %d rows (one per device_id)  "
        "Multi-event devices deduplicated: %d",
        len(serving),
        len(scored) - len(serving),
    )

    # ── Add provenance columns ─────────────────────────────────────────────
    scored_at = datetime.now(timezone.utc).isoformat()
    serving["scored_at"] = scored_at

    # Rename for clarity in serving schema
    serving = serving.rename(columns={
        "event_date": "serving_event_date",
        "calibrated_probability": "calibrated_probability",
        "id": "event_id",
    })

    # Final column order
    serving_cols = [
        "device_id",
        "event_id",
        "serving_event_date",
        "raw_probability",
        "calibrated_probability",
        "risk_score",
        "risk_level",
        "is_class_i_predicted",
        "decision_threshold",
        "model_version",
        "scored_at",
    ]
    # Keep only the defined columns (drop _split etc.)
    serving = serving[[c for c in serving_cols if c in serving.columns]]

    # ── Risk level distribution ────────────────────────────────────────────
    band_counts = serving["risk_level"].value_counts()
    total = len(serving)
    log.info("\n  Risk level distribution:")
    for band in ["HIGH", "MEDIUM", "LOW"]:
        n = int(band_counts.get(band, 0))
        log.info("    %s: %d  (%.1f%%)", band, n, 100.0 * n / total)

    log.info("\n  Risk score stats:")
    log.info("    min=%.2f  mean=%.2f  median=%.2f  max=%.2f",
             serving["risk_score"].min(),
             serving["risk_score"].mean(),
             serving["risk_score"].median(),
             serving["risk_score"].max())

    # ── Save ───────────────────────────────────────────────────────────────
    RISK_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serving.to_parquet(RISK_SNAPSHOT_PATH, index=False)

    elapsed = time.time() - t0
    log.info("\n" + "=" * 70)
    log.info("Serving table complete in %.1f seconds.", elapsed)
    log.info("  Output: %s  (%d rows, %d columns)",
             RISK_SNAPSHOT_PATH, len(serving), len(serving.columns))
    log.info("=" * 70)


if __name__ == "__main__":
    main()
