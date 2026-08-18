"""
backend/services/model_service.py
===================================
Stage 7 — Model singleton service.

Loads all production artifacts ONCE at process startup via a module-level
singleton pattern. Never re-loaded per request.

Artifacts loaded:
  - artifacts/risk/device_risk_snapshot.parquet  → risk lookup dict
  - data/processed/merged.parquet                → device detail lookup
  - data/features/train.parquet + test.parquet   → feature lookup for SHAP
  - models/production/model_card.json            → model metadata
  - data/processed/_manifest.json                → data provenance hash

Public API
----------
  get_model_service() -> ModelService   (call this everywhere; returns singleton)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---- path resolution (relative to project root) ----------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_RISK_SNAPSHOT = _PROJECT_ROOT / "artifacts" / "risk" / "device_risk_snapshot.parquet"
_MERGED_PARQUET = _PROJECT_ROOT / "data" / "processed" / "merged.parquet"
_TRAIN_PARQUET  = _PROJECT_ROOT / "data" / "features" / "train.parquet"
_TEST_PARQUET   = _PROJECT_ROOT / "data" / "features" / "test.parquet"
_VAL_PARQUET    = _PROJECT_ROOT / "data" / "features" / "validation.parquet"
_MODEL_CARD     = _PROJECT_ROOT / "models" / "production" / "model_card.json"
_MANIFEST       = _PROJECT_ROOT / "data" / "processed" / "_manifest.json"
_FEAT_IMPORTANCE = _PROJECT_ROOT / "models" / "production" / "feature_importance.json"

# Columns from merged.parquet to include in device detail responses
_DEVICE_DETAIL_COLS = [
    "device_id", "device_name", "device_description", "device_classification",
    "device_risk_class", "device_implanted", "device_country", "device_number",
    "device_distributed_to", "manufacturer_id", "mfr_name", "mfr_parent_company",
    "mfr_source", "mfr_address",
]

# Columns from merged.parquet needed for list display
_DEVICE_LIST_COLS = [
    "device_id", "device_name", "device_classification", "device_risk_class",
    "device_country", "mfr_name", "mfr_parent_company",
]

# Feature metadata columns (not part of X but useful for lookups)
_META_COLS = frozenset(
    {"id", "device_id", "manufacturer_id", "event_date", "event_date_available", "is_class_i"}
)


class ModelService:
    """
    Singleton holding all in-memory artifacts. Do not instantiate directly —
    call get_model_service() instead.
    """

    def __init__(self) -> None:
        log.info("ModelService: loading all production artifacts ...")

        # 1. Risk serving table
        if not _RISK_SNAPSHOT.exists():
            raise FileNotFoundError(
                f"Risk snapshot not found: {_RISK_SNAPSHOT}. "
                "Run `python -m src.risk.build_serving_table` first."
            )
        self._risk_df: pd.DataFrame = pd.read_parquet(_RISK_SNAPSHOT)
        # Normalize device_id to str for consistent HTTP-layer lookups (parquet stores int64)
        self._risk_df["device_id"] = self._risk_df["device_id"].astype(str)
        self._risk_index = self._risk_df.set_index("device_id")
        log.info("ModelService: risk snapshot loaded — %d devices", len(self._risk_index))

        # 2. Merged parquet (device / event / manufacturer details)
        if not _MERGED_PARQUET.exists():
            raise FileNotFoundError(f"Merged parquet not found: {_MERGED_PARQUET}")
        # Load only the columns needed for device detail responses.
        # Avoid loading all 56 columns into memory.
        merged_raw = pd.read_parquet(
            _MERGED_PARQUET,
            columns=_DEVICE_DETAIL_COLS,
        )

        # Keep only columns we need; deduplicate per device_id (keep latest row)
        avail_cols = [c for c in _DEVICE_DETAIL_COLS if c in merged_raw.columns]
        # merged.parquet has many rows per device; keep one canonical row per device_id
        device_df_raw = (
            merged_raw[avail_cols]
            .drop_duplicates(subset=["device_id"], keep="last")
            .reset_index(drop=True)
        )
        device_df_raw["device_id"] = device_df_raw["device_id"].astype(str)
        self._device_df: pd.DataFrame = device_df_raw
        self._device_index: dict[str, dict] = (
            self._device_df.set_index("device_id")
            .to_dict(orient="index")
        )
        log.info("ModelService: device store loaded — %d unique devices", len(self._device_index))

        # 3. Feature Parquet (for SHAP lookups — latest event row per device)
        self._feature_df: pd.DataFrame = self._load_feature_store()
        log.info("ModelService: feature store loaded — %d device snapshots", len(self._feature_df))

        # 4. Model card
        self._model_card: dict = {}
        if _MODEL_CARD.exists():
            with open(_MODEL_CARD) as f:
                self._model_card = json.load(f)
        self._model_version: str = self._model_card.get("experiment_dir", "unknown")
        log.info("ModelService: model version = %s", self._model_version)

        # 5. Data manifest hash
        self._manifest_hash: str = "unknown"
        if _MANIFEST.exists():
            with open(_MANIFEST) as f:
                manifest = json.load(f)
            # The manifest records hashes per file; concatenate for a single digest
            hashes = manifest.get("file_hashes", manifest.get("hashes", {}))
            if isinstance(hashes, dict):
                combined = "|".join(f"{k}={v}" for k, v in sorted(hashes.items()))
                import hashlib
                self._manifest_hash = hashlib.md5(combined.encode()).hexdigest()[:16]
            else:
                self._manifest_hash = str(hashes)[:16]

        # 6. Feature importance
        self._feature_importance: list[dict] = []
        if _FEAT_IMPORTANCE.exists():
            with open(_FEAT_IMPORTANCE) as f:
                self._feature_importance = json.load(f)

        log.info("ModelService: all artifacts loaded successfully.")

    # ------------------------------------------------------------------
    # Risk lookup
    # ------------------------------------------------------------------

    def get_device_risk(self, device_id: str) -> Optional[dict]:
        """
        Returns the serving-snapshot risk dict for a device, or None if
        the device has no valid scoreable snapshot.
        """
        try:
            row = self._risk_index.loc[str(device_id)]
        except KeyError:
            return None

        return row.to_dict()

    # ------------------------------------------------------------------
    # Device lookup
    # ------------------------------------------------------------------

    def get_device_detail(self, device_id: str) -> Optional[dict]:
        """
        Returns raw device detail dict from merged.parquet, or None.
        """
        return self._device_index.get(str(device_id))

    def get_all_devices_df(self) -> pd.DataFrame:
        """
        Returns the full device DataFrame (one row per device_id) with
        device attributes. Risk columns are NOT included — caller joins
        from risk_df if needed.
        """
        return self._device_df

    def get_risk_df(self) -> pd.DataFrame:
        return self._risk_df

    # ------------------------------------------------------------------
    # Feature lookup (for SHAP)
    # ------------------------------------------------------------------

    def get_device_feature_row(self, device_id: str) -> Optional[pd.Series]:
        """
        Returns the latest feature row for a device as a pd.Series, or None.
        """
        mask = self._feature_df["device_id"] == str(device_id)
        rows = self._feature_df[mask]
        if rows.empty:
            return None
        return rows.iloc[-1]  # latest row (df is sorted by event_date in _load_feature_store)

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_card(self) -> dict:
        return self._model_card

    @property
    def manifest_hash(self) -> str:
        return self._manifest_hash

    @property
    def feature_importance(self) -> list[dict]:
        return self._feature_importance

    # ------------------------------------------------------------------
    # Risk summary (pre-computed aggregate — called by /risk-summary)
    # ------------------------------------------------------------------

    def compute_risk_summary(self) -> dict:
        """
        Compute aggregate risk statistics from the serving table and device store.
        Returns a dict matching RiskSummaryResponse schema.
        """
        risk_df = self._risk_df
        total_scored = len(risk_df)
        total_devices = len(self._device_index)
        total_unscored = max(0, total_devices - total_scored)

        # Per-level counts
        level_counts = risk_df["risk_level"].value_counts().to_dict()
        risk_levels = {}
        for lvl in ("HIGH", "MEDIUM", "LOW"):
            n = int(level_counts.get(lvl, 0))
            risk_levels[lvl] = {
                "count": n,
                "percent": round(100.0 * n / total_scored, 2) if total_scored > 0 else 0.0,
            }

        # Risk score stats
        scores = risk_df["risk_score"]
        risk_score_stats = {
            "min": round(float(scores.min()), 2),
            "mean": round(float(scores.mean()), 2),
            "median": round(float(scores.median()), 2),
            "max": round(float(scores.max()), 2),
        }

        # Join with device attributes for breakdown
        merged = risk_df.merge(
            self._device_df[["device_id", "device_classification", "mfr_parent_company"]],
            on="device_id",
            how="left",
        )

        # Category breakdown (top 15)
        cat_breakdown = self._breakdown(merged, "device_classification", top_n=15)

        # Manufacturer breakdown (top 15)
        mfr_breakdown = self._breakdown(merged, "mfr_parent_company", top_n=15)

        return {
            "total_devices_in_data": total_devices,
            "total_scored": total_scored,
            "total_unscored": total_unscored,
            "risk_levels": risk_levels,
            "risk_score_stats": risk_score_stats,
            "category_breakdown": cat_breakdown,
            "manufacturer_breakdown": mfr_breakdown,
        }

    @staticmethod
    def _breakdown(df: pd.DataFrame, col: str, top_n: int = 15) -> list[dict]:
        """Compute per-value HIGH/MEDIUM/LOW breakdown for a column."""
        df = df.copy()
        df[col] = df[col].fillna("Unknown").astype(str)
        grouped = df.groupby([col, "risk_level"]).size().unstack(fill_value=0)
        for lvl in ("HIGH", "MEDIUM", "LOW"):
            if lvl not in grouped.columns:
                grouped[lvl] = 0
        grouped["total"] = grouped[["HIGH", "MEDIUM", "LOW"]].sum(axis=1)
        grouped = grouped.sort_values("total", ascending=False).head(top_n)
        result = []
        for idx_val, row in grouped.iterrows():
            result.append({
                "category" if col == "device_classification" else "manufacturer": str(idx_val),
                "high": int(row.get("HIGH", 0)),
                "medium": int(row.get("MEDIUM", 0)),
                "low": int(row.get("LOW", 0)),
                "total": int(row["total"]),
            })
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_feature_store(self) -> pd.DataFrame:
        """
        Load train + validation + test feature Parquets, concatenate,
        sort by event_date, and keep latest row per device_id.
        """
        dfs = []
        for path in (_TRAIN_PARQUET, _VAL_PARQUET, _TEST_PARQUET):
            if path.exists():
                dfs.append(pd.read_parquet(path))
        if not dfs:
            log.warning("ModelService: no feature Parquet files found — feature store empty.")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
            df = df.sort_values("event_date", ascending=True)
        # Keep only the latest feature row per device_id
        df = df.drop_duplicates(subset=["device_id"], keep="last").reset_index(drop=True)
        # Normalize device_id to str (parquet may store int64)
        df["device_id"] = df["device_id"].astype(str)
        return df



# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Return (and lazily initialise) the global ModelService singleton."""
    global _service
    if _service is None:
        _service = ModelService()
    return _service
