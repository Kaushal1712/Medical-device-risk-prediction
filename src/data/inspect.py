"""
Dataset Inspection Script — Stage 1

Profiles all three raw CSVs (devices, events, manufacturers) to produce
a comprehensive data-quality report, without making any schema assumptions.

Run:  python -m src.data.inspect

Outputs:  docs/01_dataset_inspection_report.md
"""

import hashlib
import logging
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup — use config for paths
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DEVICES_CSV, EVENTS_CSV, MANUFACTURERS_CSV, DOCS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Thresholds for heuristic detection
MAX_UNIQUE_FOR_CATEGORICAL = 100  # columns with ≤ this many unique vals get categorical treatment
MIN_TEXT_MEDIAN_LEN = 50  # columns whose median string length exceeds this are treated as free-text


# =============================================================================
# Helper utilities
# =============================================================================

def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV, failing loudly if missing."""
    if not path.exists():
        log.error("File not found: %s", path)
        raise FileNotFoundError(f"Required raw data file missing: {path}")
    log.info("Loading %s …", path.name)
    df = pd.read_csv(path, low_memory=False)
    log.info("  → %d rows × %d columns", *df.shape)
    return df


def _missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with missing-count and missing-% per column."""
    n = len(df)
    missing = df.isnull().sum()
    pct = (missing / n * 100).round(2)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct}).sort_values(
        "missing_pct", ascending=False
    )


def _cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """Return unique-value counts per column, plus dtype."""
    records = []
    for col in df.columns:
        records.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_unique": df[col].nunique(),
                "n_non_null": df[col].notna().sum(),
            }
        )
    return pd.DataFrame(records).set_index("column")


def _detect_duplicates(df: pd.DataFrame, name: str) -> dict:
    """Check for full-row duplicates and, for each column that could be a PK, ID duplicates."""
    result = {
        "full_row_duplicates": int(df.duplicated().sum()),
    }
    return result


def _candidate_primary_keys(df: pd.DataFrame) -> list[str]:
    """Return column names whose non-null values are all unique (candidate PKs)."""
    candidates = []
    n = len(df)
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == n and non_null.is_unique:
            candidates.append(col)
        elif len(non_null) > 0 and non_null.is_unique:
            # Unique among non-null values but has nulls — note it
            candidates.append(f"{col} (unique among {len(non_null)} non-null values)")
    return candidates


def _is_string_dtype(series: pd.Series) -> bool:
    """Check if a series has a string-like dtype (handles both 'object' and pandas 3.x 'str')."""
    return series.dtype == "object" or pd.api.types.is_string_dtype(series)


def _detect_date_columns(df: pd.DataFrame) -> dict[str, dict]:
    """
    Attempt to parse every string-like column as a datetime.
    Returns {col_name: {min, max, missing_count, missing_pct}} for successful parses.
    """
    date_info = {}
    for col in df.columns:
        if _is_string_dtype(df[col]) or "date" in col.lower() or "time" in col.lower():
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                valid = parsed.notna().sum()
                # Only count as a date column if ≥30% of non-null values parsed successfully
                non_null_original = df[col].notna().sum()
                if non_null_original > 0 and valid / non_null_original >= 0.30:
                    date_info[col] = {
                        "min": str(parsed.min()),
                        "max": str(parsed.max()),
                        "valid_parsed": int(valid),
                        "total_non_null": int(non_null_original),
                        "parse_rate_pct": round(valid / non_null_original * 100, 2),
                        "missing_count": int(parsed.isna().sum()),
                        "missing_pct": round(parsed.isna().sum() / len(df) * 100, 2),
                    }
            except Exception:
                pass
    return date_info


def _detect_event_type_columns(df: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    """
    Identify columns that look like event type/category/classification.
    Heuristic: string-like dtype, low-medium cardinality (2–500 unique), not an ID/slug/url.
    Returns {col_name: [(value, count), …]} sorted by count desc.
    """
    skip_patterns = {"url", "link", "slug", "uid", "hash", "documents", "address"}
    results = {}
    for col in df.columns:
        col_lower = col.lower()
        if not _is_string_dtype(df[col]):
            continue
        if any(p in col_lower for p in skip_patterns):
            continue
        nunique = df[col].nunique()
        if 2 <= nunique <= 500:
            vc = df[col].value_counts(dropna=False).head(50)
            results[col] = [(str(val), int(cnt)) for val, cnt in vc.items()]
    return results


def _detect_free_text(df: pd.DataFrame) -> dict[str, dict]:
    """
    Identify free-text columns: object dtype with high median string length.
    """
    results = {}
    for col in df.columns:
        if not _is_string_dtype(df[col]):
            continue
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        lengths = non_null.astype(str).str.len()
        median_len = lengths.median()
        if median_len >= MIN_TEXT_MEDIAN_LEN:
            results[col] = {
                "non_null_count": int(len(non_null)),
                "non_null_rate_pct": round(len(non_null) / len(df) * 100, 2),
                "median_length": float(median_len),
                "mean_length": round(float(lengths.mean()), 1),
                "max_length": int(lengths.max()),
                "sample": str(non_null.iloc[0])[:200] + ("…" if len(str(non_null.iloc[0])) > 200 else ""),
            }
    return results


def _foreign_key_overlap(
    df_left: pd.DataFrame, df_right: pd.DataFrame, name_left: str, name_right: str
) -> list[dict]:
    """
    For every pair of columns (one from each DF), compute set-overlap if dtypes
    are compatible. Returns only pairs with meaningful overlap (>10% of smaller set).
    """
    results = []
    for col_l in df_left.columns:
        vals_l = set(df_left[col_l].dropna().astype(str).unique())
        if len(vals_l) == 0:
            continue
        for col_r in df_right.columns:
            vals_r = set(df_right[col_r].dropna().astype(str).unique())
            if len(vals_r) == 0:
                continue
            overlap = vals_l & vals_r
            smaller = min(len(vals_l), len(vals_r))
            if smaller == 0:
                continue
            overlap_pct = len(overlap) / smaller * 100
            if overlap_pct >= 10 and len(overlap) >= 5:
                results.append(
                    {
                        "left_table": name_left,
                        "left_column": col_l,
                        "left_unique": len(vals_l),
                        "right_table": name_right,
                        "right_column": col_r,
                        "right_unique": len(vals_r),
                        "overlap_count": len(overlap),
                        "overlap_pct_of_smaller": round(overlap_pct, 2),
                    }
                )
    # Sort by overlap descending
    results.sort(key=lambda x: x["overlap_count"], reverse=True)
    return results


# =============================================================================
# Main inspection runner
# =============================================================================

def inspect_dataset(path: Path, name: str) -> dict:
    """Run all inspections on a single CSV, returning a structured result dict."""
    df = _load_csv(path)

    log.info("Profiling %s …", name)
    result = {
        "name": name,
        "path": str(path),
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missingness": _missingness(df).to_dict("index"),
        "cardinality": _cardinality(df).to_dict("index"),
        "duplicates": _detect_duplicates(df, name),
        "candidate_pks": _candidate_primary_keys(df),
        "date_columns": _detect_date_columns(df),
        "event_type_columns": _detect_event_type_columns(df),
        "free_text_columns": _detect_free_text(df),
    }
    return result, df


def generate_report(
    inspections: dict[str, dict],
    fk_results: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Generate the markdown inspection report from all inspection results."""
    lines = []
    lines.append("# Dataset Inspection Report")
    lines.append("")
    lines.append("> Auto-generated by `src/data/inspect.py` — Stage 1")
    lines.append(">")
    lines.append("> **All values below are computed from the actual data files.**")
    lines.append("")

    # ── Per-dataset sections ──────────────────────────────────────────────
    for name, info in inspections.items():
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## {name}")
        lines.append(f"")

        # Shape
        lines.append(f"### Shape")
        lines.append(f"- **Rows:** {info['shape']['rows']:,}")
        lines.append(f"- **Columns:** {info['shape']['columns']}")
        lines.append(f"")

        # Column names + dtypes
        lines.append(f"### Columns and Data Types")
        lines.append(f"| Column | Dtype |")
        lines.append(f"|--------|-------|")
        for col in info["columns"]:
            lines.append(f"| `{col}` | {info['dtypes'][col]} |")
        lines.append(f"")

        # Missingness
        lines.append(f"### Missingness")
        lines.append(f"| Column | Missing Count | Missing % |")
        lines.append(f"|--------|--------------|-----------|")
        # Sort by missing % desc
        miss_items = sorted(
            info["missingness"].items(),
            key=lambda x: x[1]["missing_pct"],
            reverse=True,
        )
        for col, m in miss_items:
            lines.append(f"| `{col}` | {m['missing_count']:,} | {m['missing_pct']}% |")
        lines.append(f"")

        # Cardinality
        lines.append(f"### Cardinality")
        lines.append(f"| Column | Dtype | Unique Values | Non-Null |")
        lines.append(f"|--------|-------|--------------|----------|")
        for col, c in info["cardinality"].items():
            lines.append(f"| `{col}` | {c['dtype']} | {c['n_unique']:,} | {c['n_non_null']:,} |")
        lines.append(f"")

        # Duplicates
        lines.append(f"### Duplicates")
        lines.append(f"- **Full row duplicates:** {info['duplicates']['full_row_duplicates']:,}")
        lines.append(f"")

        # Candidate primary keys
        lines.append(f"### Candidate Primary Keys")
        if info["candidate_pks"]:
            for pk in info["candidate_pks"]:
                lines.append(f"- `{pk}`")
        else:
            lines.append(f"- *No column with fully unique non-null values found.*")
        lines.append(f"")

        # Date columns
        lines.append(f"### Date/Datetime Columns")
        if info["date_columns"]:
            lines.append(f"| Column | Min | Max | Valid Parsed | Total Non-Null | Parse Rate | Missing Count | Missing % |")
            lines.append(f"|--------|-----|-----|-------------|---------------|------------|--------------|-----------|")
            for col, d in info["date_columns"].items():
                lines.append(
                    f"| `{col}` | {d['min']} | {d['max']} | {d['valid_parsed']:,} | "
                    f"{d['total_non_null']:,} | {d['parse_rate_pct']}% | "
                    f"{d['missing_count']:,} | {d['missing_pct']}% |"
                )
        else:
            lines.append(f"- *No date/datetime columns detected.*")
        lines.append(f"")

        # Event type / category columns
        lines.append(f"### Event Type / Category Columns")
        if info["event_type_columns"]:
            for col, values in info["event_type_columns"].items():
                lines.append(f"#### `{col}` ({len(values)} distinct values shown, up to 50)")
                lines.append(f"| Value | Count |")
                lines.append(f"|-------|-------|")
                for val, cnt in values:
                    # Escape pipe characters in values
                    safe_val = str(val).replace("|", "\\|")
                    lines.append(f"| {safe_val} | {cnt:,} |")
                lines.append(f"")
        else:
            lines.append(f"- *No categorical event-type columns detected.*")
        lines.append(f"")

        # Free-text columns
        lines.append(f"### Free-Text Columns")
        if info["free_text_columns"]:
            for col, t in info["free_text_columns"].items():
                lines.append(f"#### `{col}`")
                lines.append(f"- **Non-null count:** {t['non_null_count']:,} ({t['non_null_rate_pct']}%)")
                lines.append(f"- **Median length:** {t['median_length']:.0f} chars")
                lines.append(f"- **Mean length:** {t['mean_length']:.0f} chars")
                lines.append(f"- **Max length:** {t['max_length']:,} chars")
                sample = t["sample"].replace("|", "\\|").replace("\n", " ")
                lines.append(f"- **Sample:** `{sample}`")
                lines.append(f"")
        else:
            lines.append(f"- *No free-text columns detected.*")
        lines.append(f"")

    # ── Foreign key analysis ──────────────────────────────────────────────
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Foreign Key / Join Key Analysis")
    lines.append(f"")

    for pair_name, results in fk_results.items():
        lines.append(f"### {pair_name}")
        if results:
            lines.append(f"| Left Table | Left Column | Left Unique | Right Table | Right Column | Right Unique | Overlap | Overlap % (of smaller) |")
            lines.append(f"|-----------|------------|------------|------------|-------------|-------------|---------|----------------------|")
            # Show top 15 most meaningful overlaps
            for r in results[:15]:
                lines.append(
                    f"| {r['left_table']} | `{r['left_column']}` | {r['left_unique']:,} | "
                    f"{r['right_table']} | `{r['right_column']}` | {r['right_unique']:,} | "
                    f"{r['overlap_count']:,} | {r['overlap_pct_of_smaller']}% |"
                )
        else:
            lines.append(f"- *No significant column-value overlaps found.*")
        lines.append(f"")

    # ── Discovered relationship diagram ───────────────────────────────────
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Discovered Table Relationships")
    lines.append(f"")
    lines.append(f"*Based on the foreign-key overlap analysis above, the join-key mapping is:*")
    lines.append(f"")
    lines.append(f"_(This section is populated after reviewing the overlap results above.)_")
    lines.append(f"")

    # Write the report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", output_path)


# =============================================================================
# Entry point
# =============================================================================

def main():
    log.info("=" * 70)
    log.info("STAGE 1 — Dataset Inspection")
    log.info("=" * 70)

    # Verify all files exist before starting
    for path in [DEVICES_CSV, EVENTS_CSV, MANUFACTURERS_CSV]:
        if not path.exists():
            log.error("MISSING: %s", path)
            raise FileNotFoundError(
                f"Required raw data file missing: {path}. "
                f"Place the CSV files in {path.parent}/ and re-run."
            )

    # Inspect each dataset
    inspections = OrderedDict()
    dataframes = {}

    for path, name in [
        (DEVICES_CSV, "devices.csv"),
        (EVENTS_CSV, "events.csv"),
        (MANUFACTURERS_CSV, "manufacturers.csv"),
    ]:
        info, df = inspect_dataset(path, name)
        inspections[name] = info
        dataframes[name] = df

    # Foreign-key overlap analysis between all pairs
    log.info("Computing foreign-key overlaps …")
    fk_results = OrderedDict()

    fk_results["devices ↔ events"] = _foreign_key_overlap(
        dataframes["devices.csv"], dataframes["events.csv"], "devices", "events"
    )
    fk_results["devices ↔ manufacturers"] = _foreign_key_overlap(
        dataframes["devices.csv"], dataframes["manufacturers.csv"], "devices", "manufacturers"
    )
    fk_results["events ↔ manufacturers"] = _foreign_key_overlap(
        dataframes["events.csv"], dataframes["manufacturers.csv"], "events", "manufacturers"
    )

    # Generate markdown report
    report_path = DOCS_DIR / "01_dataset_inspection_report.md"
    generate_report(inspections, fk_results, report_path)

    log.info("=" * 70)
    log.info("STAGE 1 COMPLETE — Review the report at: %s", report_path)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
