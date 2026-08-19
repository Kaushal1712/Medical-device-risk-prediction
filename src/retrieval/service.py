"""
src/retrieval/service.py
=========================
Historical Evidence Retrieval Service — SQLite-backed, per-request reads.

Wraps the artifacts/serving/historical_evidence.sqlite database to provide:
  - Device metadata lookup
  - Device-specific historical events
  - FTS5-powered similar-event search
  - Pre-aggregated historical facts

Design notes
------------
- All reads are per-request (no eager in-memory loading).
- The SQLite connection is opened lazily in each method call and closed
  immediately, keeping memory footprint near zero.
- The historical_evidence database contains ONLY pre-cutoff training-era
  data (events up to 2018) plus device/manufacturer metadata.
  No future-leaking fields (action_summary, determined_cause, etc.) are
  surfaced to the prediction path.
- device_id is used ONLY for retrieval context (metadata and evidence),
  NEVER as a model feature.

Database schema (from build_serving_db.py):
  device  (device_id, device_name, device_description, device_classification,
            device_risk_class, device_implanted, device_country,
            manufacturer_id, mfr_name, mfr_parent_company, mfr_source)
  event   (event_id, device_id, manufacturer_id, event_date,
            action_classification, type, reason, device_name,
            device_description, device_classification, mfr_name,
            mfr_parent_company)
  event_fts (event_id, search_text)  — FTS5 table
  risk_snapshot (device_id, event_id, serving_event_date,
                 raw_probability, calibrated_probability, risk_score,
                 risk_level, ...)
  feature_snapshot (device_id, event_date, values_blob)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PATH = _PROJECT_ROOT / "artifacts" / "serving" / "historical_evidence.sqlite"

# Maximum events returned for a device lookup
_DEFAULT_EVENT_LIMIT = 10

# Maximum FTS search results
_DEFAULT_SEARCH_LIMIT = 8


def _get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA query_only=1")
    return conn


def get_device_info(device_id: str) -> Optional[dict]:
    """
    Return basic device metadata for the given device_id.

    Returns None if the device is not in the database.
    """
    try:
        conn = _get_connection()
        try:
            cur = conn.execute(
                "SELECT * FROM device WHERE device_id = ? LIMIT 1",
                (device_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    except Exception as exc:
        log.warning("get_device_info(%s) failed: %s", device_id, exc)
        return None


def get_device_events(
    device_id: str, limit: int = _DEFAULT_EVENT_LIMIT
) -> list[dict]:
    """
    Return recent historical events for a device.

    Events are ordered most-recent first.  action_classification and other
    post-event fields are excluded from the returned dict to prevent any
    risk of leakage through the API response.
    """
    try:
        conn = _get_connection()
        try:
            cur = conn.execute(
                """
                SELECT
                    event_id,
                    event_date,
                    type,
                    reason,
                    device_name,
                    device_description,
                    device_classification,
                    mfr_name,
                    mfr_parent_company
                FROM event
                WHERE device_id = ?
                ORDER BY event_date DESC
                LIMIT ?
                """,
                (device_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        log.warning("get_device_events(%s) failed: %s", device_id, exc)
        return []


def get_risk_snapshot(device_id: str) -> Optional[dict]:
    """
    Return the pre-computed risk snapshot for a device (if available).
    """
    try:
        conn = _get_connection()
        try:
            cur = conn.execute(
                """
                SELECT device_id, serving_event_date, calibrated_probability,
                       risk_score, risk_level, model_version, scored_at
                FROM risk_snapshot
                WHERE device_id = ?
                ORDER BY serving_event_date DESC
                LIMIT 1
                """,
                (device_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    except Exception as exc:
        log.warning("get_risk_snapshot(%s) failed: %s", device_id, exc)
        return None


def get_historical_facts(device_id: str) -> dict:
    """
    Return aggregated historical event statistics for a device.

    Returns
    -------
    dict with keys:
      total_events, recall_events, class_i_events (from action_classification)
      event_date_range: (earliest, latest)
    """
    result = {
        "total_events": 0,
        "recall_events": 0,
        "class_i_events": 0,
        "earliest_event": None,
        "latest_event": None,
    }
    try:
        conn = _get_connection()
        try:
            cur = conn.execute(
                """
                SELECT
                    COUNT(*)                                    AS total_events,
                    SUM(CASE WHEN type = 'Recall' THEN 1 ELSE 0 END)   AS recall_events,
                    SUM(CASE WHEN action_classification = 'Class I' THEN 1 ELSE 0 END)
                                                                        AS class_i_events,
                    MIN(event_date)                             AS earliest_event,
                    MAX(event_date)                             AS latest_event
                FROM event
                WHERE device_id = ?
                """,
                (device_id,),
            )
            row = cur.fetchone()
            if row:
                result["total_events"] = row["total_events"] or 0
                result["recall_events"] = row["recall_events"] or 0
                result["class_i_events"] = row["class_i_events"] or 0
                result["earliest_event"] = row["earliest_event"]
                result["latest_event"] = row["latest_event"]
        finally:
            conn.close()
    except Exception as exc:
        log.warning("get_historical_facts(%s) failed: %s", device_id, exc)
    return result


def search_similar_events(
    query_text: str,
    limit: int = _DEFAULT_SEARCH_LIMIT,
    device_classification: Optional[str] = None,
) -> list[dict]:
    """
    Search for historically similar events using FTS5 full-text search.

    The query text should be the user's problem_description (and optionally
    device_information).  Results are ranked by FTS5 BM25 relevance.

    action_classification is EXCLUDED from the returned dicts to avoid
    surfacing post-event information to the user in a way that could
    influence prediction interpretation.

    Parameters
    ----------
    query_text : str
        Text to search for (user's problem description).
    limit : int
        Maximum number of results.
    device_classification : str, optional
        If provided, filter results to this device classification.

    Returns
    -------
    list of dicts with: event_id, event_date, type, reason,
                        device_name, device_classification, mfr_name
    """
    if not query_text or not query_text.strip():
        return []

    # FTS5 requires the query to be escaped
    safe_query = _fts_escape(query_text)

    try:
        conn = _get_connection()
        try:
            if device_classification:
                cur = conn.execute(
                    """
                    SELECT
                        e.event_id,
                        e.event_date,
                        e.type,
                        e.reason,
                        e.device_name,
                        e.device_description,
                        e.device_classification,
                        e.mfr_name,
                        e.mfr_parent_company
                    FROM event e
                    JOIN event_fts f ON e.event_id = f.event_id
                    WHERE event_fts MATCH ?
                      AND e.device_classification = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, device_classification, limit),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT
                        e.event_id,
                        e.event_date,
                        e.type,
                        e.reason,
                        e.device_name,
                        e.device_description,
                        e.device_classification,
                        e.mfr_name,
                        e.mfr_parent_company
                    FROM event e
                    JOIN event_fts f ON e.event_id = f.event_id
                    WHERE event_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, limit),
                )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        log.warning("search_similar_events failed for query '%s': %s", query_text[:50], exc)
        # Fallback: keyword-only search
        return _keyword_fallback_search(query_text, limit, device_classification)


def _fts_escape(query: str) -> str:
    """
    Prepare a user query string for FTS5 MATCH.

    FTS5 is sensitive to special characters (+, -, *, etc.).
    We extract alphanumeric tokens and join them as an OR query.
    """
    import re
    tokens = re.findall(r"[a-zA-Z]{3,}", query)
    if not tokens:
        return query.strip()
    # Use top-10 most distinctive tokens (longest first)
    tokens_sorted = sorted(set(tokens), key=len, reverse=True)[:10]
    return " OR ".join(tokens_sorted)


def _keyword_fallback_search(
    query_text: str,
    limit: int,
    device_classification: Optional[str],
) -> list[dict]:
    """Fallback: simple LIKE search when FTS5 fails."""
    try:
        conn = _get_connection()
        try:
            keywords = [w for w in query_text.split() if len(w) >= 4][:3]
            if not keywords:
                return []
            like_expr = "%" + keywords[0].lower() + "%"
            if device_classification:
                cur = conn.execute(
                    """
                    SELECT event_id, event_date, type, reason,
                           device_name, device_description, device_classification,
                           mfr_name, mfr_parent_company
                    FROM event
                    WHERE LOWER(reason) LIKE ?
                      AND device_classification = ?
                    ORDER BY event_date DESC
                    LIMIT ?
                    """,
                    (like_expr, device_classification, limit),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT event_id, event_date, type, reason,
                           device_name, device_description, device_classification,
                           mfr_name, mfr_parent_company
                    FROM event
                    WHERE LOWER(reason) LIKE ?
                    ORDER BY event_date DESC
                    LIMIT ?
                    """,
                    (like_expr, limit),
                )
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        log.warning("keyword_fallback_search failed: %s", exc)
        return []


def db_healthy() -> bool:
    """Return True if the SQLite database is reachable."""
    try:
        conn = _get_connection()
        conn.execute("SELECT COUNT(*) FROM device LIMIT 1")
        conn.close()
        return True
    except Exception:
        return False
