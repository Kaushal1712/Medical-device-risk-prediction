"""
tests/api/conftest.py
======================
Shared fixtures for API tests.

Uses FastAPI's TestClient (backed by httpx) which runs the ASGI app
in-process — no running server required.

Real artifacts (serving table, merged parquet, model card) are loaded from
the actual project directories. All tests in this suite are marked
`requires_trained_model` and will be skipped if artifacts are absent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Known real device IDs from the production serving table
HIGH_RISK_DEVICE_ID = "80508"
LOW_RISK_DEVICE_ID  = "91519"
MED_RISK_DEVICE_ID  = "82768"   # risk_level=MEDIUM in current snapshot (updated after model retrain)
UNSCORED_DEVICE_ID  = "1"        # present in merged.parquet but not in serving table
UNKNOWN_DEVICE_ID   = "DEVICE_DOES_NOT_EXIST_XYZ_999"


def _artifacts_exist() -> bool:
    from pathlib import Path
    snapshot = Path("artifacts/risk/device_risk_snapshot.parquet")
    model_card = Path("models/production/model_card.json")
    merged = Path("data/processed/merged.parquet")
    return snapshot.exists() and model_card.exists() and merged.exists()


@pytest.fixture(scope="module")
def client():
    """Return a TestClient for the FastAPI app. Skips if artifacts are missing."""
    if not _artifacts_exist():
        pytest.skip("Production artifacts not found — skipping API tests.")
    from backend.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
