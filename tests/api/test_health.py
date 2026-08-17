"""
tests/api/test_health.py
=========================
Tests for GET /health
"""

import pytest
from tests.api.conftest import client  # noqa: F401


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        for field in ("status", "model_version", "data_manifest_hash", "trained_at", "disclaimer"):
            assert field in data, f"Missing field: {field}"

    def test_health_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_model_version_nonempty(self, client):
        data = client.get("/health").json()
        assert data["model_version"] and data["model_version"] != ""

    def test_health_disclaimer_contains_prototype(self, client):
        data = client.get("/health").json()
        assert "prototype" in data["disclaimer"].lower() or "decision-support" in data["disclaimer"].lower()

    def test_health_manifest_hash_nonempty(self, client):
        data = client.get("/health").json()
        assert data["data_manifest_hash"] and data["data_manifest_hash"] != ""
