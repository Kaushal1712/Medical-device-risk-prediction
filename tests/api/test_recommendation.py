"""
tests/api/test_recommendation.py
===================================
Tests for GET /recommendation/{device_id}
"""

import pytest
from tests.api.conftest import (  # noqa: F401
    HIGH_RISK_DEVICE_ID,
    LOW_RISK_DEVICE_ID,
    MED_RISK_DEVICE_ID,
    UNKNOWN_DEVICE_ID,
    UNSCORED_DEVICE_ID,
    client,
)

_VALID_PRIORITIES = {"Critical", "High", "Medium", "Low"}


class TestRecommendation:
    def test_high_risk_device_returns_200(self, client):
        resp = client.get(f"/recommendation/{HIGH_RISK_DEVICE_ID}")
        assert resp.status_code == 200

    def test_recommendation_has_required_fields(self, client):
        data = client.get(f"/recommendation/{HIGH_RISK_DEVICE_ID}").json()
        for field in (
            "device_id", "risk_level", "criticality_tier",
            "maintenance_priority", "recommended_actions",
            "rule_inputs", "disclaimer", "available",
        ):
            assert field in data, f"Missing field: {field}"

    def test_high_risk_device_priority_in_valid_set(self, client):
        data = client.get(f"/recommendation/{HIGH_RISK_DEVICE_ID}").json()
        assert data["available"] is True
        assert data["maintenance_priority"] in _VALID_PRIORITIES

    def test_high_risk_device_priority_is_critical_or_high(self, client):
        """HIGH risk should always yield Critical or High priority."""
        data = client.get(f"/recommendation/{HIGH_RISK_DEVICE_ID}").json()
        assert data["risk_level"] == "HIGH"
        assert data["maintenance_priority"] in ("Critical", "High")

    def test_low_risk_device_priority_is_low_or_medium(self, client):
        data = client.get(f"/recommendation/{LOW_RISK_DEVICE_ID}").json()
        assert data["risk_level"] == "LOW"
        assert data["maintenance_priority"] in ("Low", "Medium")

    def test_recommended_actions_is_nonempty_list(self, client):
        data = client.get(f"/recommendation/{HIGH_RISK_DEVICE_ID}").json()
        assert isinstance(data["recommended_actions"], list)
        assert len(data["recommended_actions"]) >= 1
        for action in data["recommended_actions"]:
            assert isinstance(action, str) and len(action) > 0

    def test_disclaimer_present(self, client):
        data = client.get(f"/recommendation/{HIGH_RISK_DEVICE_ID}").json()
        assert "prototype" in data["disclaimer"].lower() or "decision-support" in data["disclaimer"].lower()

    def test_rule_inputs_populated(self, client):
        data = client.get(f"/recommendation/{HIGH_RISK_DEVICE_ID}").json()
        ri = data["rule_inputs"]
        assert "risk_level" in ri
        assert "criticality_tier" in ri
        assert "calibrated_probability" in ri

    def test_unscored_device_returns_unavailable(self, client):
        data = client.get(f"/recommendation/{UNSCORED_DEVICE_ID}").json()
        assert data["available"] is False
        assert data["unavailable_reason"] != ""

    def test_unknown_device_returns_unavailable(self, client):
        data = client.get(f"/recommendation/{UNKNOWN_DEVICE_ID}").json()
        assert data["available"] is False

    def test_medium_risk_device_has_medium_priority(self, client):
        data = client.get(f"/recommendation/{MED_RISK_DEVICE_ID}").json()
        assert data["risk_level"] == "MEDIUM"
        assert data["maintenance_priority"] in ("High", "Medium")
