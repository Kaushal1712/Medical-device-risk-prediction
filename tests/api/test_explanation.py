"""
tests/api/test_explanation.py
==============================
Tests for GET /explanation/{device_id}

Note: SHAP computation can be slow on first call (model loading + TreeExplainer
initialisation). The module-scoped TestClient fixture keeps the app loaded for
all tests, so SHAP is initialised only once per test session.
"""

import pytest
from tests.api.conftest import (  # noqa: F401
    HIGH_RISK_DEVICE_ID,
    UNKNOWN_DEVICE_ID,
    UNSCORED_DEVICE_ID,
    client,
)


class TestExplanation:
    def test_scoreable_device_returns_200(self, client):
        resp = client.get(f"/explanation/{HIGH_RISK_DEVICE_ID}")
        assert resp.status_code == 200

    def test_scoreable_device_has_required_fields(self, client):
        data = client.get(f"/explanation/{HIGH_RISK_DEVICE_ID}").json()
        for field in (
            "device_id", "model_version", "available",
            "base_value", "predicted_value",
            "top_positive", "top_negative",
        ):
            assert field in data, f"Missing field: {field}"

    def test_scoreable_device_available_true(self, client):
        data = client.get(f"/explanation/{HIGH_RISK_DEVICE_ID}").json()
        assert data["available"] is True
        assert data["device_id"] == HIGH_RISK_DEVICE_ID

    def test_top_positive_is_list_of_contributions(self, client):
        data = client.get(f"/explanation/{HIGH_RISK_DEVICE_ID}").json()
        assert isinstance(data["top_positive"], list)
        for item in data["top_positive"]:
            assert "feature" in item
            assert "shap_value" in item
            assert "direction" in item
            assert item["direction"] == "positive"
            assert item["shap_value"] >= 0

    def test_top_negative_is_list_of_contributions(self, client):
        data = client.get(f"/explanation/{HIGH_RISK_DEVICE_ID}").json()
        assert isinstance(data["top_negative"], list)
        for item in data["top_negative"]:
            assert item["direction"] == "negative"
            assert item["shap_value"] < 0

    def test_model_version_nonempty(self, client):
        data = client.get(f"/explanation/{HIGH_RISK_DEVICE_ID}").json()
        assert data["model_version"] != ""

    def test_unscored_device_returns_available_false(self, client):
        data = client.get(f"/explanation/{UNSCORED_DEVICE_ID}").json()
        assert data["available"] is False
        assert "unavailable_reason" in data
        assert data["unavailable_reason"] != ""
        assert data["top_positive"] == []
        assert data["top_negative"] == []

    def test_unknown_device_returns_available_false(self, client):
        data = client.get(f"/explanation/{UNKNOWN_DEVICE_ID}").json()
        assert data["available"] is False

    def test_cached_result_consistent(self, client):
        """Second call should return identical results (cache hit)."""
        r1 = client.get(f"/explanation/{HIGH_RISK_DEVICE_ID}").json()
        r2 = client.get(f"/explanation/{HIGH_RISK_DEVICE_ID}").json()
        assert r1["base_value"] == r2["base_value"]
        assert r1["predicted_value"] == r2["predicted_value"]
        assert len(r1["top_positive"]) == len(r2["top_positive"])
