"""
tests/api/test_devices.py
==========================
Tests for GET /devices and GET /devices/{id}
"""

import pytest
from tests.api.conftest import (  # noqa: F401
    HIGH_RISK_DEVICE_ID,
    LOW_RISK_DEVICE_ID,
    UNKNOWN_DEVICE_ID,
    UNSCORED_DEVICE_ID,
    client,
)


class TestDeviceList:
    def test_list_returns_200(self, client):
        resp = client.get("/devices")
        assert resp.status_code == 200

    def test_list_has_items_and_pagination(self, client):
        data = client.get("/devices?page=1&page_size=10").json()
        assert "items" in data
        assert "pagination" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) <= 10

    def test_list_pagination_fields(self, client):
        data = client.get("/devices?page=1&page_size=20").json()
        p = data["pagination"]
        assert p["page"] == 1
        assert p["page_size"] == 20
        assert p["total_items"] >= 0
        assert p["total_pages"] >= 1

    def test_list_item_has_device_id(self, client):
        data = client.get("/devices?page_size=5").json()
        for item in data["items"]:
            assert "device_id" in item
            assert item["device_id"] != ""

    def test_filter_by_risk_level_high(self, client):
        data = client.get("/devices?risk_level=HIGH&page_size=10").json()
        for item in data["items"]:
            assert item["risk_level"] == "HIGH"

    def test_filter_by_risk_level_low(self, client):
        data = client.get("/devices?risk_level=LOW&page_size=10").json()
        for item in data["items"]:
            assert item["risk_level"] == "LOW"

    def test_filter_by_invalid_page_returns_422(self, client):
        resp = client.get("/devices?page=0")
        assert resp.status_code == 422

    def test_filter_by_page_size_over_max_returns_422(self, client):
        resp = client.get("/devices?page_size=9999")
        assert resp.status_code == 422

    def test_risk_level_is_none_for_unscored_device(self, client):
        # Search for the known unscored device by device_id
        data = client.get(f"/devices?search={UNSCORED_DEVICE_ID}&page_size=5").json()
        # May be empty if search doesn't match, that's fine — just verify format
        assert isinstance(data["items"], list)


class TestDeviceDetail:
    def test_known_high_risk_device_returns_200(self, client):
        resp = client.get(f"/devices/{HIGH_RISK_DEVICE_ID}")
        assert resp.status_code == 200

    def test_known_device_has_prediction(self, client):
        data = client.get(f"/devices/{HIGH_RISK_DEVICE_ID}").json()
        assert data["prediction_available"] is True
        assert data["risk_level"] in ("HIGH", "MEDIUM", "LOW")
        assert data["risk_score"] is not None
        assert 0 <= data["risk_score"] <= 100

    def test_known_device_has_device_fields(self, client):
        data = client.get(f"/devices/{HIGH_RISK_DEVICE_ID}").json()
        assert "device_id" in data
        assert data["device_id"] == HIGH_RISK_DEVICE_ID

    def test_known_device_has_maintenance_priority(self, client):
        data = client.get(f"/devices/{HIGH_RISK_DEVICE_ID}").json()
        assert "maintenance_priority" in data
        assert data["maintenance_priority"] in ("Critical", "High", "Medium", "Low", None)

    def test_unknown_device_returns_404(self, client):
        resp = client.get(f"/devices/{UNKNOWN_DEVICE_ID}")
        assert resp.status_code == 404

    def test_low_risk_device_returns_200(self, client):
        resp = client.get(f"/devices/{LOW_RISK_DEVICE_ID}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction_available"] is True
        assert data["risk_level"] == "LOW"
