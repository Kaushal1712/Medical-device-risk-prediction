"""
tests/api/test_predict.py
==========================
Tests for POST /predict
"""

import pytest
from tests.api.conftest import (  # noqa: F401
    HIGH_RISK_DEVICE_ID,
    LOW_RISK_DEVICE_ID,
    UNKNOWN_DEVICE_ID,
    UNSCORED_DEVICE_ID,
    client,
)


class TestPredict:
    def test_known_high_risk_device_returns_200(self, client):
        resp = client.post("/predict", json={"device_id": HIGH_RISK_DEVICE_ID})
        assert resp.status_code == 200

    def test_predict_available_has_correct_schema(self, client):
        data = client.post("/predict", json={"device_id": HIGH_RISK_DEVICE_ID}).json()
        assert data["prediction_available"] is True
        assert data["device_id"] == HIGH_RISK_DEVICE_ID
        assert "risk_level" in data
        assert data["risk_level"] in ("HIGH", "MEDIUM", "LOW")
        assert "risk_score" in data
        assert 0 <= data["risk_score"] <= 100
        assert "calibrated_probability" in data
        assert 0.0 <= data["calibrated_probability"] <= 1.0
        assert "serving_event_date" in data
        assert "model_version" in data

    def test_predict_low_risk_device(self, client):
        data = client.post("/predict", json={"device_id": LOW_RISK_DEVICE_ID}).json()
        assert data["prediction_available"] is True
        assert data["risk_level"] == "LOW"

    def test_unscored_device_returns_prediction_unavailable(self, client):
        data = client.post("/predict", json={"device_id": UNSCORED_DEVICE_ID}).json()
        assert data["prediction_available"] is False
        assert "unavailable_reason" in data
        assert data["unavailable_reason"] != ""
        # No fabricated score
        assert data.get("risk_score") is None
        assert data.get("risk_level") is None

    def test_malformed_payload_returns_422(self, client):
        # Missing device_id
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_empty_device_id_returns_422(self, client):
        resp = client.post("/predict", json={"device_id": ""})
        assert resp.status_code == 422

    def test_note_field_present(self, client):
        data = client.post("/predict", json={"device_id": HIGH_RISK_DEVICE_ID}).json()
        assert "note" in data
        assert len(data["note"]) > 0

    def test_predict_does_not_recompute_returns_snapshot(self, client):
        """
        The endpoint must return the pre-materialized snapshot.
        Call twice — results should be identical (deterministic serving).
        """
        r1 = client.post("/predict", json={"device_id": HIGH_RISK_DEVICE_ID}).json()
        r2 = client.post("/predict", json={"device_id": HIGH_RISK_DEVICE_ID}).json()
        assert r1["risk_score"] == r2["risk_score"]
        assert r1["risk_level"] == r2["risk_level"]

    def test_predict_unknown_device_returns_unavailable_not_error(self, client):
        """
        An entirely unknown device ID must return prediction_unavailable=False
        (with a reason), not a 500 error or a fabricated score.
        Section 9: /predict returns a valid schema for a known device and a
        clear 'unavailable' for invalid input — not a crash.
        """
        data = client.post("/predict", json={"device_id": UNKNOWN_DEVICE_ID}).json()
        assert data["prediction_available"] is False, (
            f"Unknown device must return prediction_available=False, got: {data}"
        )
        assert data.get("risk_score") is None, (
            "Unknown device must not have a fabricated risk_score"
        )
        assert data.get("risk_level") is None, (
            "Unknown device must not have a fabricated risk_level"
        )
        assert "unavailable_reason" in data, (
            "Unknown device response must include unavailable_reason"
        )
        assert len(data["unavailable_reason"]) > 0
