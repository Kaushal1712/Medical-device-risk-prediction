"""
tests/api/test_assess.py
=========================
Tests for POST /assess  — new query-driven risk assessment workflow.

These tests verify that:
  1. The endpoint accepts valid requests and returns 200
  2. device_id is never used as a predictive ML feature
  3. The prediction changes when the problem_description changes materially
  4. The response schema is complete
  5. Validation errors are returned for malformed requests
  6. The disclaimer and limitations are always present

Tests run against the real API (same as other API tests) — they require
the production artifacts to be present.
"""

import pytest
from tests.api.conftest import client  # noqa: F401


# ---------------------------------------------------------------------------
# Valid representative requests
# ---------------------------------------------------------------------------

CARDIAC_DEFIBRILLATOR_REQUEST = {
    "device_information": "Implanted cardiac defibrillator, model ICD-X200",
    "problem_description": (
        "Repeated electrical faults and abnormal battery behavior observed. "
        "Device delivering inappropriate shocks."
    ),
    "device_classification": "Cardiovascular Devices",
    "device_risk_class": "3",
    "device_implanted": "YES",
    "country": "USA",
}

LABELING_ERROR_REQUEST = {
    "device_information": "Surgical scissors",
    "problem_description": "Minor labeling error on outer packaging only, no patient risk.",
}

VENTILATOR_REQUEST = {
    "device_information": "ICU ventilator",
    "problem_description": (
        "Device failure during patient operation, requiring emergency manual ventilation."
    ),
    "device_classification": "Anesthesiology Devices",
}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestAssessSchema:
    def test_returns_200_with_valid_request(self, client):
        resp = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_response_has_prediction_block(self, client):
        data = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST).json()
        assert "prediction" in data
        pred = data["prediction"]
        assert "risk_level" in pred
        assert "risk_score" in pred
        assert "raw_probability" in pred
        assert "model_version" in pred
        assert "target_description" in pred

    def test_risk_level_is_valid_string(self, client):
        data = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST).json()
        assert data["prediction"]["risk_level"] in ("HIGH", "MEDIUM", "LOW")

    def test_risk_score_is_in_range(self, client):
        data = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST).json()
        score = data["prediction"]["risk_score"]
        assert 0.0 <= score <= 100.0, f"risk_score {score} out of range [0, 100]"

    def test_raw_probability_is_in_range(self, client):
        data = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST).json()
        prob = data["prediction"]["raw_probability"]
        assert 0.0 <= prob <= 1.0, f"raw_probability {prob} out of [0, 1]"

    def test_response_has_historical_evidence(self, client):
        data = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST).json()
        assert "historical_evidence" in data
        he = data["historical_evidence"]
        assert "device_events" in he
        assert "similar_events" in he
        assert isinstance(he["device_events"], list)
        assert isinstance(he["similar_events"], list)

    def test_response_has_disclaimer(self, client):
        data = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST).json()
        assert "disclaimer" in data
        assert len(data["disclaimer"]) > 20

    def test_response_has_limitations(self, client):
        data = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST).json()
        assert "limitations" in data
        assert isinstance(data["limitations"], list)
        assert len(data["limitations"]) > 0

    def test_target_description_mentions_class_i(self, client):
        data = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST).json()
        target_desc = data["prediction"]["target_description"].lower()
        assert "class i" in target_desc or "class 1" in target_desc, (
            "target_description must mention 'Class I' to avoid misinterpretation"
        )

    def test_target_description_does_not_claim_future_failure(self, client):
        data = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST).json()
        target_desc = data["prediction"]["target_description"].lower()
        # The description must NOT claim to predict future failure
        assert "future" not in target_desc or "fail" not in target_desc, (
            "target_description must not claim to predict future device failure"
        )


# ---------------------------------------------------------------------------
# Prediction discrimination
# ---------------------------------------------------------------------------

class TestAssessPredictionDiscrimination:
    def test_high_severity_description_vs_low_severity(self, client):
        """
        A device with a severe safety-critical problem should score higher than
        one with a minor labeling error.
        """
        high_resp = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST).json()
        low_resp = client.post("/assess", json=LABELING_ERROR_REQUEST).json()

        high_score = high_resp["prediction"]["raw_probability"]
        low_score = low_resp["prediction"]["raw_probability"]

        assert high_score > low_score, (
            f"Severe problem should have higher probability than minor error. "
            f"Got high={high_score:.4f}, low={low_score:.4f}"
        )

    def test_problem_description_affects_prediction(self, client):
        """Changing only the problem_description should change the prediction."""
        req_a = {
            "device_information": "Pacemaker",
            "problem_description": "Complete device failure, no output, patient in danger",
        }
        req_b = {
            "device_information": "Pacemaker",
            "problem_description": "Routine firmware update, no safety concerns",
        }
        resp_a = client.post("/assess", json=req_a).json()["prediction"]
        resp_b = client.post("/assess", json=req_b).json()["prediction"]

        # Raw probabilities should differ
        assert resp_a["raw_probability"] != resp_b["raw_probability"], (
            "Prediction must change when problem_description changes"
        )


# ---------------------------------------------------------------------------
# Device ID as context only (not feature)
# ---------------------------------------------------------------------------

class TestAssessDeviceIdIsContextOnly:
    def test_works_without_device_id(self, client):
        """Endpoint must work without device_id."""
        req = {
            "device_information": "Blood glucose monitor",
            "problem_description": "Incorrect readings causing false low values",
        }
        resp = client.post("/assess", json=req)
        assert resp.status_code == 200

    def test_with_device_id_returns_device_info(self, client):
        """When device_id is provided, device_info is populated."""
        req = {**CARDIAC_DEFIBRILLATOR_REQUEST, "device_id": "80508"}
        data = client.post("/assess", json=req).json()
        # device_info may be populated or null (80508 might not match this category)
        # but no crash should occur
        assert "device_info" in data

    def test_device_id_does_not_change_prediction(self, client):
        """
        Same device_information + problem_description must yield the same
        prediction regardless of whether device_id is provided.
        The device_id must not be used as a feature.
        """
        base_req = {
            "device_information": "Implanted pacemaker",
            "problem_description": "Intermittent pacing failure at rest.",
        }
        with_id = {**base_req, "device_id": "80508"}
        without_id = {**base_req}

        resp_with = client.post("/assess", json=with_id).json()["prediction"]
        resp_without = client.post("/assess", json=without_id).json()["prediction"]

        assert resp_with["raw_probability"] == resp_without["raw_probability"], (
            "Providing device_id must NOT change the raw_probability — "
            "device_id must not be used as a predictive ML feature. "
            f"with_id={resp_with['raw_probability']:.6f}, "
            f"without_id={resp_without['raw_probability']:.6f}"
        )

    def test_with_device_id_similar_events_still_present(self, client):
        """FTS similar-event search should work with or without device_id."""
        req = {
            "device_information": "Cardiac defibrillator",
            "problem_description": "Battery failure and unexpected device shutdown",
            "device_id": "80508",
        }
        data = client.post("/assess", json=req).json()
        # similar_events may be empty or populated — we just check no crash
        assert isinstance(data["historical_evidence"]["similar_events"], list)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestAssessValidation:
    def test_missing_device_information_returns_422(self, client):
        resp = client.post("/assess", json={
            "problem_description": "Device overheating during operation"
        })
        assert resp.status_code == 422

    def test_missing_problem_description_returns_422(self, client):
        resp = client.post("/assess", json={
            "device_information": "IV pump"
        })
        assert resp.status_code == 422

    def test_too_short_device_information_returns_422(self, client):
        resp = client.post("/assess", json={
            "device_information": "X",  # min_length=3
            "problem_description": "Device malfunction observed"
        })
        assert resp.status_code == 422

    def test_too_short_problem_description_returns_422(self, client):
        resp = client.post("/assess", json={
            "device_information": "Ventilator",
            "problem_description": "Bug",  # min_length=10
        })
        assert resp.status_code == 422

    def test_empty_payload_returns_422(self, client):
        resp = client.post("/assess", json={})
        assert resp.status_code == 422

    def test_model_version_is_nonempty(self, client):
        data = client.post("/assess", json=CARDIAC_DEFIBRILLATOR_REQUEST).json()
        assert len(data["prediction"]["model_version"]) > 0
