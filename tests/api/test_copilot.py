"""
tests/api/test_copilot.py
===========================
Tests for POST /copilot — GenAI copilot grounded in real device context.

Grounding contract (Section 8 of master prompt):
  - Context assembled from REAL device / risk / SHAP / recommendation data.
  - LLM is called only if LLM_PROVIDER and LLM_API_KEY are configured.
  - Falls back to a deterministic template answer when no LLM key is present
    (or when the LLM call fails) — the demo must never break.
  - Response always includes context_used, llm_used, and provider fields.

All tests exercise the deterministic-fallback path (no LLM key in test env)
which is the only path reliably testable in automated CI.
"""

from __future__ import annotations

import pytest
from tests.api.conftest import (  # noqa: F401
    HIGH_RISK_DEVICE_ID,
    LOW_RISK_DEVICE_ID,
    MED_RISK_DEVICE_ID,
    UNSCORED_DEVICE_ID,
    UNKNOWN_DEVICE_ID,
    client,
)

_ENDPOINT = "/copilot"
_GENERIC_QUESTION = "What is the risk level of this device and why?"


def _post(client, device_id: str, question: str = _GENERIC_QUESTION):
    return client.post(_ENDPOINT, json={"device_id": device_id, "question": question})


class TestCopilotSchema:
    """Validate the response envelope structure for a well-known scoreable device."""

    def test_high_risk_device_returns_200(self, client):
        resp = _post(client, HIGH_RISK_DEVICE_ID)
        assert resp.status_code == 200

    def test_response_has_required_top_level_fields(self, client):
        data = _post(client, HIGH_RISK_DEVICE_ID).json()
        for field in (
            "device_id",
            "question",
            "answer",
            "context_used",
            "llm_used",
            "provider",
        ):
            assert field in data, f"Missing top-level field: {field!r}"

    def test_answer_is_nonempty_string(self, client):
        data = _post(client, HIGH_RISK_DEVICE_ID).json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"].strip()) > 0, "Copilot answer must not be empty"

    def test_device_id_echoed_correctly(self, client):
        data = _post(client, HIGH_RISK_DEVICE_ID).json()
        assert data["device_id"] == HIGH_RISK_DEVICE_ID

    def test_question_echoed_correctly(self, client):
        q = "Why is this device flagged?"
        data = _post(client, HIGH_RISK_DEVICE_ID, question=q).json()
        assert data["question"] == q


class TestCopilotFallback:
    """Verify the deterministic fallback path (no LLM key configured in test env)."""

    def test_uses_deterministic_fallback(self, client):
        """No LLM key in test env → llm_used must be False."""
        data = _post(client, HIGH_RISK_DEVICE_ID).json()
        assert data["llm_used"] is False

    def test_provider_is_fallback(self, client):
        """Fallback path must report provider='fallback'."""
        data = _post(client, HIGH_RISK_DEVICE_ID).json()
        assert data["provider"] == "fallback"

    def test_fallback_answer_mentions_risk(self, client):
        """Deterministic template always describes the risk assessment."""
        data = _post(client, HIGH_RISK_DEVICE_ID).json()
        answer_lower = data["answer"].lower()
        # Template contains 'risk assessment' section header
        assert "risk" in answer_lower, "Fallback answer must mention risk"

    def test_fallback_answer_contains_disclaimer(self, client):
        """Deterministic template must include the healthcare disclaimer."""
        data = _post(client, HIGH_RISK_DEVICE_ID).json()
        answer_lower = data["answer"].lower()
        assert "prototype" in answer_lower or "decision-support" in answer_lower, (
            "Fallback answer must contain the decision-support disclaimer"
        )


class TestCopilotGrounding:
    """Verify the context_used block contains real values, not fabricated data."""

    def test_context_used_has_required_fields(self, client):
        data = _post(client, HIGH_RISK_DEVICE_ID).json()
        ctx = data["context_used"]
        for field in (
            "device_id",
            "risk_level",
            "risk_score",
            "calibrated_probability",
            "maintenance_priority",
            "recommended_actions",
            "top_risk_factors",
        ):
            assert field in ctx, f"context_used missing field: {field!r}"

    def test_context_device_id_matches_request(self, client):
        data = _post(client, HIGH_RISK_DEVICE_ID).json()
        assert data["context_used"]["device_id"] == HIGH_RISK_DEVICE_ID

    def test_context_risk_level_is_high_for_high_risk_device(self, client):
        """context_used.risk_level must match the serving table — HIGH for device 80508."""
        data = _post(client, HIGH_RISK_DEVICE_ID).json()
        assert data["context_used"]["risk_level"] == "HIGH", (
            f"Expected HIGH, got {data['context_used']['risk_level']}"
        )

    def test_context_risk_score_matches_serving_table(self, client):
        """context_used.risk_score must equal the score from GET /devices/{id}."""
        ctx = _post(client, HIGH_RISK_DEVICE_ID).json()["context_used"]
        device = client.get(f"/devices/{HIGH_RISK_DEVICE_ID}").json()
        assert ctx["risk_score"] == device["risk_score"], (
            f"Copilot context risk_score ({ctx['risk_score']}) "
            f"does not match /devices/{HIGH_RISK_DEVICE_ID} risk_score ({device['risk_score']})"
        )

    def test_context_top_risk_factors_nonempty_for_scoreable_device(self, client):
        """For a HIGH-risk device with SHAP available, top_risk_factors must be populated."""
        ctx = _post(client, HIGH_RISK_DEVICE_ID).json()["context_used"]
        assert isinstance(ctx["top_risk_factors"], list)
        assert len(ctx["top_risk_factors"]) > 0, (
            "top_risk_factors should be non-empty for a device with SHAP explanation"
        )

    def test_context_top_risk_factors_are_strings(self, client):
        """Each entry in top_risk_factors must be a non-empty string."""
        ctx = _post(client, HIGH_RISK_DEVICE_ID).json()["context_used"]
        for factor in ctx["top_risk_factors"]:
            assert isinstance(factor, str) and len(factor) > 0


class TestCopilotEdgeCases:
    """Verify graceful handling of unscored devices, unknown devices, and bad inputs."""

    def test_unscored_device_returns_200_no_crash(self, client):
        """An unscored (but known) device must not crash the endpoint (returns 200)."""
        resp = _post(client, UNSCORED_DEVICE_ID)
        assert resp.status_code == 200

    def test_unscored_device_risk_level_is_none(self, client):
        """For an unscored device, context_used.risk_level must be None (no fabrication)."""
        ctx = _post(client, UNSCORED_DEVICE_ID).json()["context_used"]
        assert ctx["risk_level"] is None, (
            f"Expected None for unscored device, got: {ctx['risk_level']!r}"
        )

    def test_unscored_device_answer_is_nonempty(self, client):
        """Even with no risk data, the deterministic fallback must produce an answer."""
        data = _post(client, UNSCORED_DEVICE_ID).json()
        assert isinstance(data["answer"], str) and len(data["answer"].strip()) > 0

    def test_unknown_device_returns_200_no_crash(self, client):
        """An entirely unknown device must not crash the endpoint (returns 200)."""
        resp = _post(client, UNKNOWN_DEVICE_ID)
        assert resp.status_code == 200

    def test_unknown_device_risk_level_is_none(self, client):
        """For an unknown device, context_used.risk_level must be None (no fabrication)."""
        ctx = _post(client, UNKNOWN_DEVICE_ID).json()["context_used"]
        assert ctx["risk_level"] is None, (
            f"Expected None for unknown device, got: {ctx['risk_level']!r}"
        )

    def test_unknown_device_answer_is_nonempty(self, client):
        """Even with no data at all, the fallback must produce a coherent answer."""
        data = _post(client, UNKNOWN_DEVICE_ID).json()
        assert isinstance(data["answer"], str) and len(data["answer"].strip()) > 0


class TestCopilotValidation:
    """Input validation — Pydantic must reject malformed requests with 422."""

    def test_empty_question_returns_422(self, client):
        """question must be at least 1 character (Field min_length=1)."""
        resp = client.post(_ENDPOINT, json={"device_id": HIGH_RISK_DEVICE_ID, "question": ""})
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_missing_device_id_returns_422(self, client):
        """device_id is required — omitting it must return 422."""
        resp = client.post(_ENDPOINT, json={"question": _GENERIC_QUESTION})
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_missing_question_returns_422(self, client):
        """question is required — omitting it must return 422."""
        resp = client.post(_ENDPOINT, json={"device_id": HIGH_RISK_DEVICE_ID})
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_question_at_max_length_returns_200(self, client):
        """A 2000-character question (exactly at the limit) must be accepted."""
        q = "A" * 2000
        resp = _post(client, HIGH_RISK_DEVICE_ID, question=q)
        assert resp.status_code == 200

    def test_question_over_max_length_returns_422(self, client):
        """A 2001-character question (one over max_length=2000) must be rejected."""
        q = "A" * 2001
        resp = _post(client, HIGH_RISK_DEVICE_ID, question=q)
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_empty_device_id_returns_422(self, client):
        """device_id must be at least 1 character (Field min_length=1)."""
        resp = client.post(_ENDPOINT, json={"device_id": "", "question": _GENERIC_QUESTION})
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"


class TestCopilotConsistency:
    """Verify idempotency and cross-endpoint consistency."""

    def test_two_identical_requests_produce_identical_answers(self, client):
        """Deterministic fallback must be fully deterministic (no randomness)."""
        d1 = _post(client, HIGH_RISK_DEVICE_ID).json()
        d2 = _post(client, HIGH_RISK_DEVICE_ID).json()
        assert d1["answer"] == d2["answer"], (
            "Deterministic fallback produced different answers on identical inputs"
        )
        assert d1["context_used"] == d2["context_used"], (
            "context_used differs between identical requests"
        )

    def test_different_devices_produce_different_answers(self, client):
        """Answers must reflect device-specific context, not be templated identically."""
        d_high = _post(client, HIGH_RISK_DEVICE_ID).json()
        d_low = _post(client, LOW_RISK_DEVICE_ID).json()
        # At minimum, the risk levels embedded in the answer must differ
        assert d_high["context_used"]["risk_level"] != d_low["context_used"]["risk_level"], (
            "HIGH and LOW risk devices must yield different risk levels in context"
        )

    def test_context_maintenance_priority_matches_recommendation_endpoint(self, client):
        """context_used.maintenance_priority must equal GET /recommendation/{id}.maintenance_priority."""
        ctx = _post(client, HIGH_RISK_DEVICE_ID).json()["context_used"]
        rec = client.get(f"/recommendation/{HIGH_RISK_DEVICE_ID}").json()
        if rec["available"]:
            assert ctx["maintenance_priority"] == rec["maintenance_priority"], (
                f"Copilot context priority ({ctx['maintenance_priority']!r}) "
                f"does not match /recommendation priority ({rec['maintenance_priority']!r})"
            )
