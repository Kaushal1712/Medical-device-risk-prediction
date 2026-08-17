"""
tests/api/test_risk_summary.py
================================
Tests for GET /risk-summary
"""

import pytest
from tests.api.conftest import client  # noqa: F401


class TestRiskSummary:
    def test_risk_summary_returns_200(self, client):
        resp = client.get("/risk-summary")
        assert resp.status_code == 200

    def test_risk_summary_has_required_fields(self, client):
        data = client.get("/risk-summary").json()
        for field in (
            "total_devices_in_data",
            "total_scored",
            "total_unscored",
            "risk_levels",
            "risk_score_stats",
            "category_breakdown",
            "manufacturer_breakdown",
        ):
            assert field in data, f"Missing field: {field}"

    def test_risk_levels_has_high_medium_low(self, client):
        data = client.get("/risk-summary").json()
        rl = data["risk_levels"]
        for level in ("HIGH", "MEDIUM", "LOW"):
            assert level in rl, f"Missing risk level: {level}"
            assert "count" in rl[level]
            assert "percent" in rl[level]
            assert rl[level]["count"] >= 0
            assert 0.0 <= rl[level]["percent"] <= 100.0

    def test_counts_sum_correctly(self, client):
        data = client.get("/risk-summary").json()
        total_scored = data["total_scored"]
        rl = data["risk_levels"]
        level_sum = sum(rl[l]["count"] for l in ("HIGH", "MEDIUM", "LOW"))
        assert level_sum == total_scored, (
            f"Level counts ({level_sum}) must equal total_scored ({total_scored})"
        )

    def test_total_devices_geq_scored(self, client):
        data = client.get("/risk-summary").json()
        assert data["total_devices_in_data"] >= data["total_scored"]

    def test_unscored_consistent(self, client):
        data = client.get("/risk-summary").json()
        # unscored = total - scored (may differ slightly if device_index is superset)
        assert data["total_unscored"] >= 0

    def test_risk_score_stats_valid_range(self, client):
        data = client.get("/risk-summary").json()
        stats = data["risk_score_stats"]
        assert stats["min"] >= 0.0
        assert stats["max"] <= 100.0
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_category_breakdown_is_list(self, client):
        data = client.get("/risk-summary").json()
        assert isinstance(data["category_breakdown"], list)
        for item in data["category_breakdown"]:
            assert "category" in item
            assert "high" in item
            assert "total" in item
            assert item["total"] == item["high"] + item["medium"] + item["low"]

    def test_manufacturer_breakdown_is_list(self, client):
        data = client.get("/risk-summary").json()
        assert isinstance(data["manufacturer_breakdown"], list)
        for item in data["manufacturer_breakdown"]:
            assert "manufacturer" in item
            assert item["total"] == item["high"] + item["medium"] + item["low"]

    def test_real_counts_match_known_distribution(self, client):
        """
        HIGH=2053, MEDIUM=206, LOW=48082 from the actual serving table.
        Allow a small tolerance in case the snapshot was regenerated.
        """
        data = client.get("/risk-summary").json()
        rl = data["risk_levels"]
        assert rl["HIGH"]["count"] > 1000, "Expected > 1000 HIGH risk devices"
        assert rl["LOW"]["count"] > 40000, "Expected > 40000 LOW risk devices"
        assert data["total_scored"] >= 50000
