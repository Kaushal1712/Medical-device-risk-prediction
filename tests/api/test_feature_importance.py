"""
tests/api/test_feature_importance.py
======================================
Tests for GET /feature-importance

Verifies that the endpoint returns the pre-computed global feature importance
list with the correct structure and consistent values.
"""

import pytest
from tests.api.conftest import client  # noqa: F401


class TestFeatureImportance:
    def test_returns_200(self, client):
        resp = client.get("/feature-importance")
        assert resp.status_code == 200

    def test_has_required_top_level_fields(self, client):
        data = client.get("/feature-importance").json()
        for field in ("model_version", "count", "features"):
            assert field in data, f"Missing top-level field: {field}"

    def test_model_version_nonempty(self, client):
        data = client.get("/feature-importance").json()
        assert data["model_version"] != ""

    def test_features_is_nonempty_list(self, client):
        data = client.get("/feature-importance").json()
        assert isinstance(data["features"], list)
        assert len(data["features"]) > 0, "Expected at least one feature"

    def test_count_matches_features_length(self, client):
        data = client.get("/feature-importance").json()
        assert data["count"] == len(data["features"])

    def test_each_item_has_required_fields(self, client):
        data = client.get("/feature-importance").json()
        for item in data["features"]:
            for key in ("feature", "importance", "rank"):
                assert key in item, f"Missing key '{key}' in item: {item}"

    def test_feature_names_are_nonempty_strings(self, client):
        data = client.get("/feature-importance").json()
        for item in data["features"]:
            assert isinstance(item["feature"], str)
            assert item["feature"] != ""

    def test_importance_values_are_nonnegative(self, client):
        data = client.get("/feature-importance").json()
        for item in data["features"]:
            assert item["importance"] >= 0, (
                f"Negative importance for feature '{item['feature']}': {item['importance']}"
            )

    def test_ranks_are_sequential_from_one(self, client):
        data = client.get("/feature-importance").json()
        ranks = [item["rank"] for item in data["features"]]
        expected = list(range(1, len(ranks) + 1))
        assert ranks == expected, f"Ranks are not sequential: {ranks[:10]}"

    def test_importance_is_descending_by_rank(self, client):
        """Higher rank (lower number) should have >= importance than lower rank."""
        data = client.get("/feature-importance").json()
        importances = [item["importance"] for item in data["features"]]
        for i in range(len(importances) - 1):
            assert importances[i] >= importances[i + 1], (
                f"Importance not descending at rank {i+1}: "
                f"{importances[i]} < {importances[i+1]}"
            )

    def test_consistent_across_calls(self, client):
        """Two calls must return identical feature lists (no randomness)."""
        r1 = client.get("/feature-importance").json()
        r2 = client.get("/feature-importance").json()
        assert r1["count"] == r2["count"]
        assert [f["feature"] for f in r1["features"]] == [f["feature"] for f in r2["features"]]
        assert [f["importance"] for f in r1["features"]] == [f["importance"] for f in r2["features"]]
