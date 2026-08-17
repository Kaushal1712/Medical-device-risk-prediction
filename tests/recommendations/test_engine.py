"""
tests/recommendations/test_engine.py
======================================
Stage 7 — Rule table boundary tests for MaintenanceEngine.

Tests:
- Every (risk_level, criticality_tier) combination in the rule table
- Edge cases: zero events, missing risk_class, unknown risk_level
- Disclaimer is always present
"""

from __future__ import annotations

import pytest
from src.recommendations.engine import (
    DISCLAIMER,
    DeviceContext,
    MaintenanceEngine,
    RecommendationResult,
    _criticality_tier,
)

engine = MaintenanceEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(risk_level: str, risk_class=None, class_i_count=0.0, event_count=0.0, recall_count=0.0):
    return DeviceContext(
        device_id="test-device",
        risk_level=risk_level,
        device_risk_class=risk_class,
        hist_device_class_i_count=class_i_count,
        hist_device_event_count=event_count,
        hist_device_recall_count=recall_count,
    )


# ---------------------------------------------------------------------------
# Criticality tier derivation
# ---------------------------------------------------------------------------

class TestCriticalityTier:
    def test_risk_class_1_is_high(self):
        assert _criticality_tier("1", 0) == "HIGH"

    def test_risk_class_hde_is_high(self):
        assert _criticality_tier("HDE", 0) == "HIGH"

    def test_risk_class_hde_lowercase(self):
        assert _criticality_tier("hde", 0) == "HIGH"

    def test_risk_class_2_is_medium(self):
        assert _criticality_tier("2", 0) == "MEDIUM"

    def test_risk_class_not_classified_is_medium(self):
        assert _criticality_tier("Not Classified", 0) == "MEDIUM"

    def test_risk_class_3_is_low(self):
        assert _criticality_tier("3", 0) == "LOW"

    def test_missing_class_with_class_i_events_is_high(self):
        assert _criticality_tier(None, 2.0) == "HIGH"

    def test_missing_class_no_class_i_events_is_medium(self):
        assert _criticality_tier(None, 0.0) == "MEDIUM"

    def test_class_2_but_has_class_i_events_is_high(self):
        # hist_class_i_count overrides when > 0
        assert _criticality_tier("2", 1.0) == "HIGH"


# ---------------------------------------------------------------------------
# Rule table — HIGH risk
# ---------------------------------------------------------------------------

class TestHighRisk:
    def test_high_risk_high_criticality_is_critical(self):
        result = engine.recommend(_ctx("HIGH", risk_class="1"))
        assert result.maintenance_priority == "Critical"
        assert result.criticality_tier == "HIGH"
        assert result.available is True
        assert len(result.recommended_actions) >= 1

    def test_high_risk_medium_criticality_is_critical(self):
        result = engine.recommend(_ctx("HIGH", risk_class="2"))
        assert result.maintenance_priority == "Critical"

    def test_high_risk_low_criticality_is_high(self):
        result = engine.recommend(_ctx("HIGH", risk_class="3"))
        assert result.maintenance_priority == "High"

    def test_high_risk_no_class_no_events_is_critical(self):
        # No risk_class + no class_i events → criticality=MEDIUM → Critical
        result = engine.recommend(_ctx("HIGH", risk_class=None, class_i_count=0.0))
        assert result.maintenance_priority == "Critical"


# ---------------------------------------------------------------------------
# Rule table — MEDIUM risk
# ---------------------------------------------------------------------------

class TestMediumRisk:
    def test_medium_risk_high_criticality_is_high(self):
        result = engine.recommend(_ctx("MEDIUM", risk_class="1"))
        assert result.maintenance_priority == "High"
        assert result.criticality_tier == "HIGH"

    def test_medium_risk_medium_criticality_is_medium(self):
        result = engine.recommend(_ctx("MEDIUM", risk_class="2"))
        assert result.maintenance_priority == "Medium"

    def test_medium_risk_low_criticality_is_medium(self):
        result = engine.recommend(_ctx("MEDIUM", risk_class="3"))
        assert result.maintenance_priority == "Medium"


# ---------------------------------------------------------------------------
# Rule table — LOW risk
# ---------------------------------------------------------------------------

class TestLowRisk:
    def test_low_risk_high_criticality_is_medium(self):
        result = engine.recommend(_ctx("LOW", risk_class="1"))
        assert result.maintenance_priority == "Medium"

    def test_low_risk_medium_criticality_is_low(self):
        result = engine.recommend(_ctx("LOW", risk_class="2"))
        assert result.maintenance_priority == "Low"

    def test_low_risk_low_criticality_is_low(self):
        result = engine.recommend(_ctx("LOW", risk_class="3"))
        assert result.maintenance_priority == "Low"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_risk_level_returns_unavailable(self):
        result = engine.recommend(_ctx("EXTREME"))
        assert result.available is False
        assert result.maintenance_priority == "Unknown"
        assert "EXTREME" in result.unavailable_reason or "not recognised" in result.unavailable_reason

    def test_empty_risk_level_returns_unavailable(self):
        result = engine.recommend(_ctx(""))
        assert result.available is False

    def test_lowercase_risk_level_normalised(self):
        # Engine normalises to uppercase
        result = engine.recommend(_ctx("high", risk_class="2"))
        assert result.available is True
        assert result.maintenance_priority in ("Critical", "High", "Medium", "Low")

    def test_zero_event_counts_no_crash(self):
        result = engine.recommend(_ctx("LOW", risk_class=None, class_i_count=0.0, event_count=0.0))
        assert result.available is True
        assert isinstance(result.recommended_actions, list)
        assert len(result.recommended_actions) >= 1

    def test_disclaimer_always_present(self):
        for risk_level in ("LOW", "MEDIUM", "HIGH"):
            result = engine.recommend(_ctx(risk_level, risk_class="2"))
            assert DISCLAIMER in result.disclaimer

    def test_rule_inputs_populated(self):
        result = engine.recommend(_ctx("HIGH", risk_class="1", class_i_count=3.0, event_count=5.0))
        assert "risk_level" in result.rule_inputs
        assert "criticality_tier" in result.rule_inputs
        assert result.rule_inputs["hist_device_class_i_count"] == 3.0

    def test_to_dict_roundtrip(self):
        result = engine.recommend(_ctx("MEDIUM", risk_class="2"))
        d = result.to_dict()
        assert d["device_id"] == "test-device"
        assert d["risk_level"] == "MEDIUM"
        assert isinstance(d["recommended_actions"], list)
        assert "disclaimer" in d
