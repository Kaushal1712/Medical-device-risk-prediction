"""
src/recommendations/engine.py
==============================
Stage 7 — Rule-based Maintenance Decision Engine.

Combines risk level (LOW / MEDIUM / HIGH from the Stage 6 serving table)
with a device criticality proxy derived from the actual data columns:
  - device_risk_class  (1 / 2 / 3 / HDE / Not Classified / Unclassified)
  - hist_device_class_i_count  (count of historical Class-I severity events)
  - hist_device_event_count    (total historical events)
  - hist_device_recall_count   (historical recall events)

No ML model is used. The rule table is documented in docs/07_recommendations_rules.md.

Public API
----------
  MaintenanceEngine()
      .recommend(context: DeviceContext) -> RecommendationResult

  DeviceContext  — input struct (risk_level + device attributes)
  RecommendationResult — output struct
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Healthcare disclaimer (must accompany all recommendations)
# ---------------------------------------------------------------------------
DISCLAIMER = (
    "This system is a decision-support prototype and does not replace qualified "
    "maintenance, biomedical engineering, regulatory, or clinical judgment. "
    "It is not a certified medical device and does not guarantee patient safety outcomes."
)

# ---------------------------------------------------------------------------
# Criticality levels derived from device_risk_class
# Risk class 1 is the most severe FDA recall class (matches our target variable).
# HDE (Humanitarian Device Exemption) devices are also high-stakes.
# ---------------------------------------------------------------------------
_HIGH_CRITICALITY_CLASSES = frozenset({"1", "hde"})
_MEDIUM_CRITICALITY_CLASSES = frozenset({"2", "not classified", "unclassified"})
# Risk class 3 is least-restrictive FDA class → treated as lower criticality here
_LOW_CRITICALITY_CLASSES = frozenset({"3"})


def _criticality_tier(risk_class: Optional[str], hist_class_i_count: float) -> str:
    """
    Derive criticality tier: 'HIGH' | 'MEDIUM' | 'LOW'.

    Uses device_risk_class as the primary signal. Falls back to
    hist_device_class_i_count if risk_class is missing.
    """
    if risk_class:
        normalized = risk_class.strip().lower()
        if normalized in _HIGH_CRITICALITY_CLASSES or (
            # if device has ever had a Class I event, treat as high criticality
            hist_class_i_count and hist_class_i_count > 0
        ):
            return "HIGH"
        if normalized in _MEDIUM_CRITICALITY_CLASSES:
            return "MEDIUM"
        if normalized in _LOW_CRITICALITY_CLASSES:
            return "LOW"

    # risk_class missing — use hist_class_i_count as proxy
    if hist_class_i_count and hist_class_i_count > 0:
        return "HIGH"
    return "MEDIUM"  # conservative default when data is absent


# ---------------------------------------------------------------------------
# Rule table
#
# (risk_level, criticality_tier) → (priority, action_list)
#
# Priority ranks: "Critical" > "High" > "Medium" > "Low"
# ---------------------------------------------------------------------------
_RULE_TABLE: dict[tuple[str, str], tuple[str, list[str]]] = {
    # HIGH risk
    ("HIGH", "HIGH"):   ("Critical", [
        "Immediately remove from service pending safety review.",
        "Schedule emergency preventive inspection.",
        "Escalate to biomedical engineering and risk officer.",
        "Review all historical Class I event records.",
    ]),
    ("HIGH", "MEDIUM"): ("Critical", [
        "Prioritize for immediate preventive inspection.",
        "Escalate to biomedical engineering within 24 hours.",
        "Review all historical failure-related events.",
    ]),
    ("HIGH", "LOW"):    ("High", [
        "Schedule preventive inspection this week.",
        "Review historical event records.",
        "Monitor closely until inspection is completed.",
    ]),
    # MEDIUM risk
    ("MEDIUM", "HIGH"): ("High", [
        "Schedule preventive inspection within 7 days.",
        "Review historical Class I event records.",
        "Apply enhanced monitoring protocol.",
    ]),
    ("MEDIUM", "MEDIUM"): ("Medium", [
        "Schedule inspection within 30 days.",
        "Review historical event records.",
        "Continue standard monitoring.",
    ]),
    ("MEDIUM", "LOW"):  ("Medium", [
        "Schedule inspection within 30 days.",
        "Continue standard monitoring.",
    ]),
    # LOW risk
    ("LOW", "HIGH"):    ("Medium", [
        "Monitor closely; review historical safety information.",
        "Ensure preventive maintenance is up to date.",
    ]),
    ("LOW", "MEDIUM"):  ("Low", [
        "Continue routine monitoring and standard maintenance schedule.",
    ]),
    ("LOW", "LOW"):     ("Low", [
        "Continue routine monitoring and standard maintenance schedule.",
    ]),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DeviceContext:
    """
    Input context for the maintenance decision engine.

    All numeric fields default to 0.0 (safe fallback — engine will not crash
    on missing features).
    """
    device_id: str
    risk_level: str                        # "LOW" | "MEDIUM" | "HIGH"
    risk_score: float = 0.0               # 0–100 risk score from serving table
    calibrated_probability: float = 0.0   # raw calibrated probability [0, 1]

    # Criticality proxy — device_risk_class column from merged.parquet
    device_risk_class: Optional[str] = None

    # Historical event counts from feature Parquet
    hist_device_event_count: float = 0.0
    hist_device_class_i_count: float = 0.0
    hist_device_recall_count: float = 0.0

    # Serving snapshot metadata (for traceability)
    serving_event_date: Optional[str] = None
    model_version: Optional[str] = None


@dataclass
class RecommendationResult:
    """Output of the maintenance decision engine for one device."""
    device_id: str
    risk_level: str
    criticality_tier: str                  # derived from rules above
    maintenance_priority: str              # "Critical" | "High" | "Medium" | "Low"
    recommended_actions: list[str] = field(default_factory=list)
    rule_inputs: dict = field(default_factory=dict)    # inputs used in rule evaluation
    disclaimer: str = DISCLAIMER
    available: bool = True
    unavailable_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "risk_level": self.risk_level,
            "criticality_tier": self.criticality_tier,
            "maintenance_priority": self.maintenance_priority,
            "recommended_actions": self.recommended_actions,
            "rule_inputs": self.rule_inputs,
            "disclaimer": self.disclaimer,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MaintenanceEngine:
    """
    Rule-based maintenance decision engine.

    Stateless: no model, no external I/O. Instantiate once and call .recommend().
    """

    def recommend(self, ctx: DeviceContext) -> RecommendationResult:
        """
        Apply the rule table to produce a maintenance recommendation.

        Parameters
        ----------
        ctx : DeviceContext
            Device risk and attribute context.

        Returns
        -------
        RecommendationResult
        """
        risk_level = (ctx.risk_level or "").strip().upper()
        if risk_level not in ("LOW", "MEDIUM", "HIGH"):
            return RecommendationResult(
                device_id=ctx.device_id,
                risk_level=risk_level or "UNKNOWN",
                criticality_tier="UNKNOWN",
                maintenance_priority="Unknown",
                available=False,
                unavailable_reason=(
                    f"Risk level '{ctx.risk_level}' is not recognised. "
                    "Expected LOW, MEDIUM, or HIGH."
                ),
            )

        criticality = _criticality_tier(
            ctx.device_risk_class,
            ctx.hist_device_class_i_count,
        )

        priority, actions = _RULE_TABLE.get(
            (risk_level, criticality),
            ("Medium", ["Continue standard monitoring."])   # safe default
        )

        rule_inputs = {
            "risk_level": risk_level,
            "criticality_tier": criticality,
            "device_risk_class": ctx.device_risk_class,
            "hist_device_event_count": ctx.hist_device_event_count,
            "hist_device_class_i_count": ctx.hist_device_class_i_count,
            "hist_device_recall_count": ctx.hist_device_recall_count,
            "calibrated_probability": round(ctx.calibrated_probability, 4),
            "risk_score": round(ctx.risk_score, 2),
            "serving_event_date": ctx.serving_event_date,
            "criticality_proxy_note": (
                "device_risk_class used as criticality proxy. "
                "No explicit criticality field was found in the dataset."
                if ctx.device_risk_class
                else
                "device_risk_class missing; hist_device_class_i_count used as fallback proxy."
            ),
        }

        log.debug(
            "MaintenanceEngine: device=%s  risk=%s  criticality=%s  priority=%s",
            ctx.device_id,
            risk_level,
            criticality,
            priority,
        )

        return RecommendationResult(
            device_id=ctx.device_id,
            risk_level=risk_level,
            criticality_tier=criticality,
            maintenance_priority=priority,
            recommended_actions=actions,
            rule_inputs=rule_inputs,
        )
