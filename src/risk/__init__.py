"""
src/risk
========
Stage 6 — Risk Scoring Engine.

Provides probability calibration, 0–100 risk scoring, LOW/MEDIUM/HIGH
risk band classification, and serving table materialisation.

Public API
----------
  RiskScorer       — model loader + scorer
  ScoringResult    — structured result dataclass
  probability_to_score(p)            -> float in [0, 100]
  score_to_band(p, t_medium, t_high) -> "LOW" | "MEDIUM" | "HIGH"

Scripts
-------
  python -m src.risk.calibrate           # fit calibration + derive thresholds
  python -m src.risk.build_serving_table # materialise serving table
"""

from src.risk.scorer import (
    RiskScorer,
    ScoringResult,
    probability_to_score,
    score_to_band,
)

__all__ = [
    "RiskScorer",
    "ScoringResult",
    "probability_to_score",
    "score_to_band",
]
