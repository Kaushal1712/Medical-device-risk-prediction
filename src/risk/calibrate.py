"""
src/risk/calibrate.py
=====================
Stage 6 — Probability Calibration + Risk Band Threshold Derivation.

Fits isotonic calibration on the training split only (leakage-safe),
computes before/after Brier score and ECE on validation, derives
RISK_THRESHOLD_MEDIUM and RISK_THRESHOLD_HIGH from the calibrated
validation precision/recall curve, and writes back the thresholds
into src/config.py.

Run:
    python -m src.risk.calibrate

Prerequisites:
    python -m src.models.train   (models/experiments/ populated)
    python -m src.models.evaluate (models/production/model.pkl exists)

Outputs:
    models/production/calibrated_model.pkl
    models/production/calibration_report.json
    src/config.py  — RISK_THRESHOLD_HIGH and RISK_THRESHOLD_MEDIUM updated
"""

import json
import logging
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CALIBRATION_METHOD,
    CALIBRATION_REPORT_PATH,
    CALIBRATED_MODEL_PATH,
    FEATURES_DATA_DIR,
    PRODUCTION_MODEL_DIR,
    RISK_SCORE_VERSION,
)
from src.risk.scorer import RiskScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

METADATA_COLS = frozenset(
    {"id", "device_id", "manufacturer_id", "event_date", "event_date_available", "is_class_i"}
)


def load_split(name: str):
    """Load a feature split, return (X float32, y int, feature_cols)."""
    df = pd.read_parquet(FEATURES_DATA_DIR / f"{name}.parquet")
    feat_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feat_cols].values.astype(np.float32)
    y = df["is_class_i"].values.astype(int)
    return X, y, feat_cols


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) — weighted mean abs calibration gap."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_proba[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_reliability_curve(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10):
    """Return (mean_pred, frac_pos) arrays for a reliability diagram."""
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    return mean_pred.tolist(), frac_pos.tolist()


def derive_thresholds(
    y_val: np.ndarray,
    cal_proba_val: np.ndarray,
) -> dict:
    """
    Derive MEDIUM and HIGH band thresholds from the calibrated validation
    precision/recall curve, with documented business reasoning.

    Business reasoning (regulatory safety context):
    - A missed HIGH-risk device (false negative on HIGH) carries significant
      patient-safety cost: the device goes uninspected when it should be flagged.
    - A false HIGH flag (false positive) wastes inspection effort and causes alarm
      fatigue, but is far less costly than a missed recall/safety event.
    - Therefore: the HIGH band boundary prioritises RECALL >= 0.35 while
      maximising precision subject to that constraint. This ensures the HIGH
      band catches a material fraction of Class I events rather than being
      nearly empty (which would happen with a very high precision cutoff).
    - The MEDIUM band boundary is set at the F1-maximising threshold, capturing
      the "monitor closely" population — balanced recall/precision.

    Both thresholds are computed from calibrated probabilities on the validation
    set only (no test/holdout data used).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, cal_proba_val)
    # precision_recall_curve returns n+1 values; thresholds has n values
    # precisions[-1]=1.0, recalls[-1]=0.0 (sentinel) — trim them
    prec = precisions[:-1]
    rec = recalls[:-1]
    thr = thresholds

    # F1 curve
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)

    # --- MEDIUM threshold: maximise F1 ---
    medium_idx = int(np.argmax(f1))
    t_medium = float(thr[medium_idx])
    medium_prec = float(prec[medium_idx])
    medium_rec = float(rec[medium_idx])
    medium_f1 = float(f1[medium_idx])

    # --- HIGH threshold: highest threshold where recall >= 0.35 ---
    # (ensures HIGH band catches at least 35% of Class I events on validation)
    TARGET_RECALL_HIGH = 0.35
    valid_mask = rec >= TARGET_RECALL_HIGH
    if valid_mask.any():
        # Among thresholds with sufficient recall, take the highest (most precise)
        high_idx = int(np.where(valid_mask)[0][-1])
        t_high = float(thr[high_idx])
        high_prec = float(prec[high_idx])
        high_rec = float(rec[high_idx])
        high_f1 = float(f1[high_idx])
    else:
        # Fallback: use the threshold with best F1 (same as MEDIUM — will be bumped)
        high_idx = medium_idx
        t_high = t_medium
        high_prec = medium_prec
        high_rec = medium_rec
        high_f1 = medium_f1
        log.warning(
            "No calibrated threshold achieves recall >= %.2f on validation. "
            "Setting T_HIGH = T_MEDIUM. HIGH band may be sparse.",
            TARGET_RECALL_HIGH,
        )

    # Ensure t_high > t_medium — if derived equal, bump t_high slightly
    if t_high <= t_medium:
        t_high = min(1.0, t_medium + 0.05)
        log.warning(
            "T_HIGH (%.4f) <= T_MEDIUM (%.4f) after derivation — "
            "bumped T_HIGH to %.4f.",
            t_high - 0.05, t_medium, t_high,
        )

    return {
        "t_medium": t_medium,
        "t_medium_precision": medium_prec,
        "t_medium_recall": medium_rec,
        "t_medium_f1": medium_f1,
        "t_medium_reasoning": (
            "Threshold that maximises F1 on calibrated validation probabilities. "
            "Defines the lower boundary of the MEDIUM band (balanced recall/precision, "
            "suitable for 'monitor closely' triage actions)."
        ),
        "t_high": t_high,
        "t_high_precision": high_prec,
        "t_high_recall": high_rec,
        "t_high_f1": high_f1,
        "t_high_reasoning": (
            f"Highest calibrated threshold where validation recall >= {TARGET_RECALL_HIGH:.0%}. "
            "Regulatory safety context: a missed HIGH-risk device (false negative) has "
            "significantly higher cost than a spurious HIGH flag (false positive). "
            "Setting a minimum recall floor ensures the HIGH band is not effectively empty. "
            "Precision at this threshold is maximised subject to the recall constraint."
        ),
    }


def update_config_thresholds(t_medium: float, t_high: float, config_path: Path) -> None:
    """Overwrite RISK_THRESHOLD_HIGH and RISK_THRESHOLD_MEDIUM in src/config.py."""
    text = config_path.read_text(encoding="utf-8")

    text = re.sub(
        r"^(RISK_THRESHOLD_HIGH\s*:\s*float\s*=\s*)[\d.]+",
        lambda m: m.group(1) + repr(round(t_high, 6)),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^(RISK_THRESHOLD_MEDIUM\s*:\s*float\s*=\s*)[\d.]+",
        lambda m: m.group(1) + repr(round(t_medium, 6)),
        text,
        flags=re.MULTILINE,
    )
    config_path.write_text(text, encoding="utf-8")
    log.info(
        "Updated src/config.py  RISK_THRESHOLD_MEDIUM=%.6f  RISK_THRESHOLD_HIGH=%.6f",
        t_medium,
        t_high,
    )


def main() -> None:
    t0 = time.time()
    log.info("=" * 70)
    log.info("STAGE 6 -- Probability Calibration")
    log.info("=" * 70)

    # ── Load data splits ───────────────────────────────────────────────────
    log.info("Loading feature splits...")
    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("validation")
    log.info("  Train: %d  Val: %d", len(y_train), len(y_val))

    # ── Load base model ────────────────────────────────────────────────────
    scorer = RiskScorer(PRODUCTION_MODEL_DIR)
    pos_rate_val = y_val.mean()
    random_brier = float(pos_rate_val * (1 - pos_rate_val))

    # ── BEFORE calibration ─────────────────────────────────────────────────
    log.info("\nComputing pre-calibration metrics on validation...")
    raw_proba_val = scorer._raw_proba(X_val)

    before_brier = float(brier_score_loss(y_val, raw_proba_val))
    before_brier_skill = float(1 - before_brier / random_brier) if random_brier > 0 else float("nan")
    before_ece = compute_ece(y_val, raw_proba_val)
    before_mean_pred, before_frac_pos = compute_reliability_curve(y_val, raw_proba_val)
    before_pr_auc = float(average_precision_score(y_val, raw_proba_val))

    log.info("  Brier score (before): %.6f  (random baseline: %.6f)", before_brier, random_brier)
    log.info("  Brier skill score:    %.4f", before_brier_skill)
    log.info("  ECE (before):         %.6f", before_ece)
    log.info("  PR-AUC (before):      %.4f", before_pr_auc)

    # ── Fit calibration ────────────────────────────────────────────────────
    log.info("\nFitting isotonic calibration on training split only...")
    scorer.fit_calibration(X_train, y_train, save=True)

    # ── AFTER calibration ──────────────────────────────────────────────────
    log.info("\nComputing post-calibration metrics on validation...")
    cal_proba_val = scorer._calibrated_proba(X_val)

    after_brier = float(brier_score_loss(y_val, cal_proba_val))
    after_brier_skill = float(1 - after_brier / random_brier) if random_brier > 0 else float("nan")
    after_ece = compute_ece(y_val, cal_proba_val)
    after_mean_pred, after_frac_pos = compute_reliability_curve(y_val, cal_proba_val)
    after_pr_auc = float(average_precision_score(y_val, cal_proba_val))

    log.info("  Brier score (after):  %.6f", after_brier)
    log.info("  Brier skill score:    %.4f", after_brier_skill)
    log.info("  ECE (after):          %.6f", after_ece)
    log.info("  PR-AUC (after):       %.4f", after_pr_auc)

    improvement = before_brier - after_brier
    log.info("\n  Brier improvement:    %+.6f (%s)",
             improvement,
             "IMPROVED" if improvement > 0 else "DEGRADED or UNCHANGED")

    # ── Derive thresholds ──────────────────────────────────────────────────
    log.info("\nDeriving risk band thresholds from calibrated validation P/R curve...")
    thresholds = derive_thresholds(y_val, cal_proba_val)
    t_medium = thresholds["t_medium"]
    t_high = thresholds["t_high"]

    log.info(
        "  T_MEDIUM=%.6f  (P=%.4f  R=%.4f  F1=%.4f)",
        t_medium, thresholds["t_medium_precision"],
        thresholds["t_medium_recall"], thresholds["t_medium_f1"],
    )
    log.info(
        "  T_HIGH  =%.6f  (P=%.4f  R=%.4f  F1=%.4f)",
        t_high, thresholds["t_high_precision"],
        thresholds["t_high_recall"], thresholds["t_high_f1"],
    )

    # Risk band distribution on validation (preview)
    from src.risk.scorer import score_to_band
    bands = np.array([score_to_band(float(p), t_medium, t_high) for p in cal_proba_val])
    log.info(
        "\n  Val band distribution: LOW=%d  MEDIUM=%d  HIGH=%d",
        (bands == "LOW").sum(), (bands == "MEDIUM").sum(), (bands == "HIGH").sum(),
    )

    # ── Save calibration report ────────────────────────────────────────────
    report = {
        "sklearn_version": __import__("sklearn").__version__,
        "calibration_method": CALIBRATION_METHOD,
        "risk_score_version": RISK_SCORE_VERSION,
        "base_model": scorer._card.get("model_name", "random_forest"),
        "calibration_split": "train",
        "evaluation_split": "validation_2015",
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "positive_rate_val": float(pos_rate_val),
        "random_brier_baseline": random_brier,
        "before_calibration": {
            "brier_score": before_brier,
            "brier_skill_score": before_brier_skill,
            "ece": before_ece,
            "pr_auc": before_pr_auc,
            "reliability_mean_pred": before_mean_pred,
            "reliability_frac_pos": before_frac_pos,
        },
        "after_calibration": {
            "brier_score": after_brier,
            "brier_skill_score": after_brier_skill,
            "ece": after_ece,
            "pr_auc": after_pr_auc,
            "reliability_mean_pred": after_mean_pred,
            "reliability_frac_pos": after_frac_pos,
        },
        "brier_improvement": improvement,
        "risk_band_thresholds": thresholds,
        "validation_band_distribution": {
            "LOW": int((bands == "LOW").sum()),
            "MEDIUM": int((bands == "MEDIUM").sum()),
            "HIGH": int((bands == "HIGH").sum()),
        },
    }

    CALIBRATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("\nSaved calibration report -> %s", CALIBRATION_REPORT_PATH)

    # ── Write thresholds back to src/config.py ─────────────────────────────
    config_path = Path(__file__).resolve().parent.parent.parent / "src" / "config.py"
    update_config_thresholds(t_medium, t_high, config_path)

    elapsed = time.time() - t0
    log.info("\n" + "=" * 70)
    log.info("Calibration complete in %.1f seconds.", elapsed)
    log.info("  calibrated_model.pkl  -> %s", CALIBRATED_MODEL_PATH)
    log.info("  calibration_report.json -> %s", CALIBRATION_REPORT_PATH)
    log.info("  src/config.py updated with RISK_THRESHOLD_MEDIUM=%.6f  RISK_THRESHOLD_HIGH=%.6f",
             t_medium, t_high)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
