"""
Stage 5 — ML Risk Engine: Model Evaluation
===========================================
Loads all experiment artifacts, selects the best model by PR-AUC on the
validation split, tunes the decision threshold on validation (maximises F1),
evaluates on the held-out test set (2016-2017) and 2018 holdout, extracts
native feature importances, and promotes the winner to models/production/.

Run:
    python -m src.models.evaluate

Prerequisites:
    python -m src.models.train   must have run first.

Outputs:
    models/production/model.pkl
    models/production/preprocessor.pkl       (if model uses one)
    models/production/test_metrics.json
    models/production/feature_importance.json
    models/production/model_card.json
"""

import json
import logging
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import FEATURES_DATA_DIR, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = ROOT / "models" / "experiments"
PRODUCTION_DIR = ROOT / "models" / "production"
METADATA_COLS = {"id", "device_id", "manufacturer_id", "event_date", "event_date_available"}
TARGET_COL = "is_class_i"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_split(name: str) -> tuple:
    """Load a feature split from data/features/, return X, y, feature_cols."""
    path = FEATURES_DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS and c != TARGET_COL]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(int)
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------

def load_experiments() -> list:
    """
    Scan models/experiments/ for val_metrics.json files.
    Returns a list of dicts sorted descending by val PR-AUC.
    """
    experiments = []
    for metrics_path in sorted(EXPERIMENTS_DIR.rglob("val_metrics.json")):
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        experiments.append({
            "name": meta["model_name"],
            "exp_dir": metrics_path.parent,
            "val_pr_auc": meta["val_metrics"]["pr_auc"],
            "val_roc_auc": meta["val_metrics"]["roc_auc"],
            "val_recall": meta["val_metrics"]["recall"],
            "val_f1": meta["val_metrics"]["f1"],
            "meta": meta,
        })

    if not experiments:
        raise FileNotFoundError(
            f"No experiments found under {EXPERIMENTS_DIR}. "
            "Run: python -m src.models.train"
        )

    experiments.sort(key=lambda e: e["val_pr_auc"], reverse=True)
    return experiments


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold_on_validation(y_val: np.ndarray, y_val_proba: np.ndarray) -> float:
    """
    Find the decision threshold that maximises F1 on the validation set.
    Searches over thresholds returned by precision_recall_curve.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    # precision_recall_curve returns N+1 precision/recall points; thresholds has N values
    f1s = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0.0,
    )
    best_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])

    log.info(
        "  Threshold tuning (val): best_threshold=%.4f  F1=%.4f  "
        "Precision=%.4f  Recall=%.4f",
        best_threshold,
        best_f1,
        float(precisions[best_idx]),
        float(recalls[best_idx]),
    )
    return best_threshold


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    split_name: str,
) -> dict:
    """Full metrics dict at a given decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "n_actual_positive": int(y_true.sum()),
        "n_actual_negative": int((y_true == 0).sum()),
        "threshold": round(threshold, 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_predicted_positive": int(y_pred.sum()),
        "positive_rate_actual_pct": round(float(y_true.mean() * 100), 4),
        "positive_rate_predicted_pct": round(float(y_pred.mean() * 100), 4),
    }

    log.info(
        "  [%s] PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  "
        "Prec=%.4f  Rec=%.4f  (threshold=%.4f)",
        split_name,
        metrics["pr_auc"], metrics["roc_auc"], metrics["f1"],
        metrics["precision"], metrics["recall"], metrics["threshold"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def extract_feature_importance(model, feature_cols: list) -> list:
    """
    Extract native feature importances for tree-based models.
    Returns a list of {feature, importance} dicts sorted descending,
    or None if the model does not expose feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        log.info("  Model has no feature_importances_ attribute — skipping.")
        return None

    importances = model.feature_importances_
    fi = [
        {"feature": name, "importance": round(float(imp), 8)}
        for name, imp in zip(feature_cols, importances)
    ]
    fi.sort(key=lambda x: x["importance"], reverse=True)
    return fi


# ---------------------------------------------------------------------------
# Promotion to production
# ---------------------------------------------------------------------------

def promote_to_production(
    best: dict,
    threshold: float,
    test_metrics: dict,
    holdout_metrics: dict,
    feature_importance: list,
    feature_cols: list,
):
    """Copy model artifacts from the best experiment into models/production/."""
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    exp_dir = best["exp_dir"]

    shutil.copy2(exp_dir / "model.pkl", PRODUCTION_DIR / "model.pkl")
    log.info("  Copied model.pkl -> models/production/")

    pre_src = exp_dir / "preprocessor.pkl"
    if pre_src.exists():
        shutil.copy2(pre_src, PRODUCTION_DIR / "preprocessor.pkl")
        log.info("  Copied preprocessor.pkl -> models/production/")

    feat_meta_src = FEATURES_DATA_DIR / "feature_metadata.json"
    if feat_meta_src.exists():
        shutil.copy2(feat_meta_src, PRODUCTION_DIR / "feature_metadata.json")

    test_metrics_payload = {
        "validation": best["meta"]["val_metrics"],
        "test": test_metrics,
        "holdout_2018": holdout_metrics,
        "decision_threshold": threshold,
    }
    (PRODUCTION_DIR / "test_metrics.json").write_text(
        json.dumps(test_metrics_payload, indent=2), encoding="utf-8"
    )
    log.info("  Wrote test_metrics.json")

    if feature_importance is not None:
        (PRODUCTION_DIR / "feature_importance.json").write_text(
            json.dumps(feature_importance, indent=2), encoding="utf-8"
        )
        log.info("  Wrote feature_importance.json (%d features)", len(feature_importance))

    model_card = {
        "model_name": best["name"],
        "experiment_dir": str(exp_dir.name),
        "random_seed": RANDOM_SEED,
        "target": TARGET_COL,
        "target_positive": "Class I (most severe FDA recall class)",
        "target_negative": "Class II or Class III",
        "selection_metric": "PR-AUC on validation set (2015)",
        "decision_threshold": threshold,
        "threshold_strategy": "Maximise F1 on validation set",
        "primary_metric": "PR-AUC",
        "rationale_for_pr_auc": (
            "Class I events are 7.6% of labeled data (1:12.2 imbalance). "
            "PR-AUC is more informative than ROC-AUC under class imbalance."
        ),
        "hyperparameters": best["meta"]["hyperparameters"],
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "temporal_split": {
            "train": "<= 2014-12-31",
            "validation": "2015-01-01 to 2015-12-31",
            "test": "2016-01-01 to 2017-12-31",
            "holdout": "2018-01-01 to 2018-07-09",
        },
        "known_limitations": [
            "Labels exist only for USA, Canada, Australia, El Salvador (4 countries).",
            "Model classifies severity of already-initiated events, not future failure.",
            "Positive rate shifts from 8.34% (train) to 5.52% (test); temporal drift expected.",
            "Device attributes (classification, risk_class, implanted) missing for ~70% of rows.",
            "reason text excluded from primary model due to borderline leakage risk.",
        ],
        "metrics": {
            "validation": best["meta"]["val_metrics"],
            "test": test_metrics,
            "holdout_2018": holdout_metrics,
        },
    }
    (PRODUCTION_DIR / "model_card.json").write_text(
        json.dumps(model_card, indent=2), encoding="utf-8"
    )
    log.info("  Wrote model_card.json")


# ---------------------------------------------------------------------------
# Prediction helper (handles preprocessor-based models)
# ---------------------------------------------------------------------------

def predict_proba(model, preprocessor, X: np.ndarray) -> np.ndarray:
    if preprocessor is not None:
        X = preprocessor.transform(X)
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 70)
    log.info("STAGE 5 -- Model Evaluation")
    log.info("=" * 70)
    start_time = time.time()

    # 1. Discover experiments
    log.info("Scanning experiments ...")
    experiments = load_experiments()

    log.info("\n  Validation leaderboard (sorted by PR-AUC):")
    log.info("  %-30s  %8s  %8s  %8s", "Model", "PR-AUC", "ROC-AUC", "F1")
    log.info("  " + "-" * 60)
    for e in experiments:
        log.info(
            "  %-30s  %8.4f  %8.4f  %8.4f",
            e["name"], e["val_pr_auc"], e["val_roc_auc"], e["val_f1"],
        )

    best = experiments[0]
    log.info("\n  Best model: %s  (val PR-AUC=%.4f)", best["name"], best["val_pr_auc"])

    # 2. Load best model
    log.info("\nLoading best model ...")
    model = joblib.load(best["exp_dir"] / "model.pkl")
    preprocessor_path = best["exp_dir"] / "preprocessor.pkl"
    preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None

    # 3. Tune threshold on validation
    log.info("\nTuning decision threshold on validation ...")
    val_npz = np.load(best["exp_dir"] / "val_predictions.npz")
    y_val, y_val_proba = val_npz["y_true"], val_npz["y_proba"]
    best_threshold = tune_threshold_on_validation(y_val, y_val_proba)

    # 4. Evaluate on test set
    log.info("\nEvaluating on test set (2016-2017) ...")
    X_test, y_test, feature_cols = load_split("test")
    y_test_proba = predict_proba(model, preprocessor, X_test)
    test_metrics = compute_metrics_at_threshold(
        y_test, y_test_proba, best_threshold, "test_2016_2017"
    )

    # 5. Evaluate on holdout
    log.info("\nEvaluating on holdout (2018) ...")
    X_holdout, y_holdout, _ = load_split("holdout_2018")
    y_holdout_proba = predict_proba(model, preprocessor, X_holdout)
    holdout_metrics = compute_metrics_at_threshold(
        y_holdout, y_holdout_proba, best_threshold, "holdout_2018"
    )

    # 6. Feature importance
    log.info("\nExtracting feature importances ...")
    fi = extract_feature_importance(model, feature_cols)
    if fi:
        log.info("  Top-10 features:")
        for item in fi[:10]:
            log.info("    %-50s  %.6f", item["feature"], item["importance"])

    # 7. Promote to production
    log.info("\nPromoting best model to models/production/ ...")
    promote_to_production(
        best=best,
        threshold=best_threshold,
        test_metrics=test_metrics,
        holdout_metrics=holdout_metrics,
        feature_importance=fi,
        feature_cols=feature_cols,
    )

    # 8. Final summary
    elapsed = time.time() - start_time
    log.info("\n" + "=" * 70)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 70)
    log.info("  Best model        : %s", best["name"])
    log.info("  Decision threshold: %.4f (tuned on val, maximises F1)", best_threshold)
    log.info("")
    log.info("  %-22s  %8s  %8s  %8s  %8s  %8s",
             "Split", "PR-AUC", "ROC-AUC", "F1", "Prec", "Rec")
    log.info("  " + "-" * 72)
    val_m = best["meta"]["val_metrics"]
    log.info("  %-22s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f",
             "validation_2015",
             val_m["pr_auc"], val_m["roc_auc"], val_m["f1"],
             val_m["precision"], val_m["recall"])
    for m in [test_metrics, holdout_metrics]:
        if m is None:
            continue
        log.info("  %-22s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f",
                 m["split"],
                 m["pr_auc"], m["roc_auc"], m["f1"],
                 m["precision"], m["recall"])
    log.info("=" * 70)
    log.info("Evaluation complete in %.1f seconds.", elapsed)
    log.info("Production artifacts -> models/production/")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
