"""
Stage 5 — ML Risk Engine: Model Training
=========================================
Trains 4 model candidates on the Stage-4 feature matrix and evaluates each
on the validation split.  All model artifacts are saved under
models/experiments/<name>_<timestamp>/.

Run:
    python -m src.models.train

Candidates (in order):
    1. MajorityClassBaseline   — anchors "better-than-nothing" performance
    2. LogisticRegression       — interpretable linear baseline
    3. RandomForest             — ensemble, handles mixed feature types
    4. XGBoost                  — gradient boosting, often strongest

Class-imbalance strategy:
    - Logistic Regression / Random Forest: class_weight='balanced'
    - XGBoost: scale_pos_weight = neg / pos (equivalent reweighting)
    No SMOTE/oversampling used; class weighting is sufficient and avoids
    introducing synthetic samples into a temporally-ordered dataset.

Outputs per experiment:
    model.pkl, preprocessor.pkl, val_metrics.json, val_predictions.npz
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

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
METADATA_COLS = {"id", "device_id", "manufacturer_id", "event_date", "event_date_available"}
TARGET_COL = "is_class_i"

# Default decision threshold for binary metrics in experiments
DEFAULT_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a feature split, drop metadata, return X, y, feature_cols."""
    path = FEATURES_DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS and c != TARGET_COL]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(int)
    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Compute classification metrics dict at the given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": threshold,
        "n_positive_pred": int(y_pred.sum()),
        "n_actual_positive": int(y_true.sum()),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Experiment saving
# ─────────────────────────────────────────────────────────────────────────────

def save_experiment(
    name: str,
    model,
    preprocessor,
    y_val: np.ndarray,
    y_val_proba: np.ndarray,
    y_train: np.ndarray,
    y_train_proba: np.ndarray,
    feature_cols: list[str],
    hyperparams: dict,
    run_timestamp: str,
) -> Path:
    """Persist a model experiment under models/experiments/<name>_<ts>/."""
    exp_dir = EXPERIMENTS_DIR / f"{name}_{run_timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save model and preprocessor
    joblib.dump(model, exp_dir / "model.pkl")
    if preprocessor is not None:
        joblib.dump(preprocessor, exp_dir / "preprocessor.pkl")

    # Metrics on validation
    val_metrics = compute_metrics(y_val, y_val_proba)
    train_metrics = compute_metrics(y_train, y_train_proba)

    # Save predictions for later analysis
    np.savez_compressed(
        exp_dir / "val_predictions.npz",
        y_true=y_val,
        y_proba=y_val_proba,
    )
    np.savez_compressed(
        exp_dir / "train_predictions.npz",
        y_true=y_train,
        y_proba=y_train_proba,
    )

    # Metadata
    meta = {
        "model_name": name,
        "run_timestamp": run_timestamp,
        "random_seed": RANDOM_SEED,
        "hyperparameters": hyperparams,
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "target": TARGET_COL,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "imbalance_handling": hyperparams.get("imbalance_strategy", "none"),
    }
    (exp_dir / "val_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info(
        "  Saved experiment → %s  |  val PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f",
        exp_dir.name,
        val_metrics["pr_auc"],
        val_metrics["roc_auc"],
        val_metrics["f1"],
    )
    return exp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(X_train, y_train, X_val, y_val, feature_cols, ts):
    """Majority-class baseline — always predicts prior probability."""
    log.info("Training 1/4: MajorityClassBaseline …")
    clf = DummyClassifier(strategy="prior", random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_val_proba = clf.predict_proba(X_val)[:, 1]

    return save_experiment(
        name="baseline_majority",
        model=clf,
        preprocessor=None,
        y_val=y_val,
        y_val_proba=y_val_proba,
        y_train=y_train,
        y_train_proba=y_train_proba,
        feature_cols=feature_cols,
        hyperparams={"strategy": "prior", "imbalance_strategy": "prior_rate"},
        run_timestamp=ts,
    )


def train_logistic_regression(X_train, y_train, X_val, y_val, feature_cols, ts):
    """Logistic Regression with StandardScaler, balanced class weights."""
    log.info("Training 2/4: Logistic Regression …")

    # Scale numeric features (LR is sensitive to scale)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    clf = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train_sc, y_train)

    y_train_proba = clf.predict_proba(X_train_sc)[:, 1]
    y_val_proba = clf.predict_proba(X_val_sc)[:, 1]

    return save_experiment(
        name="logistic_regression",
        model=clf,
        preprocessor=scaler,
        y_val=y_val,
        y_val_proba=y_val_proba,
        y_train=y_train,
        y_train_proba=y_train_proba,
        feature_cols=feature_cols,
        hyperparams={
            "class_weight": "balanced",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "imbalance_strategy": "class_weight_balanced",
        },
        run_timestamp=ts,
    )


def train_random_forest(X_train, y_train, X_val, y_val, feature_cols, ts):
    """Random Forest with balanced class weights — no scaling needed."""
    log.info("Training 3/4: Random Forest …")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)

    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_val_proba = clf.predict_proba(X_val)[:, 1]

    return save_experiment(
        name="random_forest",
        model=clf,
        preprocessor=None,
        y_val=y_val,
        y_val_proba=y_val_proba,
        y_train=y_train,
        y_train_proba=y_train_proba,
        feature_cols=feature_cols,
        hyperparams={
            "n_estimators": 300,
            "max_depth": "None",
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "imbalance_strategy": "class_weight_balanced",
        },
        run_timestamp=ts,
    )


def train_xgboost(X_train, y_train, X_val, y_val, feature_cols, ts, neg_pos_ratio: float):
    """XGBoost with scale_pos_weight for class imbalance."""
    log.info("Training 4/4: XGBoost (scale_pos_weight=%.2f) …", neg_pos_ratio)

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=30,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_val_proba = clf.predict_proba(X_val)[:, 1]

    return save_experiment(
        name="xgboost",
        model=clf,
        preprocessor=None,
        y_val=y_val,
        y_val_proba=y_val_proba,
        y_train=y_train,
        y_train_proba=y_train_proba,
        feature_cols=feature_cols,
        hyperparams={
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": round(neg_pos_ratio, 4),
            "early_stopping_rounds": 30,
            "imbalance_strategy": "scale_pos_weight",
        },
        run_timestamp=ts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("STAGE 5 — Model Training")
    log.info("=" * 70)

    start_time = time.time()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    log.info("Loading feature splits …")
    X_train, y_train, feature_cols = load_split("train")
    X_val, y_val, _ = load_split("validation")
    log.info("  Train: %d × %d  (pos=%d, neg=%d, pos_rate=%.2f%%)",
             *X_train.shape, y_train.sum(), (y_train == 0).sum(),
             y_train.mean() * 100)
    log.info("  Val:   %d × %d  (pos=%d, neg=%d, pos_rate=%.2f%%)",
             *X_val.shape, y_val.sum(), (y_val == 0).sum(),
             y_val.mean() * 100)

    # Imbalance ratio for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = y_train.sum()
    neg_pos_ratio = neg_count / pos_count
    log.info("  Class imbalance ratio (neg/pos): %.2f", neg_pos_ratio)

    # Train all candidates
    exp_dirs = {}
    exp_dirs["baseline"] = train_baseline(X_train, y_train, X_val, y_val, feature_cols, run_timestamp)
    exp_dirs["logistic_regression"] = train_logistic_regression(X_train, y_train, X_val, y_val, feature_cols, run_timestamp)
    exp_dirs["random_forest"] = train_random_forest(X_train, y_train, X_val, y_val, feature_cols, run_timestamp)
    exp_dirs["xgboost"] = train_xgboost(X_train, y_train, X_val, y_val, feature_cols, run_timestamp, neg_pos_ratio)

    # Summary table
    log.info("=" * 70)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 70)
    log.info("  %-25s  %8s  %8s  %8s  %8s  %8s",
             "Model", "PR-AUC", "ROC-AUC", "Recall", "Prec", "F1")
    log.info("  " + "-" * 75)

    for name, exp_dir in exp_dirs.items():
        meta = json.loads((exp_dir / "val_metrics.json").read_text())
        vm = meta["val_metrics"]
        log.info("  %-25s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f",
                 name,
                 vm["pr_auc"], vm["roc_auc"],
                 vm["recall"], vm["precision"], vm["f1"])

    elapsed = time.time() - start_time
    log.info("=" * 70)
    log.info("Training complete in %.1f seconds. Run evaluate.py next.", elapsed)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
