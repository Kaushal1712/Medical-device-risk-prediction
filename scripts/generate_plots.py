#!/usr/bin/env python3
"""
Generate required PR/ROC/calibration/SHAP plots from existing model artifacts.

This script does NOT retrain models or rerun any pipeline.
It reads:
  - models/experiments/*/val_predictions.npz   (y_true, y_proba for 4 candidates)
  - models/production/model.pkl                 (Random Forest for SHAP)
  - data/features/validation.parquet            (feature data for SHAP)
  - models/production/model_card.json           (feature column list)

Outputs (PNG) to artifacts/plots/:
  - pr_curve.png
  - roc_curve.png
  - calibration_plot.png
  - shap_summary.png
"""

import os
import glob
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_PLOTS = "artifacts/plots"
EXPERIMENTS_DIR = "models/experiments"
MODEL_PATH = "models/production/model.pkl"
MODEL_CARD_PATH = "models/production/model_card.json"
VAL_FEATURES_PATH = "data/features/validation.parquet"

os.makedirs(ARTIFACTS_PLOTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Load all candidate predictions
# ---------------------------------------------------------------------------
CANDIDATE_DISPLAY = {
    "baseline_majority": "Majority Baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest ✓",
    "xgboost": "XGBoost",
}
CANDIDATE_COLORS = {
    "baseline_majority": "#94a3b8",
    "logistic_regression": "#60a5fa",
    "random_forest": "#f97316",
    "xgboost": "#a78bfa",
}
CANDIDATE_LINESTYLES = {
    "baseline_majority": "--",
    "logistic_regression": "-.",
    "random_forest": "-",
    "xgboost": ":",
}

candidates = {}
for npz_path in sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "*/val_predictions.npz"))):
    exp_dir = os.path.dirname(npz_path)
    exp_name = os.path.basename(exp_dir)
    # strip timestamp suffix like _20260816_145309
    parts = exp_name.rsplit("_", 2)
    key = "_".join(parts[:-2]) if len(parts) >= 3 and parts[-2].isdigit() else exp_name
    arr = np.load(npz_path)
    candidates[key] = {
        "y_true": arr["y_true"].astype(int),
        "y_proba": arr["y_proba"].astype(float),
    }

print(f"Loaded {len(candidates)} candidate prediction arrays: {list(candidates.keys())}")


# ---------------------------------------------------------------------------
# 1. PR Curve
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

for key, data in candidates.items():
    y_true = data["y_true"]
    y_proba = data["y_proba"]
    ap = average_precision_score(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    label = f"{CANDIDATE_DISPLAY.get(key, key)}  (AP={ap:.3f})"
    ax.plot(
        rec, prec,
        label=label,
        color=CANDIDATE_COLORS.get(key, None),
        linestyle=CANDIDATE_LINESTYLES.get(key, "-"),
        linewidth=2.0,
    )

# Baseline (random predictor) line at positive rate
pos_rate = candidates["random_forest"]["y_true"].mean()
ax.axhline(y=pos_rate, color="#64748b", linestyle="--", linewidth=1.0,
           label=f"Random predictor  (AP={pos_rate:.3f})")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves — Validation Set (2015)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
fig.tight_layout()
pr_path = os.path.join(ARTIFACTS_PLOTS, "pr_curve.png")
fig.savefig(pr_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {pr_path}")


# ---------------------------------------------------------------------------
# 2. ROC Curve
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

for key, data in candidates.items():
    y_true = data["y_true"]
    y_proba = data["y_proba"]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc_val = auc(fpr, tpr)
    label = f"{CANDIDATE_DISPLAY.get(key, key)}  (AUC={roc_auc_val:.3f})"
    ax.plot(
        fpr, tpr,
        label=label,
        color=CANDIDATE_COLORS.get(key, None),
        linestyle=CANDIDATE_LINESTYLES.get(key, "-"),
        linewidth=2.0,
    )

ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1.0, label="Random classifier (AUC=0.500)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Validation Set (2015)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
fig.tight_layout()
roc_path = os.path.join(ARTIFACTS_PLOTS, "roc_curve.png")
fig.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {roc_path}")


# ---------------------------------------------------------------------------
# 3. Calibration Plot (Random Forest only — winner)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Calibration curve (reliability diagram)
ax = axes[0]
y_true_rf = candidates["random_forest"]["y_true"]
y_proba_rf = candidates["random_forest"]["y_proba"]

# Uncalibrated RF
prob_true_raw, prob_pred_raw = calibration_curve(y_true_rf, y_proba_rf, n_bins=10, strategy="quantile")
ax.plot(prob_pred_raw, prob_true_raw, "o-", color="#f97316", linewidth=2, markersize=6,
        label="Random Forest (uncalibrated)")

# Load calibrated model if available
cal_model_path = "models/production/calibrated_model.pkl"
if os.path.exists(cal_model_path):
    cal_model = joblib.load(cal_model_path)
    # Load validation features
    with open(MODEL_CARD_PATH) as f:
        mc = json.load(f)
    feature_cols = mc["feature_columns"]
    val_df = pd.read_parquet(VAL_FEATURES_PATH)
    X_val = val_df[feature_cols]
    y_proba_cal = cal_model.predict_proba(X_val)[:, 1]
    prob_true_cal, prob_pred_cal = calibration_curve(y_true_rf, y_proba_cal, n_bins=10, strategy="quantile")
    ax.plot(prob_pred_cal, prob_true_cal, "s-", color="#22c55e", linewidth=2, markersize=6,
            label="Random Forest (isotonic calibrated)")
    print("Loaded calibrated model for calibration plot.")
else:
    print("Warning: calibrated_model.pkl not found; plotting uncalibrated RF only.")

ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration")
ax.set_xlabel("Mean predicted probability", fontsize=11)
ax.set_ylabel("Fraction of positives", fontsize=11)
ax.set_title("Calibration Curve (Reliability Diagram)\nValidation Set (2015)", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

# Probability histogram
ax2 = axes[1]
ax2.hist(y_proba_rf, bins=40, alpha=0.6, color="#f97316", label="Uncalibrated", density=True)
if os.path.exists(cal_model_path):
    ax2.hist(y_proba_cal, bins=40, alpha=0.6, color="#22c55e", label="Calibrated", density=True)
ax2.set_xlabel("Predicted probability", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title("Predicted Probability Distribution\nRandom Forest — Validation Set (2015)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3)

fig.suptitle("Calibration Analysis — Random Forest", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
cal_path = os.path.join(ARTIFACTS_PLOTS, "calibration_plot.png")
fig.savefig(cal_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {cal_path}")


# ---------------------------------------------------------------------------
# 4. SHAP Summary Plot
# ---------------------------------------------------------------------------
try:
    import shap

    print("Loading production Random Forest for SHAP analysis...")
    rf_model = joblib.load(MODEL_PATH)

    with open(MODEL_CARD_PATH) as f:
        mc = json.load(f)
    feature_cols = mc["feature_columns"]

    val_df = pd.read_parquet(VAL_FEATURES_PATH)
    X_val = val_df[feature_cols].copy()
    print(f"Validation features loaded: {X_val.shape}")

    # Use a subsample for speed (still representative — 500 rows from 4,273)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_val), size=min(500, len(X_val)), replace=False)
    X_sample = X_val.iloc[sample_idx].reset_index(drop=True)

    print("Computing SHAP values with TreeExplainer (may take ~30s)...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    # shap_values may be list [class0, class1] for binary classifier
    if isinstance(shap_values, list):
        sv = shap_values[1]  # class 1 = is_class_i
    else:
        sv = shap_values

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv,
        X_sample,
        feature_names=feature_cols,
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance — Random Forest\n(Global, validation set sample n=500)",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    shap_path = os.path.join(ARTIFACTS_PLOTS, "shap_summary.png")
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {shap_path}")

except ImportError:
    print("ERROR: shap not installed. Cannot generate SHAP summary plot.")
    raise
except Exception as e:
    print(f"ERROR generating SHAP plot: {e}")
    raise

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== artifacts/plots/ contents ===")
for f in sorted(os.listdir(ARTIFACTS_PLOTS)):
    path = os.path.join(ARTIFACTS_PLOTS, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  {f}  ({size_kb:.1f} KB)")
