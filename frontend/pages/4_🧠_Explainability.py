"""
frontend/pages/4_🧠_Explainability.py
=======================================
Explainability page.

Sections:
  1. Global Feature Importance — GET /feature-importance
     Horizontal bar chart of all features ranked by model importance.

  2. Per-Device Local SHAP Explanation — GET /explanation/{id}
     Same SHAP waterfall chart reused from Device Details.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils.api_client import BACKEND_URL, get_explanation, get_feature_importance, get_health
from utils.charts import global_importance_bar, shap_waterfall

st.set_page_config(
    page_title="Explainability | Medical Device Risk",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.disclaimer { background:#f8fafc; border-left:4px solid #94a3b8;
              padding:10px 16px; border-radius:4px; font-size:0.78em; color:#64748b; }
.unavailable { background:#fff7ed; border-left:4px solid #f59e0b;
               padding:12px 16px; border-radius:4px; color:#92400e; }
</style>
""", unsafe_allow_html=True)

DISCLAIMER = (
    "This system is a decision-support prototype and does not replace qualified "
    "maintenance, biomedical engineering, regulatory, or clinical judgment. "
    "It is not a certified medical device and does not guarantee patient safety outcomes."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Explainability")
    health = get_health()
    if health:
        st.success("🟢 Backend connected")
        st.caption(f"Model: `{health.get('model_version','unknown')}`")
    else:
        st.error(f"🔴 Backend unreachable\n`{BACKEND_URL}`")
    st.divider()
    top_n = st.slider("Features to show (global)", min_value=5, max_value=62, value=20, step=5)
    st.divider()
    st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)

st.title("🧠 Explainability")
st.caption(
    "**Global importance** shows which features most influenced the model overall. "
    "**Local SHAP** shows how each feature pushed a specific device's prediction up or down."
)

# ── Section 1: Global feature importance ─────────────────────────────────────
st.subheader("📊 Global Feature Importance")
st.caption("Source: `GET /feature-importance` — pre-computed from Stage 5 Random Forest training.")

with st.spinner("Loading feature importance..."):
    fi_resp = get_feature_importance()

if fi_resp is None:
    st.error(f"Could not reach the backend at `{BACKEND_URL}`.")
elif fi_resp.get("count", 0) == 0:
    st.warning("No feature importance data available. Check that `feature_importance.json` exists.")
else:
    features    = fi_resp.get("features", [])
    model_ver   = fi_resp.get("model_version", "unknown")
    total_feats = fi_resp.get("count", len(features))

    st.caption(
        f"Model version: `{model_ver}` · {total_feats} features total · "
        f"showing top {min(top_n, total_feats)}"
    )

    fig_global = global_importance_bar(features, top_n=top_n)
    st.plotly_chart(fig_global, use_container_width=True)

    with st.expander(f"View all {total_feats} features as table"):
        import pandas as pd
        df_fi = pd.DataFrame(features)[["rank", "feature", "importance"]]
        df_fi.columns = ["Rank", "Feature", "Importance"]
        st.dataframe(df_fi, use_container_width=True, hide_index=True)

st.divider()

# ── Section 2: Per-device local SHAP ──────────────────────────────────────────
st.subheader("🔬 Per-Device Local SHAP Explanation")
st.caption("Source: `GET /explanation/{id}` — real SHAP values from shap.TreeExplainer.")
st.caption(
    "🔴 Red bars increase predicted risk. 🔵 Blue bars decrease it. "
    "Cached to `artifacts/explanations/` — first call per device may take a few seconds."
)

device_id_input = st.text_input(
    "Device ID:",
    placeholder="e.g. 80508",
    key="expl_device_id",
)

if not device_id_input:
    st.info("Enter a device ID above to load its local SHAP explanation.")
else:
    device_id = device_id_input.strip()
    with st.spinner(f"Loading SHAP explanation for device {device_id}..."):
        expl = get_explanation(device_id)

    if expl is None:
        st.error(f"Could not reach the backend at `{BACKEND_URL}`.")
    elif not expl.get("available"):
        st.markdown(
            f"<div class='unavailable'>"
            f"<strong>Explanation Unavailable</strong><br>"
            f"{expl.get('unavailable_reason', 'No explanation available for this device.')}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        model_ver = expl.get("model_version", "unknown")
        base_val  = expl.get("base_value", 0.0)
        pred_val  = expl.get("predicted_value", 0.0)

        meta_l, meta_r = st.columns(2)
        meta_l.metric("Base Value (mean prediction)", f"{base_val:.4f}")
        meta_r.metric("Predicted Value (this device)", f"{pred_val:.4f}",
                      delta=f"{pred_val - base_val:+.4f}",
                      delta_color="inverse")

        fig_local = shap_waterfall(
            top_positive=expl.get("top_positive", []),
            top_negative=expl.get("top_negative", []),
            base_value=base_val,
            predicted_value=pred_val,
        )
        st.plotly_chart(fig_local, use_container_width=True)
        st.caption(f"Model version: `{model_ver}`")

        with st.expander("Raw SHAP data"):
            st.json(expl)

st.divider()
st.markdown(f"<div class='disclaimer'>⚕️ {DISCLAIMER}</div>", unsafe_allow_html=True)
