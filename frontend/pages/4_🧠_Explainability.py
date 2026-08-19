"""
frontend/pages/4_🧠_Explainability.py
=======================================
Explainability page.

1. Global Feature Importance — GET /feature-importance
2. Per-Device Local SHAP Explanation — GET /explanation/{id}
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils.api_client import BACKEND_URL, get_explanation, get_feature_importance, get_health
from utils.charts import global_importance_bar, shap_waterfall
from utils.styles import DISCLAIMER, inject, page_header, sidebar_base

st.set_page_config(
    page_title="Explainability | Medical Device Risk",
    page_icon="🧠",
    layout="wide",
)

st.markdown(inject(), unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_base(active_page="Explainability", show_model=True)

    st.markdown(
        "<span style='font-size:0.78em;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:var(--text-color);opacity:0.55;'>Chart Controls</span>",
        unsafe_allow_html=True,
    )
    top_n = st.slider("Features to show (global)", min_value=5, max_value=62, value=20, step=5)

# ── Page header ───────────────────────────────────────────────────────────────
page_header(
    icon="🧠",
    title="Explainability",
    subtitle=(
        "Global importance shows which features most influenced the model overall. "
        "Local SHAP shows how each feature pushed a specific device's prediction up or down."
    ),
)

st.markdown(
    "<div class='info-note'>"
    "🔍 <b>Why explainability matters:</b> These features contributed most strongly to the "
    "model's assessment of event severity (Class I likelihood). Understanding which signals "
    "drive a high-severity classification helps investigators evaluate whether the same conditions "
    "exist in currently active devices — supporting early identification and preventive maintenance decisions. "
    "Red (positive SHAP) features push the prediction toward Class I (high severity). "
    "Blue (negative SHAP) features push it toward lower severity. "
    "<br><em>Feature contributions show model reasoning — they do not imply direct medical causation.</em>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Section 1: Global feature importance ─────────────────────────────────────
st.markdown("<div class='section-title'>📊 Global Feature Importance</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-intro'>Source: <code>GET /feature-importance</code> — pre-computed from Stage 5 Random Forest training.</div>",
    unsafe_allow_html=True,
)

with st.spinner("Loading feature importance…"):
    fi_resp = get_feature_importance()

if fi_resp is None:
    st.error(f"Could not reach the backend at `{BACKEND_URL}`.")
elif fi_resp.get("count", 0) == 0:
    st.warning("No feature importance data available. Check that `feature_importance.json` exists.")
else:
    features    = fi_resp.get("features", [])
    model_ver   = fi_resp.get("model_version", "unknown")
    total_feats = fi_resp.get("count", len(features))

    c_left, c_right = st.columns([3, 1])
    with c_left:
        st.markdown(
            f"<div style='font-size:0.82em;color:var(--text-color);opacity:0.55;padding-bottom:6px;'>"
            f"Model: <code>{model_ver}</code> &nbsp;·&nbsp; {total_feats} features total "
            f"&nbsp;·&nbsp; showing top {min(top_n, total_feats)}"
            f"</div>",
            unsafe_allow_html=True,
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
st.markdown("<div class='section-title'>🔬 Per-Device Local SHAP Explanation</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-intro'>"
    "Source: <code>GET /explanation/{id}</code> — real SHAP values from <code>shap.TreeExplainer</code>. "
    "🔴 Red bars increase predicted risk. 🔵 Blue bars decrease it. "
    "Cached to <code>artifacts/explanations/</code> — first call per device may take a few seconds."
    "</div>",
    unsafe_allow_html=True,
)

device_id_input = st.text_input(
    "Device ID:",
    placeholder="e.g. 80508",
    key="expl_device_id",
)

if not device_id_input:
    st.markdown(
        "<div class='info-note'>Enter a device ID above to load its local SHAP explanation.</div>",
        unsafe_allow_html=True,
    )
else:
    device_id = device_id_input.strip()
    with st.spinner(f"Loading SHAP explanation for device {device_id}…"):
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
