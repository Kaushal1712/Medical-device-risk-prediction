"""
frontend/app.py
================
Stage 8 — Medical Device Risk Dashboard (Streamlit entry point).

Run:
    streamlit run frontend/app.py

The BACKEND_URL environment variable controls which FastAPI server is used.
Defaults to http://localhost:8000.
"""

import sys
from pathlib import Path

# Make frontend/utils importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from utils.api_client import BACKEND_URL, get_health

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Medical Device Risk Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
}

/* Risk badges */
.badge-HIGH   { background:#fef2f2; color:#dc2626; padding:3px 12px;
                border-radius:20px; font-weight:600; font-size:0.82em; }
.badge-MEDIUM { background:#fffbeb; color:#d97706; padding:3px 12px;
                border-radius:20px; font-weight:600; font-size:0.82em; }
.badge-LOW    { background:#f0fdf4; color:#16a34a; padding:3px 12px;
                border-radius:20px; font-weight:600; font-size:0.82em; }
.badge-NA     { background:#f1f5f9; color:#64748b; padding:3px 12px;
                border-radius:20px; font-weight:600; font-size:0.82em; }

/* Priority badges */
.prio-Critical { background:#fef2f2; color:#b91c1c; padding:3px 12px;
                 border-radius:20px; font-weight:700; font-size:0.85em; }
.prio-High     { background:#fff7ed; color:#c2410c; padding:3px 12px;
                 border-radius:20px; font-weight:700; font-size:0.85em; }
.prio-Medium   { background:#fffbeb; color:#b45309; padding:3px 12px;
                 border-radius:20px; font-weight:700; font-size:0.85em; }
.prio-Low      { background:#f0fdf4; color:#15803d; padding:3px 12px;
                 border-radius:20px; font-weight:700; font-size:0.85em; }

/* Disclaimer box */
.disclaimer {
    background: #f8fafc;
    border-left: 4px solid #94a3b8;
    padding: 10px 16px;
    border-radius: 4px;
    font-size: 0.78em;
    color: #64748b;
    margin-top: 8px;
}

/* Section headers */
.section-header {
    font-size: 1.05em;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

DISCLAIMER = (
    "⚕️ **Disclaimer:** This system is a decision-support prototype and does not replace "
    "qualified maintenance, biomedical engineering, regulatory, or clinical judgment. "
    "It is not a certified medical device and does not guarantee patient safety outcomes."
)

# ── Sidebar — backend status ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=60)
    st.title("Medical Device\nRisk Dashboard")
    st.caption("Cognizant NPN Hackathon — Healthcare Track")
    st.divider()

    health = get_health()
    if health:
        st.success("🟢 Backend connected")
        st.caption(f"Model: `{health.get('model_version', 'unknown')}`")
        st.caption(f"Manifest: `{health.get('data_manifest_hash', 'unknown')}`")
    else:
        st.error(f"🔴 Backend unreachable\n`{BACKEND_URL}`")
        st.info("Start the FastAPI server:\n```\nuvicorn backend.main:app --reload\n```")

    st.divider()
    st.markdown(
        "<div class='disclaimer'>" + DISCLAIMER.replace("⚕️ **Disclaimer:** ", "") + "</div>",
        unsafe_allow_html=True,
    )

# ── Landing page content ───────────────────────────────────────────────────────
st.title("🏥 Medical Device Failure Risk System")
st.markdown(
    "Real FDA device, event, and manufacturer data — "
    "ML risk scoring, SHAP explanations, and rule-based maintenance prioritisation."
)
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.page_link("pages/1_📊_Overview.py",        label="📊 Overview",          icon="📊")
col2.page_link("pages/2_🔍_Device_Search.py",   label="🔍 Device Search",     icon="🔍")
col3.page_link("pages/3_📋_Device_Detail.py",   label="📋 Device Details",    icon="📋")
col4.page_link("pages/4_🧠_Explainability.py",  label="🧠 Explainability",    icon="🧠")

st.divider()
st.markdown(DISCLAIMER)
