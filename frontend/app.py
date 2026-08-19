"""
frontend/app.py
================
Medical Device Risk Dashboard — Streamlit entry point.

Run:
    streamlit run frontend/app.py

BACKEND_URL env var controls the FastAPI server. Defaults to http://localhost:8000.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from utils.api_client import BACKEND_URL, get_health
from utils.styles import DISCLAIMER, inject, page_header, sidebar_base

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Medical Device Risk Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(inject(), unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_base(active_page="Home", show_model=True)

# ── Landing page header ────────────────────────────────────────────────────────
page_header(
    icon="🏥",
    title="Medical Device Risk Intelligence",
    subtitle=(
        "An AI-powered Medical Device Safety Intelligence platform that analyzes reported safety events, "
        "classifies their severity, retrieves similar historical incidents, explains the model's decision, "
        "identifies historical risk patterns, and provides maintenance-priority decision support."
    ),
    eyebrow="Medical Device Risk Dashboard",
)

st.divider()

# ── Navigation cards ───────────────────────────────────────────────────────────
# NOTE: icon= and label= in st.page_link must NOT both contain the same emoji.
# We remove the emoji from label= and keep only icon= to prevent duplication.
st.markdown("<div style='margin-bottom: 6px; font-size: 0.82em; font-weight: 600; color: var(--text-color); opacity: 0.5; text-transform: uppercase; letter-spacing: 0.07em;'>Navigate to</div>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.page_link("pages/1_📊_Overview.py",        label="Overview",        icon="📊")
col2.page_link("pages/2_🔍_Device_Search.py",   label="Device Search",   icon="🔍")
col3.page_link("pages/3_📋_Device_Detail.py",   label="Device Details",  icon="📋")
col4.page_link("pages/4_🧠_Explainability.py",  label="Explainability",  icon="🧠")
col5.page_link("pages/5_🩺_Risk_Assessment.py", label="Risk Assessment", icon="🩺")

st.divider()

# ── Feature summary ────────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-title'>What this system provides</div>",
    unsafe_allow_html=True,
)

f1, f2, f3, f4, f5 = st.columns(5)
with f1:
    st.markdown(
        "<div class='card'>"
        "<div style='font-size:1.4em; margin-bottom:8px;'>📊</div>"
        "<div style='font-weight:700; font-size:0.9em; color:var(--text-color); margin-bottom:5px;'>Fleet Overview</div>"
        "<div style='font-size:0.81em; color:var(--text-color); opacity:0.55; line-height:1.4;'>Aggregate risk statistics, device counts, and category/manufacturer breakdowns from historical data.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        "<div class='card'>"
        "<div style='font-size:1.4em; margin-bottom:8px;'>🔍</div>"
        "<div style='font-weight:700; font-size:0.9em; color:var(--text-color); margin-bottom:5px;'>Device Search</div>"
        "<div style='font-size:0.81em; color:var(--text-color); opacity:0.55; line-height:1.4;'>Filter and search across the full historical device database by risk level, manufacturer, or category.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
with f3:
    st.markdown(
        "<div class='card'>"
        "<div style='font-size:1.4em; margin-bottom:8px;'>📋</div>"
        "<div style='font-weight:700; font-size:0.9em; color:var(--text-color); margin-bottom:5px;'>Device Detail</div>"
        "<div style='font-size:0.81em; color:var(--text-color); opacity:0.55; line-height:1.4;'>Full device profile with historical event risk score, SHAP explanation, and maintenance-priority recommendation.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
with f4:
    st.markdown(
        "<div class='card'>"
        "<div style='font-size:1.4em; margin-bottom:8px;'>🧠</div>"
        "<div style='font-weight:700; font-size:0.9em; color:var(--text-color); margin-bottom:5px;'>SHAP Explainability</div>"
        "<div style='font-size:0.81em; color:var(--text-color); opacity:0.55; line-height:1.4;'>Feature importance and per-device SHAP explanations — understand why the model produced its result.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
with f5:
    st.markdown(
        "<div class='card'>"
        "<div style='font-size:1.4em; margin-bottom:8px;'>🩺</div>"
        "<div style='font-weight:700; font-size:0.9em; color:var(--text-color); margin-bottom:5px;'>Risk Assessment</div>"
        "<div style='font-size:0.81em; color:var(--text-color); opacity:0.55; line-height:1.4;'>Describe a device problem → ML event risk classification + historical evidence + decision support.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()
st.markdown(f"<div class='disclaimer'>⚕️ {DISCLAIMER}</div>", unsafe_allow_html=True)
