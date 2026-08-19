"""
frontend/pages/1_📊_Overview.py
=================================
Overview page — aggregate risk statistics from GET /risk-summary.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils.api_client import BACKEND_URL, get_health, get_risk_summary
from utils.charts import RISK_COLOR, breakdown_stacked_bar, risk_distribution_bar
from utils.styles import DISCLAIMER, inject, page_header, sidebar_base

st.set_page_config(
    page_title="Overview | Medical Device Risk",
    page_icon="📊",
    layout="wide",
)

st.markdown(inject(), unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_base(active_page="Overview", show_model=True)

# ── Page header ───────────────────────────────────────────────────────────────
page_header(
    icon="📊",
    title="Risk Overview",
    subtitle="Aggregate statistics from the production serving snapshot — historical risk distribution across the full device fleet.",
)

# ── System mission note ─────────────────────────────────────────────────────
st.markdown(
    "<div class='info-note'>"
    "🏥 <b>Medical Device Risk Intelligence</b> — "
    "An AI-assisted platform for historical medical-device safety event analysis and maintenance prioritization. "
    "By classifying reported safety events using ML and surfacing historical adverse-event patterns, "
    "the system helps biomedical engineers and investigators quickly identify devices associated with higher-severity outcomes "
    "and prioritize them for inspection or preventive maintenance — reducing patient risk and minimizing unplanned downtime. "
    "<br><em>This platform classifies the severity of already-reported safety events. "
    "It does not predict future device failure. Historical event risk patterns are used to support preventive investigation decisions.</em>"
    "</div>",
    unsafe_allow_html=True,
)

with st.spinner("Loading risk summary…"):
    summary = get_risk_summary()

if summary is None:
    st.error(
        f"Could not reach the backend API. "
        f"Make sure `uvicorn backend.main:app --reload` is running at `{BACKEND_URL}`."
    )
    st.stop()

risk_levels = summary.get("risk_levels", {})
score_stats = summary.get("risk_score_stats", {})

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Fleet Summary</div>", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Total Devices",   f"{summary['total_devices_in_data']:,}")
c2.metric("Scored",          f"{summary['total_scored']:,}")
c3.metric("Unscored",        f"{summary['total_unscored']:,}")

high_count   = risk_levels.get("HIGH",   {}).get("count", 0)
medium_count = risk_levels.get("MEDIUM", {}).get("count", 0)
low_count    = risk_levels.get("LOW",    {}).get("count", 0)

c4.metric("HIGH Risk",   f"{high_count:,}",
          f"{risk_levels.get('HIGH',{}).get('percent',0):.1f}%")
c5.metric("MEDIUM Risk", f"{medium_count:,}",
          f"{risk_levels.get('MEDIUM',{}).get('percent',0):.1f}%")
c6.metric("LOW Risk",    f"{low_count:,}",
          f"{risk_levels.get('LOW',{}).get('percent',0):.1f}%")

st.divider()

# ── Risk distribution + score stats ──────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown("<div class='section-title'>Risk Level Distribution</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-intro'>Device count and percentage across HIGH / MEDIUM / LOW risk tiers.</div>", unsafe_allow_html=True)
    fig_dist = risk_distribution_bar(risk_levels)
    st.plotly_chart(fig_dist, use_container_width=True)

with right:
    st.markdown("<div class='section-title'>Risk Score Statistics (0 – 100)</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-intro'>Distribution of the continuous risk score across scored devices.</div>", unsafe_allow_html=True)

    # Theme-safe stat rows — no hardcoded light-mode hex colors
    stat_rows_html = ""
    for label, key, is_high, is_low in [
        ("Minimum", "min",    False, True),
        ("Mean",    "mean",   False, False),
        ("Median",  "median", False, False),
        ("Maximum", "max",    True,  False),
    ]:
        val = score_stats.get(key, 0)
        val_class = "val-high" if is_high else ("val-low" if is_low else "")
        stat_rows_html += (
            f"<div class='stat-row'>"
            f"<span class='stat-row-label'>{label}</span>"
            f"<span class='stat-row-value {val_class}'>{val:.1f}</span>"
            f"</div>"
        )

    st.markdown(
        f"<div class='card' style='padding: 14px 18px;'>{stat_rows_html}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Category breakdown ────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Top 15 Device Categories</div>", unsafe_allow_html=True)
st.markdown("<div class='section-intro'>HIGH / MEDIUM / LOW distribution per FDA device category.</div>", unsafe_allow_html=True)
cat_items = summary.get("category_breakdown", [])
if cat_items:
    fig_cat = breakdown_stacked_bar(cat_items, "category", "HIGH / MEDIUM / LOW by Device Category")
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("No category breakdown data available.")

st.divider()

# ── Manufacturer breakdown ────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Top 15 Manufacturers</div>", unsafe_allow_html=True)
st.markdown("<div class='section-intro'>HIGH / MEDIUM / LOW distribution per manufacturer.</div>", unsafe_allow_html=True)
mfr_items = summary.get("manufacturer_breakdown", [])
if mfr_items:
    fig_mfr = breakdown_stacked_bar(mfr_items, "manufacturer", "HIGH / MEDIUM / LOW by Manufacturer")
    st.plotly_chart(fig_mfr, use_container_width=True)
else:
    st.info("No manufacturer breakdown data available.")

st.divider()
st.markdown(f"<div class='disclaimer'>⚕️ {DISCLAIMER}</div>", unsafe_allow_html=True)
