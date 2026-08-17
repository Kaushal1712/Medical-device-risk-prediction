"""
frontend/pages/1_📊_Overview.py
=================================
Overview page — aggregate risk statistics from GET /risk-summary.

Shows:
  - KPI row: total devices, scored, unscored
  - Risk level distribution bar chart
  - Risk score statistics
  - Top-15 category breakdown (stacked bar)
  - Top-15 manufacturer breakdown (stacked bar)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils.api_client import BACKEND_URL, get_health, get_risk_summary
from utils.charts import (
    RISK_COLOR,
    breakdown_stacked_bar,
    risk_distribution_bar,
)

st.set_page_config(
    page_title="Overview | Medical Device Risk",
    page_icon="📊",
    layout="wide",
)

# ── Global styles (shared CSS injected on every page) ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
[data-testid="metric-container"] {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:12px; padding:16px 20px;
}
.disclaimer { background:#f8fafc; border-left:4px solid #94a3b8;
    padding:10px 16px; border-radius:4px; font-size:0.78em; color:#64748b; }
.stat-label { font-size:0.78em; color:#64748b; font-weight:500; text-transform:uppercase; letter-spacing:0.05em; }
.stat-value { font-size:1.6em; font-weight:700; color:#1e293b; }
</style>
""", unsafe_allow_html=True)

DISCLAIMER = (
    "This system is a decision-support prototype and does not replace qualified "
    "maintenance, biomedical engineering, regulatory, or clinical judgment. "
    "It is not a certified medical device and does not guarantee patient safety outcomes."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Overview")
    health = get_health()
    if health:
        st.success("🟢 Backend connected")
        st.caption(f"Model: `{health.get('model_version','unknown')}`")
    else:
        st.error(f"🔴 Backend unreachable\n`{BACKEND_URL}`")
    st.divider()
    st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📊 Risk Overview")
st.caption("All statistics are derived from the production serving snapshot (Stage 3f policy).")

with st.spinner("Loading risk summary..."):
    summary = get_risk_summary()

if summary is None:
    st.error(
        "Could not reach the backend API. "
        f"Make sure `uvicorn backend.main:app --reload` is running at `{BACKEND_URL}`."
    )
    st.stop()

risk_levels = summary.get("risk_levels", {})
score_stats = summary.get("risk_score_stats", {})

# ── KPI row ───────────────────────────────────────────────────────────────────
st.subheader("Fleet Summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Total Devices",   f"{summary['total_devices_in_data']:,}")
c2.metric("Scored",          f"{summary['total_scored']:,}")
c3.metric("Unscored",        f"{summary['total_unscored']:,}")

high_count   = risk_levels.get("HIGH",   {}).get("count", 0)
medium_count = risk_levels.get("MEDIUM", {}).get("count", 0)
low_count    = risk_levels.get("LOW",    {}).get("count", 0)

c4.metric("🔴 HIGH",   f"{high_count:,}",
          f"{risk_levels.get('HIGH',{}).get('percent',0):.1f}%")
c5.metric("🟡 MEDIUM", f"{medium_count:,}",
          f"{risk_levels.get('MEDIUM',{}).get('percent',0):.1f}%")
c6.metric("🟢 LOW",    f"{low_count:,}",
          f"{risk_levels.get('LOW',{}).get('percent',0):.1f}%")

st.divider()

# ── Risk distribution + score stats ──────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("Risk Level Distribution")
    fig_dist = risk_distribution_bar(risk_levels)
    st.plotly_chart(fig_dist, use_container_width=True)

with right:
    st.subheader("Risk Score Statistics (0 – 100)")
    for label, key in [
        ("Minimum",  "min"),
        ("Mean",     "mean"),
        ("Median",   "median"),
        ("Maximum",  "max"),
    ]:
        val = score_stats.get(key, 0)
        # Color the maximum red, minimum green
        color = "#dc2626" if key == "max" else "#16a34a" if key == "min" else "#1e293b"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:8px 0;"
            f"border-bottom:1px solid #f1f5f9'>"
            f"<span class='stat-label'>{label}</span>"
            f"<span style='font-weight:700;color:{color}'>{val:.1f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Category breakdown ────────────────────────────────────────────────────────
st.subheader("Top 15 Device Categories")
cat_items = summary.get("category_breakdown", [])
if cat_items:
    fig_cat = breakdown_stacked_bar(cat_items, "category", "HIGH / MEDIUM / LOW by Device Category")
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("No category breakdown data available.")

st.divider()

# ── Manufacturer breakdown ────────────────────────────────────────────────────
st.subheader("Top 15 Manufacturers")
mfr_items = summary.get("manufacturer_breakdown", [])
if mfr_items:
    fig_mfr = breakdown_stacked_bar(mfr_items, "manufacturer", "HIGH / MEDIUM / LOW by Manufacturer")
    st.plotly_chart(fig_mfr, use_container_width=True)
else:
    st.info("No manufacturer breakdown data available.")

st.divider()
st.markdown(f"<div class='disclaimer'>⚕️ {DISCLAIMER}</div>", unsafe_allow_html=True)
