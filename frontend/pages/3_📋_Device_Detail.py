"""
frontend/pages/3_📋_Device_Detail.py
======================================
Device Detail page — full profile for a single device.

Sections:
  1. Device attributes        (GET /devices/{id})
  2. Risk Prediction card     (from /devices/{id} data)
  3. SHAP Explanation chart   (GET /explanation/{id})
  4. Historical Event Summary (from recommendation rule_inputs)
  5. Maintenance Recommendation (GET /recommendation/{id})
  6. Copilot Q&A panel        (POST /copilot)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils.api_client import (
    BACKEND_URL,
    get_device_detail,
    get_explanation,
    get_health,
    get_recommendation,
    post_copilot,
)
from utils.charts import risk_score_gauge, shap_waterfall

st.set_page_config(
    page_title="Device Details | Medical Device Risk",
    page_icon="📋",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.badge-HIGH   { background:#fef2f2;color:#dc2626;padding:3px 12px;border-radius:20px;font-weight:600;font-size:0.85em; }
.badge-MEDIUM { background:#fffbeb;color:#d97706;padding:3px 12px;border-radius:20px;font-weight:600;font-size:0.85em; }
.badge-LOW    { background:#f0fdf4;color:#16a34a;padding:3px 12px;border-radius:20px;font-weight:600;font-size:0.85em; }
.badge-NA     { background:#f1f5f9;color:#64748b;padding:3px 12px;border-radius:20px;font-weight:600;font-size:0.85em; }
.prio-Critical { background:#fef2f2;color:#b91c1c;padding:4px 14px;border-radius:20px;font-weight:700; }
.prio-High     { background:#fff7ed;color:#c2410c;padding:4px 14px;border-radius:20px;font-weight:700; }
.prio-Medium   { background:#fffbeb;color:#b45309;padding:4px 14px;border-radius:20px;font-weight:700; }
.prio-Low      { background:#f0fdf4;color:#15803d;padding:4px 14px;border-radius:20px;font-weight:700; }
.info-row { display:flex; justify-content:space-between; padding:7px 0;
            border-bottom:1px solid #f1f5f9; font-size:0.9em; }
.info-label { color:#64748b; font-weight:500; }
.info-value { color:#1e293b; font-weight:500; text-align:right; max-width:60%; }
.card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:16px 20px; margin-bottom:16px; }
.unavailable { background:#fff7ed; border-left:4px solid #f59e0b;
               padding:12px 16px; border-radius:4px; color:#92400e; }
.disclaimer { background:#f8fafc; border-left:4px solid #94a3b8;
              padding:10px 16px; border-radius:4px; font-size:0.78em; color:#64748b; }
</style>
""", unsafe_allow_html=True)

DISCLAIMER = (
    "This system is a decision-support prototype and does not replace qualified "
    "maintenance, biomedical engineering, regulatory, or clinical judgment. "
    "It is not a certified medical device and does not guarantee patient safety outcomes."
)


def _clean(val):
    if val is None or str(val).lower() in ("nan", "none", ""):
        return "—"
    return str(val)


def _badge(level):
    if not level:
        return "<span class='badge-NA'>N/A</span>"
    return f"<span class='badge-{level}'>{level}</span>"


def _prio_badge(priority):
    cls = f"prio-{priority}" if priority in ("Critical", "High", "Medium", "Low") else "badge-NA"
    return f"<span class='{cls}'>{priority}</span>"


def _info_row(label, value):
    return (
        f"<div class='info-row'>"
        f"<span class='info-label'>{label}</span>"
        f"<span class='info-value'>{value}</span>"
        f"</div>"
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Device Details")
    health = get_health()
    if health:
        st.success("🟢 Backend connected")
    else:
        st.error(f"🔴 Backend unreachable\n`{BACKEND_URL}`")
    st.divider()
    st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)

# ── Device ID input ───────────────────────────────────────────────────────────
st.title("📋 Device Details")

# Pre-fill from session_state (set by Device Search page)
prefill = st.session_state.get("detail_device_id", "")
device_id_input = st.text_input(
    "Device ID",
    value=prefill,
    placeholder="Enter a device ID, e.g. 80508",
    key="detail_device_id_input",
)

if not device_id_input:
    st.info("Enter a device ID above to load its full profile.")
    st.stop()

device_id = device_id_input.strip()

# ── Fetch all data ─────────────────────────────────────────────────────────────
with st.spinner(f"Loading device {device_id}..."):
    device   = get_device_detail(device_id)
    rec      = get_recommendation(device_id)
    expl     = get_explanation(device_id)

if device is None:
    st.error(f"Device **{device_id}** was not found. Check the ID and try again.")
    st.stop()

prediction_available = device.get("prediction_available", False)
risk_level           = device.get("risk_level")
risk_score           = device.get("risk_score")
cal_prob             = device.get("calibrated_probability")
serving_date         = device.get("serving_event_date")
model_version        = device.get("model_version")

st.divider()

# ── Row 1: Device info + Risk gauge ──────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("Device Information")
    info_html = ""
    info_html += _info_row("Device ID",        _clean(device.get("device_id")))
    info_html += _info_row("Name",             _clean(device.get("device_name")))
    info_html += _info_row("Classification",   _clean(device.get("device_classification")))
    info_html += _info_row("Risk Class (FDA)", _clean(device.get("device_risk_class")))
    info_html += _info_row("Country",          _clean(device.get("device_country")))
    info_html += _info_row("Manufacturer",     _clean(device.get("mfr_parent_company") or device.get("mfr_name")))
    info_html += _info_row("Implanted",        _clean(device.get("device_implanted")))
    info_html += _info_row("Distributed To",   _clean(device.get("device_distributed_to")))
    st.markdown(f"<div class='card'>{info_html}</div>", unsafe_allow_html=True)

with right_col:
    st.subheader("Risk Prediction")
    if prediction_available and risk_score is not None:
        fig_gauge = risk_score_gauge(risk_score, risk_level or "LOW")
        st.plotly_chart(fig_gauge, use_container_width=True)

        details_html = ""
        details_html += _info_row("Risk Level",            _badge(risk_level))
        details_html += _info_row("Calibrated Probability", f"{cal_prob:.4f}" if cal_prob is not None else "—")
        details_html += _info_row("Snapshot Date",          _clean(serving_date))
        details_html += _info_row("Model Version",          _clean(model_version))
        maint = device.get("maintenance_priority")
        if maint:
            details_html += _info_row("Maintenance Priority", _prio_badge(maint))
        st.markdown(f"<div class='card'>{details_html}</div>", unsafe_allow_html=True)
    else:
        reason = device.get("prediction_unavailable_reason", "No valid serving snapshot.")
        st.markdown(
            f"<div class='unavailable'>"
            f"<strong>Prediction Unavailable</strong><br>{reason}"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── SHAP Explanation ──────────────────────────────────────────────────────────
st.subheader("🔬 SHAP Feature Contributions")

if expl is None:
    st.warning("Could not retrieve explanation from the backend.")
elif not expl.get("available"):
    st.markdown(
        f"<div class='unavailable'>"
        f"<strong>Explanation Unavailable</strong><br>"
        f"{expl.get('unavailable_reason', 'No explanation available.')}"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    with st.spinner("Rendering SHAP chart..."):
        fig_shap = shap_waterfall(
            top_positive=expl.get("top_positive", []),
            top_negative=expl.get("top_negative", []),
            base_value=expl.get("base_value", 0.0),
            predicted_value=expl.get("predicted_value", 0.0),
        )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.caption(
        "🔴 Red bars increase predicted risk; 🔵 blue bars decrease it. "
        "SHAP values show each feature's contribution relative to the model's average prediction."
    )

st.divider()

# ── Historical event summary ──────────────────────────────────────────────────
st.subheader("📅 Historical Event Summary")

rule_inputs = {}
if rec and rec.get("available"):
    rule_inputs = rec.get("rule_inputs", {})

h1, h2, h3 = st.columns(3)
h1.metric("Total Events",       int(rule_inputs.get("hist_device_event_count",   0) or 0))
h2.metric("Class I Events",     int(rule_inputs.get("hist_device_class_i_count", 0) or 0),
          help="Class I = most severe FDA recall classification")
h3.metric("Recall Events",      int(rule_inputs.get("hist_device_recall_count",  0) or 0))

st.divider()

# ── Maintenance Recommendation ────────────────────────────────────────────────
st.subheader("🔧 Maintenance Recommendation")

if rec is None:
    st.warning("Could not retrieve recommendation from the backend.")
elif not rec.get("available"):
    st.markdown(
        f"<div class='unavailable'>"
        f"<strong>Recommendation Unavailable</strong><br>"
        f"{rec.get('unavailable_reason', '')}"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    priority = rec.get("maintenance_priority", "Unknown")
    risk_lv  = rec.get("risk_level", "")
    crit     = rec.get("criticality_tier", "")
    actions  = rec.get("recommended_actions", [])

    rec_l, rec_r = st.columns([2, 3])
    with rec_l:
        st.markdown(
            f"<div class='card' style='text-align:center;'>"
            f"<div style='font-size:0.8em;color:#64748b;margin-bottom:8px'>MAINTENANCE PRIORITY</div>"
            f"<div style='font-size:1.6em;font-weight:700'>{_prio_badge(priority)}</div>"
            f"<br>"
            f"<div style='font-size:0.8em;color:#64748b'>Risk: {_badge(risk_lv)} &nbsp;&nbsp; Criticality: <strong>{crit}</strong></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with rec_r:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Recommended Actions**")
        for action in actions:
            st.markdown(f"• {action}")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Rule inputs (full context)"):
        st.json(rule_inputs)

    st.markdown(
        f"<div class='disclaimer'>⚕️ {rec.get('disclaimer', DISCLAIMER)}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Copilot Q&A ───────────────────────────────────────────────────────────────
st.subheader("🤖 Copilot Q&A")
st.caption(
    "Ask a question about this device. The copilot answers only from the structured "
    "context above — it never invents facts."
)

question = st.text_input(
    "Your question:",
    placeholder=f"e.g. Why is device {device_id} flagged as high risk?",
    key="copilot_question",
)

if st.button("Ask Copilot", type="primary", disabled=not question):
    with st.spinner("Asking copilot..."):
        copilot_resp = post_copilot(device_id, question)

    if copilot_resp is None:
        st.error("Copilot is unavailable. Check that the backend is running.")
    else:
        answer   = copilot_resp.get("answer", "")
        llm_used = copilot_resp.get("llm_used", False)
        provider = copilot_resp.get("provider", "fallback")

        st.markdown("**Answer**")
        st.markdown(answer)
        st.caption(
            f"Answered by: `{provider}`"
            + (" (LLM)" if llm_used else " (deterministic fallback — no LLM key configured)")
        )

        with st.expander("Context used by copilot"):
            st.json(copilot_resp.get("context_used", {}))

st.divider()
st.markdown(f"<div class='disclaimer'>⚕️ {DISCLAIMER}</div>", unsafe_allow_html=True)
