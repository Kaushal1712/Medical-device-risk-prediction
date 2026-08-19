"""
frontend/pages/3_📋_Device_Detail.py
======================================
Device Detail page — full profile for a single device.
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
from utils.styles import DISCLAIMER, inject, page_header, sidebar_base

st.set_page_config(
    page_title="Device Details | Medical Device Risk",
    page_icon="📋",
    layout="wide",
)

st.markdown(inject(), unsafe_allow_html=True)


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
    sidebar_base(active_page="Device Details")

# ── Page header ───────────────────────────────────────────────────────────────
page_header(
    icon="📋",
    title="Device Details",
    subtitle="Full device profile — historical event risk score, SHAP explanation, maintenance-priority recommendation, and Copilot Q&A.",
)

# ── Device ID input ───────────────────────────────────────────────────────────
prefill = st.session_state.get("detail_device_id", "")
device_id_input = st.text_input(
    "Device ID",
    value=prefill,
    placeholder="Enter a device ID, e.g. 80508",
    key="detail_device_id_input",
)

if not device_id_input:
    st.markdown(
        "<div class='info-note'>Enter a device ID above to load its full profile. "
        "You can copy a Device ID from the Device Search page.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

device_id = device_id_input.strip()

# ── Fetch all data ─────────────────────────────────────────────────────────────
with st.spinner(f"Loading device {device_id}…"):
    device = get_device_detail(device_id)
    rec    = get_recommendation(device_id)
    expl   = get_explanation(device_id)

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
    st.markdown("<div class='section-title'>Device Information</div>", unsafe_allow_html=True)
    info_html = ""
    info_html += _info_row("Device ID",                 _clean(device.get("device_id")))
    info_html += _info_row("Name",                      _clean(device.get("device_name")))
    info_html += _info_row("Classification",             _clean(device.get("device_classification")))
    info_html += _info_row("Regulatory Risk Class (FDA)", _clean(device.get("device_risk_class")))
    info_html += _info_row("Country",                   _clean(device.get("device_country")))
    info_html += _info_row("Manufacturer",              _clean(device.get("mfr_parent_company") or device.get("mfr_name")))
    info_html += _info_row("Implanted",                 _clean(device.get("device_implanted")))
    info_html += _info_row("Distributed To",            _clean(device.get("device_distributed_to")))
    st.markdown(f"<div class='card'>{info_html}</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='section-title'>Historical Event Risk Score</div>", unsafe_allow_html=True)
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
st.markdown("<div class='section-title'>🔬 SHAP Feature Contributions</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-intro'>🔴 Red bars increase predicted risk; 🔵 blue bars decrease it. "
    "SHAP values show each feature's contribution relative to the model's average prediction.</div>",
    unsafe_allow_html=True,
)

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
    with st.spinner("Rendering SHAP chart…"):
        fig_shap = shap_waterfall(
            top_positive=expl.get("top_positive", []),
            top_negative=expl.get("top_negative", []),
            base_value=expl.get("base_value", 0.0),
            predicted_value=expl.get("predicted_value", 0.0),
        )
    st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# ── Historical event summary ──────────────────────────────────────────────────
st.markdown("<div class='section-title'>📅 Historical Event Summary</div>", unsafe_allow_html=True)

rule_inputs = {}
if rec and rec.get("available"):
    rule_inputs = rec.get("rule_inputs", {})

h1, h2, h3 = st.columns(3)
h1.metric("Total Events",   int(rule_inputs.get("hist_device_event_count",   0) or 0))
h2.metric("Class I Events", int(rule_inputs.get("hist_device_class_i_count", 0) or 0),
          help="Class I = most severe FDA recall classification")
h3.metric("Recall Events",  int(rule_inputs.get("hist_device_recall_count",  0) or 0))

st.divider()

# ── Maintenance Recommendation ────────────────────────────────────────────────
st.markdown("<div class='section-title'>🔧 Maintenance Recommendation</div>", unsafe_allow_html=True)

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
            f"<div style='font-size:0.75em;color:var(--text-color);opacity:0.5;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.07em;margin-bottom:10px;'>Maintenance Priority</div>"
            f"<div style='font-size:1.5em;font-weight:700;margin-bottom:10px;'>{_prio_badge(priority)}</div>"
            f"<div style='font-size:0.82em;color:var(--text-color);opacity:0.6;'>"
            f"Risk: {_badge(risk_lv)} &nbsp;&nbsp; Criticality: <strong>{crit}</strong>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with rec_r:
        # Build the actions list inside a single st.markdown call — no split card issue
        actions_li = "".join(
            f"<li style='margin-bottom:5px;'>{action}</li>" for action in actions
        )
        st.markdown(
            f"<div class='card'>"
            f"<div style='font-weight:700;margin-bottom:10px;color:var(--text-color);'>Recommended Actions</div>"
            f"<ul style='margin:0;padding-left:18px;color:var(--text-color);font-size:0.9em;line-height:1.6;'>"
            f"{actions_li}"
            f"</ul>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Rule inputs (full context)"):
        st.json(rule_inputs)

    st.markdown(
        f"<div class='disclaimer'>⚕️ {rec.get('disclaimer', DISCLAIMER)}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Copilot Q&A ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>🤖 Copilot Q&A</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-intro'>"
    "Ask a question about this device. The copilot answers only from the structured "
    "context above — it never invents facts."
    "</div>",
    unsafe_allow_html=True,
)

question = st.text_input(
    "Your question:",
    placeholder=f"e.g. Why is device {device_id} flagged as high risk?",
    key="copilot_question",
)

if st.button("Ask Copilot", type="primary", disabled=not question):
    with st.spinner("Asking copilot…"):
        copilot_resp = post_copilot(device_id, question)

    if copilot_resp is None:
        st.error("Copilot is unavailable. Check that the backend is running.")
    else:
        answer   = copilot_resp.get("answer", "")
        llm_used = copilot_resp.get("llm_used", False)
        provider = copilot_resp.get("provider", "fallback")

        st.markdown(
            f"<div class='card-accent'>"
            f"<div style='font-weight:700;margin-bottom:8px;color:var(--text-color);'>Answer</div>"
            f"<div style='color:var(--text-color);font-size:0.92em;line-height:1.6;'>{answer}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Answered by: `{provider}`"
            + (" (LLM)" if llm_used else " (deterministic fallback — no LLM key configured)")
        )

        with st.expander("Context used by copilot"):
            st.json(copilot_resp.get("context_used", {}))

st.divider()
st.markdown(f"<div class='disclaimer'>⚕️ {DISCLAIMER}</div>", unsafe_allow_html=True)
