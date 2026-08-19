"""
frontend/pages/5_🩺_Risk_Assessment.py
========================================
Query-driven Risk Assessment UI — calls POST /assess.

Allows a user to describe a device and an observed problem, and get:
  - ML risk classification (Class I likelihood)
  - Historical similar events
  - Device metadata (if device_id supplied)
  - Evidence and disclaimer
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils.api_client import BACKEND_URL, post_assess
from utils.charts import shap_waterfall
from utils.styles import DISCLAIMER, inject, page_header, sidebar_base

def _badge(level):
    if not level:
        return "<span class='badge-NA'>N/A</span>"
    return f"<span class='badge-{level}'>{level}</span>"

def _prio_badge(priority):
    cls = f"prio-{priority}" if priority in ("Critical", "High", "Medium", "Low") else "badge-NA"
    return f"<span class='{cls}'>{priority}</span>"

st.set_page_config(
    page_title="Risk Assessment — Medical Device Risk",
    page_icon="🩺",
    layout="wide",
)

st.markdown(inject(), unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_base(active_page="Risk Assessment")

# ── Page header ───────────────────────────────────────────────────────────────
page_header(
    icon="🩺",
    title="Risk Assessment",
    subtitle=(
        "Describe a device and observed safety problem to receive an ML-based event "
        "risk classification, historical evidence, and decision support."
    ),
)

st.markdown(
    "<div class='info-note'>"
    "🎯 <b>What this system does:</b> The ML model estimates the probability that a "
    "described safety event would be <b>classified as Class I</b> by the FDA — events "
    "most likely to cause serious patient harm or death. &nbsp;"
    "<b>It does not predict whether a device will fail in the future.</b> "
    "Historical evidence is retrieved separately from the safety database."
    "</div>",
    unsafe_allow_html=True,
)

# ── Key distinction note ──────────────────────────────────────────────────────
st.markdown(
    "<div class='info-note' style='border-left-color:#d97706; background:rgba(217,119,6,0.08);'>"
    "⚠️ <b>Important distinction:</b> "
    "<b>Device Class</b> (Class 1 / 2 / 3) describes the <em>regulatory category</em> of the device — "
    "a device-level regulatory classification. &nbsp;"
    "<b>Risk Level</b> (LOW / MEDIUM / HIGH) is the <em>model's assessment of the reported safety event</em>, "
    "based on the problem description and historical patterns. "
    "A Class III device does not automatically mean the reported event is HIGH risk."
    "</div>",
    unsafe_allow_html=True,
)

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("assess_form", clear_on_submit=False):
    st.markdown(
        "<div class='section-title'>📝 Describe the Device &amp; Problem</div>",
        unsafe_allow_html=True,
    )

    # Required fields
    st.markdown(
        "<div class='field-group-label' style='color:#2563eb;font-size:0.72em;font-weight:700;"
        "letter-spacing:0.09em;text-transform:uppercase;margin-bottom:6px;margin-top:4px;'>"
        "✱ Required fields</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        device_information = st.text_area(
            "Device Information ✱",
            placeholder="e.g. Implanted cardiac defibrillator, model ICD-X200",
            height=100,
            help="Describe the device — type, model, characteristics. Combined with problem description for text features.",
        )
    with col_b:
        problem_description = st.text_area(
            "Problem Description ✱",
            placeholder=(
                "e.g. Repeated electrical faults and abnormal battery behavior observed. "
                "Device delivering inappropriate shocks."
            ),
            height=100,
            help=(
                "Describe the observed safety problem in detail. "
                "This is the primary ML input — the model was trained on historical 'reason' fields."
            ),
        )

    # Optional fields
    st.markdown(
        "<div style='height:8px;'></div>"
        "<div class='field-group-label' style='color:var(--text-color);opacity:0.45;font-size:0.72em;"
        "font-weight:700;letter-spacing:0.09em;text-transform:uppercase;margin-bottom:6px;'>"
        "Optional fields — improve prediction quality when provided</div>",
        unsafe_allow_html=True,
    )

    col_opt1, col_opt2 = st.columns([1, 1], gap="large")

    with col_opt1:
        device_id = st.text_input(
            "Device ID",
            placeholder="e.g. 80508",
            help=(
                "If provided, retrieves historical events and device metadata for context. "
                "Device ID is NEVER used as a predictive ML feature."
            ),
        )
        device_classification = st.text_input(
            "Device Classification",
            placeholder="e.g. Cardiovascular Devices",
        )

    with col_opt2:
        device_risk_class = st.selectbox(
            "Device Risk Class (regulatory, not event risk)",
            options=["", "1", "2", "3", "HDE", "Not Classified"],
            index=0,
            help=(
                "This is the FDA regulatory device class — a device-level category. "
                "It is NOT the same as the event risk level computed by the ML model."
            ),
        )
        device_implanted = st.selectbox(
            "Implanted?",
            options=["", "YES", "NO"],
            index=0,
        )
        country = st.selectbox(
            "Country",
            options=["", "USA", "CAN", "AUS"],
            index=0,
        )

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "🔍  Assess Risk",
        use_container_width=True,
        type="primary",
    )

# ── Validation & submission ───────────────────────────────────────────────────
if submitted:
    errors = []
    if not device_information or len(device_information.strip()) < 3:
        errors.append("**Device Information** must be at least 3 characters.")
    if not problem_description or len(problem_description.strip()) < 10:
        errors.append("**Problem Description** must be at least 10 characters.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        with st.spinner("Running ML risk assessment…"):
            result = post_assess(
                device_information=device_information.strip(),
                problem_description=problem_description.strip(),
                device_id=device_id.strip() or None,
                device_classification=device_classification.strip() or None,
                device_risk_class=device_risk_class or None,
                device_implanted=device_implanted or None,
                country=country or None,
            )

        if result is None:
            st.error(
                f"❌ Could not reach the backend at `{BACKEND_URL}`. "
                "Ensure the FastAPI server is running:\n```\nuvicorn backend.main:app --reload\n```"
            )
        else:
            # ── PREDICTION BLOCK ──────────────────────────────────────────────
            pred        = result.get("prediction", {})
            risk_level  = pred.get("risk_level", "N/A")
            risk_score  = pred.get("risk_score", 0.0)
            raw_prob    = pred.get("raw_probability", 0.0)
            target_desc = pred.get("target_description", "")
            model_ver   = pred.get("model_version", "unknown")

            st.divider()
            st.markdown(
                "<div class='section-title'>📊 Assessment Result</div>",
                unsafe_allow_html=True,
            )

            # Semantic risk color
            _color_map = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}
            rl_color   = _color_map.get(risk_level, "#64748b")

            r1, r2, r3 = st.columns(3, gap="medium")

            with r1:
                st.markdown(
                    f"<div class='result-card' style='border-top:4px solid {rl_color};'>"
                    f"<div class='label'>Event Risk Level</div>"
                    f"<div class='value-{risk_level}'>{risk_level}</div>"
                    f"<div style='font-size:0.65em;color:var(--text-color);opacity:0.45;margin-top:6px;'>"
                    f"LOW &lt;20 · MEDIUM 20–50 · HIGH ≥50"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f"<div class='result-card' style='border-top:4px solid rgba(148,163,184,0.5);'>"
                    f"<div class='label'>Risk Score</div>"
                    f"<div class='value-num'>{risk_score:.1f}"
                    f"<span class='value-sub'> / 100</span></div>"
                    f"<div style='font-size:0.65em;color:var(--text-color);opacity:0.45;margin-top:6px;'>"
                    f"Score = probability × 100"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with r3:
                pct = f"{raw_prob:.1%}"
                st.markdown(
                    f"<div class='result-card' style='border-top:4px solid #2563eb;'>"
                    f"<div class='label'>Class I Probability</div>"
                    f"<div class='value-num' style='color:#2563eb;'>{pct}</div>"
                    f"<div style='font-size:0.65em;color:var(--text-color);opacity:0.45;margin-top:6px;'>"
                    f"Raw model probability"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # What does this mean?
            st.markdown(
                "<div class='info-note' style='margin-top:8px;'>"
                "<b>What does this mean?</b><br>"
                f"The model estimates a <b>{pct}</b> likelihood that this reported safety event "
                "would be classified as Class I (the most serious FDA recall category). "
                f"A score of <b>{risk_score:.1f} / 100</b> places this event in the "
                f"<b style='color:{rl_color};'>{risk_level}</b> risk band "
                "(LOW &lt; 20 · MEDIUM 20–50 · HIGH ≥ 50). "
                "<br><em>This is NOT a prediction that the device will fail in the future.</em>"
                "</div>",
                unsafe_allow_html=True,
            )

            # Show regulatory class vs event risk level distinction if class was provided
            if device_risk_class:
                reg_class_label = f"Class {device_risk_class}" if device_risk_class not in ("HDE", "Not Classified") else device_risk_class
                st.markdown(
                    f"<div class='info-note' style='border-left-color:#d97706;background:rgba(217,119,6,0.08);margin-top:4px;'>"
                    f"📋 <b>Device regulatory class:</b> {reg_class_label} — "
                    f"this is the FDA device category, independent of the event risk assessment above. "
                    f"Event Risk Level ({risk_level}) is computed from the problem description and historical patterns."
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.caption(f"Model: `{model_ver}`  ·  device_id NOT used as a predictive feature")

            # ── PREVENTIVE RISK BLOCK ─────────────────────────────────────────
            prev_risk = result.get("preventive_risk", {})
            pr_level = prev_risk.get("level", "UNKNOWN")
            pr_note = prev_risk.get("score_note", "")

            st.divider()
            st.markdown(
                "<div class='section-title'>🛡️ Historical Preventive Risk</div>",
                unsafe_allow_html=True,
            )
            
            pr_color = _color_map.get(pr_level, "#64748b")
            
            p1, p2 = st.columns([1, 2], gap="medium")
            with p1:
                st.markdown(
                    f"<div class='result-card' style='border-top:4px solid {pr_color};'>"
                    f"<div class='label'>Preventive Risk Level</div>"
                    f"<div class='value-{pr_level}'>{pr_level}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with p2:
                st.markdown(
                    "<div class='info-note' style='height:100%; display:flex; flex-direction:column; justify-content:center;'>"
                    "<b>What does this mean?</b><br>"
                    "This risk level is based exclusively on the historical recurrence and safety patterns "
                    "associated with this device profile.<br><br>"
                    f"<i>{pr_note}</i><br><br>"
                    "<b>⚠️ NOT a guarantee of future failure.</b><br>"
                    "Direct future-failure prediction would require prospective failure labels or continuous telemetry. "
                    "This is a decision-support indicator prioritizing preventive investigation."
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ── DEVICE METADATA ───────────────────────────────────────────────
            device_info = result.get("device_info")
            if device_info:
                st.markdown(
                    "<div class='section-title'>🔧 Device Metadata</div>",
                    unsafe_allow_html=True,
                )
                di1, di2 = st.columns(2, gap="medium")
                with di1:
                    st.write(f"**Name:** {device_info.get('device_name') or '—'}")
                    st.write(f"**Classification:** {device_info.get('device_classification') or '—'}")
                    st.write(f"**Regulatory Risk Class:** {device_info.get('device_risk_class') or '—'}")
                    st.write(f"**Implanted:** {device_info.get('device_implanted') or '—'}")
                with di2:
                    st.write(f"**Manufacturer:** {device_info.get('mfr_name') or '—'}")
                    st.write(f"**Parent Company:** {device_info.get('mfr_parent_company') or '—'}")
                    st.write(f"**Country:** {device_info.get('device_country') or '—'}")
                    st.write(f"**Data Source:** {device_info.get('mfr_source') or '—'}")

            # ── HISTORICAL EVIDENCE ───────────────────────────────────────────
            hist_ev = result.get("historical_evidence", {})

            # Historical aggregate facts (show first as summary)
            device_facts = hist_ev.get("device_facts")
            if device_facts and device_facts.get("total_events", 0) > 0:
                st.markdown(
                    "<div class='section-title'>📈 Device History Summary</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class='section-intro'>Historical aggregate facts retrieved from the safety database for this device.</div>",
                    unsafe_allow_html=True,
                )
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Total Historical Events", device_facts.get("total_events", 0))
                fc2.metric("Class I Events",          device_facts.get("class_i_events", 0),
                           help="Class I = most severe FDA recall classification")
                fc3.metric("Recall Events",           device_facts.get("recall_events", 0))
                earliest = str(device_facts.get("earliest_event", ""))[:10]
                latest   = str(device_facts.get("latest_event", ""))[:10]
                if earliest:
                    st.caption(f"Historical event date range: {earliest} → {latest}")

            # Device-specific events
            device_events = hist_ev.get("device_events", [])
            if device_events:
                st.markdown(
                    f"<div class='section-title'>📂 Historical Events for This Device "
                    f"<span style='font-weight:400;color:var(--text-color);opacity:0.45;font-size:0.85em;'>({len(device_events)} shown)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class='section-intro'>Past safety events recorded for this specific device in the historical database.</div>",
                    unsafe_allow_html=True,
                )
                for ev in device_events:
                    ev_date = str(ev.get("event_date", ""))[:10] if ev.get("event_date") else "—"
                    ev_type = ev.get("event_type") or "—"
                    ev_dev  = ev.get("device_name") or "—"
                    ev_reas = ev.get("reason") or "*(no reason recorded)*"
                    st.markdown(
                        f"<div class='event-card'>"
                        f"<b>{ev_type}</b>"
                        f"<small> · {ev_date} · {ev_dev}</small>"
                        f"<br><i>{ev_reas}</i>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Similar events (FTS search)
            similar_events = hist_ev.get("similar_events", [])
            if similar_events:
                st.markdown(
                    f"<div class='section-title'>🔍 Similar Historical Events "
                    f"<span style='font-weight:400;color:var(--text-color);opacity:0.45;font-size:0.85em;'>({len(similar_events)} found via full-text search)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class='section-intro'>"
                    "Historical events with similar problem descriptions retrieved from the safety database. "
                    "Shown as <strong>historical evidence</strong> — "
                    "these events were <strong>not used in the ML prediction</strong>."
                    "</div>",
                    unsafe_allow_html=True,
                )
                for ev in similar_events:
                    ev_date = str(ev.get("event_date", ""))[:10] if ev.get("event_date") else "—"
                    ev_type = ev.get("event_type") or "—"
                    ev_dev  = ev.get("device_name") or "—"
                    ev_cls  = ev.get("device_classification") or ""
                    ev_reas = ev.get("reason") or "*(no reason recorded)*"
                    st.markdown(
                        f"<div class='event-card'>"
                        f"<b>{ev_type}</b>"
                        f"<small> · {ev_date} · {ev_dev}"
                        + (f" · {ev_cls}" if ev_cls else "")
                        + f"</small>"
                        f"<br><i>{ev_reas}</i>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            elif not device_events:
                st.info("No similar historical events found for this query.")

            retrieval_note = hist_ev.get("retrieval_note", "")
            if retrieval_note:
                st.caption(f"ℹ️ {retrieval_note}")

            # ── SHAP EXPLANATION ──────────────────────────────────────────────
            expl = result.get("explanation")
            if expl and expl.get("available"):
                st.markdown("<div class='section-title'>🔬 SHAP Feature Contributions</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-intro'>🔴 Red bars increase predicted risk; 🔵 blue bars decrease it. "
                    "SHAP values show each feature's contribution relative to the model's average prediction.</div>",
                    unsafe_allow_html=True,
                )
                with st.spinner("Rendering SHAP chart…"):
                    fig_shap = shap_waterfall(
                        top_positive=expl.get("top_positive", []),
                        top_negative=expl.get("top_negative", []),
                        base_value=expl.get("base_value", 0.0),
                        predicted_value=expl.get("predicted_value", 0.0),
                    )
                st.plotly_chart(fig_shap, use_container_width=True)

            # ── MAINTENANCE RECOMMENDATION ────────────────────────────────────
            rec = result.get("recommendation")
            if rec and rec.get("available"):
                st.markdown("<div class='section-title'>🔧 Maintenance Recommendation</div>", unsafe_allow_html=True)
                
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
                    st.json(rec.get("rule_inputs", {}))

            # ── LIMITATIONS ───────────────────────────────────────────────────
            limitations = result.get("limitations", [])
            if limitations:
                with st.expander("⚠️ Model Limitations", expanded=False):
                    for lim in limitations:
                        st.markdown(f"- {lim}")

            # ── DISCLAIMER ────────────────────────────────────────────────────
            disclaimer = result.get("disclaimer", "")
            if disclaimer:
                st.markdown(
                    f"<div class='disclaimer'>⚕️ {disclaimer}</div>",
                    unsafe_allow_html=True,
                )

# ── Default state (no submission yet) ────────────────────────────────────────
if not submitted:
    st.divider()
    st.markdown(
        "<div class='section-title'>💡 Example Queries</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-intro'>"
        "Use these examples to explore the model's behaviour across different device types and severity levels."
        "</div>",
        unsafe_allow_html=True,
    )
    examples = [
        (
            "🫀 Cardiac Defibrillator — electrical fault",
            "Implanted cardiac defibrillator, model ICD-X200",
            "Repeated electrical faults and abnormal battery behavior. Device delivering inappropriate shocks.",
            "Expected: MEDIUM risk (~33–42 / 100). Device Class III ≠ event risk level.",
        ),
        (
            "💉 Insulin Pump — software error",
            "Insulin pump, external wearable device",
            "Device over-delivering insulin due to software error causing hypoglycemia episodes.",
            "Expected: moderate-to-high risk based on severity of problem description.",
        ),
        (
            "🔬 Glucose Monitor — incorrect readings",
            "Blood glucose monitor, fingerstick type",
            "Incorrect readings causing false low values; patient received incorrect insulin treatment.",
            "Expected: moderate risk based on patient impact described.",
        ),
        (
            "✂️ Surgical Scissors — minor labeling",
            "Surgical scissors, stainless steel reusable",
            "Minor labeling error on outer packaging only. No patient contact or safety risk identified.",
            "Expected: LOW risk — labeling-only issue, no patient safety impact.",
        ),
    ]
    for title, dev_info, prob_desc, note in examples:
        with st.expander(title):
            st.markdown(f"**Device Information:** {dev_info}")
            st.markdown(f"**Problem Description:** {prob_desc}")
            st.caption(f"💡 {note}")

