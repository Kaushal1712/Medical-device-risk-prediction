"""
frontend/utils/charts.py
==========================
Reusable Plotly chart builders for the Medical Device Risk dashboard.

All functions return a go.Figure ready for st.plotly_chart().

Color conventions (used consistently across all pages):
  HIGH   → #dc2626  (red-600)
  MEDIUM → #d97706  (amber-600)
  LOW    → #16a34a  (green-600)
  Positive SHAP → #ef4444 (risk-increasing)
  Negative SHAP → #3b82f6 (risk-reducing)
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ── Risk-level color palette ─────────────────────────────────────────────────
RISK_COLOR = {
    "HIGH":   "#dc2626",
    "MEDIUM": "#d97706",
    "LOW":    "#16a34a",
}
RISK_BG = {
    "HIGH":   "#fef2f2",
    "MEDIUM": "#fffbeb",
    "LOW":    "#f0fdf4",
}


def _layout_defaults() -> dict:
    """
    Return Plotly layout kwargs adapted to the current dashboard theme.

    Reads st.session_state["md_theme_toggle"] (set by sidebar_base() in
    styles.py) to decide whether to use dark or light chart text colours.
    Falls back to dark if session state is not yet initialised.
    """
    theme_val = st.session_state.get("md_theme_toggle", "🌙 Dark")
    is_dark = "Light" not in theme_val
    text_color = "#cbd5e1" if is_dark else "#1e293b"
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=text_color),
        margin=dict(l=10, r=10, t=40, b=10),
    )


# ── Risk distribution bar chart ──────────────────────────────────────────────

def risk_distribution_bar(risk_levels: dict) -> go.Figure:
    """
    Horizontal grouped bar chart — HIGH / MEDIUM / LOW count and percent.

    Parameters
    ----------
    risk_levels : dict
        From GET /risk-summary → risk_levels.
        Keys: "HIGH", "MEDIUM", "LOW".
        Each value: {"count": int, "percent": float}.
    """
    levels = ["HIGH", "MEDIUM", "LOW"]
    counts  = [risk_levels.get(l, {}).get("count", 0)   for l in levels]
    percents = [risk_levels.get(l, {}).get("percent", 0) for l in levels]
    colors  = [RISK_COLOR[l] for l in levels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts,
        y=levels,
        orientation="h",
        marker_color=colors,
        text=[f"{c:,}  ({p:.1f}%)" for c, p in zip(counts, percents)],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        title="Devices by Risk Level",
        xaxis_title="Device Count",
        yaxis=dict(autorange="reversed"),
        height=220,
        **_layout_defaults(),
    )
    return fig


# ── Category / manufacturer stacked breakdown ─────────────────────────────────

def breakdown_stacked_bar(items: list[dict], label_key: str, title: str) -> go.Figure:
    """
    Horizontal stacked bar — HIGH / MEDIUM / LOW per category or manufacturer.

    Parameters
    ----------
    items : list[dict]
        From GET /risk-summary → category_breakdown or manufacturer_breakdown.
    label_key : str
        "category" or "manufacturer".
    title : str
        Chart title.
    """
    labels = [item[label_key] for item in items]
    high   = [item["high"]   for item in items]
    medium = [item["medium"] for item in items]
    low    = [item["low"]    for item in items]

    fig = go.Figure()
    for level, values, color in [
        ("HIGH",   high,   RISK_COLOR["HIGH"]),
        ("MEDIUM", medium, RISK_COLOR["MEDIUM"]),
        ("LOW",    low,    RISK_COLOR["LOW"]),
    ]:
        fig.add_trace(go.Bar(
            name=level,
            x=values,
            y=labels,
            orientation="h",
            marker_color=color,
            hovertemplate=f"<b>%{{y}}</b><br>{level}: %{{x:,}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Device Count",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=max(300, len(items) * 28 + 80),
        **_layout_defaults(),
    )
    return fig


# ── SHAP waterfall (local explanation) ───────────────────────────────────────

def shap_waterfall(
    top_positive: list[dict],
    top_negative: list[dict],
    base_value: float,
    predicted_value: float,
) -> go.Figure:
    """
    Horizontal bar chart showing top positive and negative SHAP contributions.

    Parameters
    ----------
    top_positive : list of FeatureContributionItem dicts (shap_value >= 0)
    top_negative : list of FeatureContributionItem dicts (shap_value < 0)
    base_value   : SHAP expected value (mean prediction)
    predicted_value : base_value + sum(all shap values)
    """
    all_contributions = top_positive + top_negative
    if not all_contributions:
        fig = go.Figure()
        fig.add_annotation(text="No SHAP contributions available.", showarrow=False,
                           font=dict(size=14, color="#6b7280"))
        fig.update_layout(height=200, **_layout_defaults())
        return fig

    features   = [c["feature"]    for c in all_contributions]
    shap_vals  = [c["shap_value"] for c in all_contributions]
    colors     = ["#ef4444" if v >= 0 else "#3b82f6" for v in shap_vals]
    directions = ["▲ increases risk" if v >= 0 else "▼ reduces risk" for v in shap_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shap_vals,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in shap_vals],
        textposition="auto",
        customdata=directions,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "SHAP: %{x:+.4f}<br>"
            "%{customdata}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_width=1, line_color="#9ca3af")
    fig.update_layout(
        title=(
            f"SHAP Contributions  |  "
            f"Base: {base_value:.4f}  →  Predicted: {predicted_value:.4f}"
        ),
        xaxis_title="SHAP Value (contribution to risk)",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(all_contributions) * 36 + 100),
        **_layout_defaults(),
    )
    return fig


# ── Global feature importance ─────────────────────────────────────────────────

def global_importance_bar(features: list[dict], top_n: int = 20) -> go.Figure:
    """
    Horizontal bar chart — top N features by importance (descending).

    Parameters
    ----------
    features : list of FeatureImportanceItem dicts {"feature", "importance", "rank"}
    top_n    : how many features to show (default 20)
    """
    subset = features[:top_n]
    names  = [f["feature"]    for f in subset]
    values = [f["importance"] for f in subset]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=[[0, "#bfdbfe"], [1, "#1d4ed8"]],  # light → dark blue
            showscale=False,
        ),
        text=[f"{v:.4f}" for v in values],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {len(subset)} Features by Global Importance",
        xaxis_title="Feature Importance",
        yaxis=dict(autorange="reversed"),
        height=max(400, len(subset) * 30 + 80),
        **_layout_defaults(),
    )
    return fig


# ── Risk score gauge (mini) ───────────────────────────────────────────────────

def risk_score_gauge(score: float, risk_level: str) -> go.Figure:
    """
    Circular gauge showing the 0–100 risk score.

    Parameters
    ----------
    score      : float in [0, 100]
    risk_level : "HIGH" | "MEDIUM" | "LOW"
    """
    theme_val = st.session_state.get("md_theme_toggle", "🌙 Dark")
    is_dark = "Light" not in theme_val
    # Gauge background adapts to theme
    gauge_bg      = "rgba(30, 41, 59, 0.6)"  if is_dark else "rgba(241, 245, 249, 0.8)"
    step_low_bg   = "rgba(22, 163, 74, 0.08)"  if is_dark else "rgba(240, 253, 244, 0.6)"
    step_med_bg   = "rgba(217, 119, 6, 0.08)"  if is_dark else "rgba(255, 251, 235, 0.6)"
    step_high_bg  = "rgba(220, 38, 38, 0.08)"  if is_dark else "rgba(254, 242, 242, 0.6)"
    tick_color    = "#64748b" if is_dark else "#94a3b8"
    text_color    = "#cbd5e1" if is_dark else "#1e293b"

    color = RISK_COLOR.get(risk_level, "#6b7280")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Risk Score<br><span style='font-size:0.8em;color:{color}'>{risk_level}</span>",
               "font": {"color": text_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": tick_color,
                     "tickfont": {"color": tick_color}},
            "bar": {"color": color},
            "bgcolor": gauge_bg,
            "steps": [
                {"range": [0,  33],  "color": step_low_bg},
                {"range": [33, 66],  "color": step_med_bg},
                {"range": [66, 100], "color": step_high_bg},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"suffix": "/100", "font": {"color": color, "size": 36}},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=text_color))
    return fig
