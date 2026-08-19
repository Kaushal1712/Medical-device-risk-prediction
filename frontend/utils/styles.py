"""
frontend/utils/styles.py
========================
Shared CSS design system + Python layout helpers for every page.

Theme system
------------
Two themes are supported: "dark" (default) and "light".
The user's selection is persisted in st.session_state["md_theme_toggle"].

On every page render:
  1.  inject() reads session_state to determine current theme.
  2.  It returns COMMON_CSS (shared classes) + the appropriate theme block
      (DARK_MODE_CSS or LIGHT_MODE_CSS), which override Streamlit's own
      light-mode base colours so our chosen palette takes effect.
  3.  sidebar_base() renders the ☀️/🌙 toggle widget. Because st.radio
      automatically triggers a rerun on change, the new CSS block is picked
      up on the very next render with no extra st.rerun() call needed.

CSS variable strategy
---------------------
* Shared component classes (.card, .info-row, .badge-*, etc.) use CSS custom
  properties (--text-color, --secondary-background-color, --background-color)
  so they always match whatever the active theme sets.
* Theme blocks redefine those custom properties on `html` (our <style> tag
  loads after Streamlit's, so same-specificity wins by cascade order) AND
  override actual background/color on Streamlit's layout containers with
  !important so Streamlit's compiled rules don't leak through.
* Semantic colours (badge red/amber/green, accent #2563eb) are intentionally
  hardcoded — they are designed to be legible in both themes.

Helper functions
-----------------
inject()        — returns full <style> block, theme-aware
page_header()   — renders consistent page hero header
sidebar_base()  — renders sidebar branding + backend status + theme toggle
get_theme()     — returns "dark" | "light" for use in charts.py etc.
"""

import streamlit as st
from utils.api_client import BACKEND_URL, get_health

# ── DISCLAIMER constant (single source of truth) ──────────────────────────────
DISCLAIMER = (
    "This system is a decision-support prototype and does not replace "
    "qualified maintenance, biomedical engineering, regulatory, or clinical "
    "judgment. It is not a certified medical device and does not guarantee "
    "patient safety outcomes."
)

# ── Theme session-state key ────────────────────────────────────────────────────
_THEME_KEY = "md_theme_toggle"


def get_theme() -> str:
    """Return 'dark' or 'light' based on current session state. Defaults to 'dark'."""
    val = st.session_state.get(_THEME_KEY, "🌙 Dark")
    return "light" if "Light" in val else "dark"


# ══════════════════════════════════════════════════════════════════════════════
#  _COMMON_CSS_RULES — shared classes used by all pages in both themes
#  (All colours reference CSS custom properties set by the theme blocks below)
# ══════════════════════════════════════════════════════════════════════════════
_COMMON_CSS_RULES = """
/* ── 1. TYPOGRAPHY ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }

/* ── 2. PAGE HEADER HERO ───────────────────────────────────────────────── */
.page-header {
    padding: 18px 0 12px 0;
    margin-bottom: 4px;
    border-bottom: 2px solid rgba(37, 99, 235, 0.25);
}
.page-header .ph-eyebrow {
    font-size: 0.72em;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #2563eb;
    margin-bottom: 4px;
}
.page-header h1 {
    font-size: 1.65em !important;
    font-weight: 700 !important;
    color: var(--text-color) !important;
    margin: 0 0 6px 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
    border: none !important;
}
.page-header .ph-subtitle {
    font-size: 0.88em;
    color: var(--text-color);
    opacity: 0.6;
    line-height: 1.5;
    margin: 0;
}

/* ── 3. SIDEBAR BRAND BLOCK ────────────────────────────────────────────── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 8px 0;
    margin-bottom: 2px;
}
.sidebar-brand .sb-icon  { font-size: 1.5em; line-height: 1; }
.sidebar-brand .sb-text  { display: flex; flex-direction: column; gap: 1px; }
.sidebar-brand .sb-title { font-size: 0.88em; font-weight: 700; color: var(--text-color); line-height: 1.2; }
.sidebar-brand .sb-sub   { font-size: 0.68em; color: var(--text-color); opacity: 0.5; font-weight: 400; }

/* Theme toggle strip */
.theme-toggle-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 4px 0 8px 0;
    padding: 5px 8px;
    background: rgba(148, 163, 184, 0.08);
    border-radius: 8px;
    border: 1px solid rgba(148, 163, 184, 0.18);
}
.theme-toggle-label {
    font-size: 0.7em;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-color);
    opacity: 0.45;
    white-space: nowrap;
}

/* Active page label */
.sidebar-page-label {
    font-size: 0.76em;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #2563eb;
    padding: 4px 0 2px 0;
    display: block;
}

/* Backend status pill */
.backend-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82em;
    font-weight: 500;
    color: var(--text-color);
    padding: 4px 0;
}
.backend-status .bs-dot          { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.backend-status .bs-dot.online   { background: #16a34a; }
.backend-status .bs-dot.offline  { background: #dc2626; }
.backend-meta {
    font-size: 0.74em;
    color: var(--text-color);
    opacity: 0.5;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    padding: 1px 0;
    word-break: break-all;
}

/* ── 4. STREAMLIT NATIVE COMPONENT POLISH ──────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--secondary-background-color) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    transition: box-shadow 0.15s ease !important;
}
[data-testid="metric-container"]:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important; }
[data-testid="stMetricLabel"]  { font-size: 0.73em !important; font-weight: 600 !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; opacity: 0.55 !important; }
[data-testid="stMetricValue"]  { font-size: 1.6em !important; font-weight: 700 !important; }

[data-testid="stButton"] > button[kind="primary"] {
    border-radius: 8px !important; font-weight: 600 !important; font-size: 0.95em !important;
    letter-spacing: 0.02em !important; padding: 10px 24px !important;
    transition: opacity 0.15s ease, transform 0.1s ease !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
[data-testid="stButton"] > button[kind="primary"]:active { transform: translateY(0) !important; }
[data-testid="stButton"] > button[kind="secondary"] { border-radius: 8px !important; font-weight: 500 !important; }

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    border-radius: 8px !important; font-size: 0.92em !important;
    border: 1px solid rgba(148, 163, 184, 0.4) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
}

[data-testid="stSelectbox"] > div > div { border-radius: 8px !important; border: 1px solid rgba(148, 163, 184, 0.4) !important; }
[data-testid="stExpander"]  { border: 1px solid rgba(148, 163, 184, 0.25) !important; border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stExpander"] summary { font-weight: 500 !important; font-size: 0.92em !important; }
[data-testid="stSidebar"]  { border-right: 1px solid rgba(148, 163, 184, 0.2) !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 1.2rem !important; }

[data-testid="stForm"] [data-testid="stButton"] > button[kind="primary"] {
    padding: 14px 24px !important; font-size: 1em !important; border-radius: 10px !important;
}

/* ── 5. RISK-LEVEL BADGES ──────────────────────────────────────────────── */
.badge-HIGH   { display:inline-block; background:rgba(220,38,38,0.12); color:#dc2626; padding:3px 12px; border-radius:20px; font-weight:700; font-size:0.82em; letter-spacing:0.03em; border:1px solid rgba(220,38,38,0.3); }
.badge-MEDIUM { display:inline-block; background:rgba(217,119,6,0.12); color:#d97706; padding:3px 12px; border-radius:20px; font-weight:700; font-size:0.82em; letter-spacing:0.03em; border:1px solid rgba(217,119,6,0.3); }
.badge-LOW    { display:inline-block; background:rgba(22,163,74,0.12); color:#16a34a; padding:3px 12px; border-radius:20px; font-weight:700; font-size:0.82em; letter-spacing:0.03em; border:1px solid rgba(22,163,74,0.3); }
.badge-NA     { display:inline-block; background:rgba(100,116,139,0.1); color:var(--text-color); opacity:0.6; padding:3px 12px; border-radius:20px; font-weight:600; font-size:0.82em; border:1px solid rgba(100,116,139,0.2); }

/* ── 6. PRIORITY BADGES ────────────────────────────────────────────────── */
.prio-Critical { display:inline-block; background:rgba(185,28,28,0.12); color:#b91c1c; padding:4px 14px; border-radius:20px; font-weight:700; border:1px solid rgba(185,28,28,0.3); }
.prio-High     { display:inline-block; background:rgba(194,65,12,0.12); color:#c2410c; padding:4px 14px; border-radius:20px; font-weight:700; border:1px solid rgba(194,65,12,0.3); }
.prio-Medium   { display:inline-block; background:rgba(180,83,9,0.12); color:#b45309; padding:4px 14px; border-radius:20px; font-weight:700; border:1px solid rgba(180,83,9,0.3); }
.prio-Low      { display:inline-block; background:rgba(21,128,61,0.12); color:#15803d; padding:4px 14px; border-radius:20px; font-weight:700; border:1px solid rgba(21,128,61,0.3); }

/* ── 7. INFO-ROW ───────────────────────────────────────────────────────── */
.info-row { display:flex; justify-content:space-between; align-items:baseline; padding:8px 0; border-bottom:1px solid rgba(148,163,184,0.15); font-size:0.9em; gap:12px; }
.info-row:last-child { border-bottom:none; padding-bottom:2px; }
.info-label { color:var(--text-color); opacity:0.55; font-weight:500; flex-shrink:0; min-width:110px; }
.info-value { color:var(--text-color); font-weight:500; text-align:right; word-break:break-word; }

/* ── 8. GENERIC CARDS ──────────────────────────────────────────────────── */
.card        { background:var(--secondary-background-color); border:1px solid rgba(148,163,184,0.22); border-radius:12px; padding:18px 22px; margin-bottom:14px; }
.card-accent { background:var(--secondary-background-color); border:1px solid rgba(148,163,184,0.22); border-left:3px solid #2563eb; border-radius:12px; padding:18px 22px; margin-bottom:14px; }
.card .info-label  { color:var(--text-color); opacity:0.55; }
.card .info-value  { color:var(--text-color); }

/* ── 9. SECTION TITLES ─────────────────────────────────────────────────── */
.section-title {
    font-size: 1.0em; font-weight: 700; color: var(--text-color);
    margin: 22px 0 10px 0; padding: 6px 0 6px 12px;
    border-left: 3px solid #2563eb; letter-spacing: -0.01em;
}
.section-intro {
    font-size: 0.84em; color: var(--text-color); opacity: 0.55;
    margin: -6px 0 12px 0; line-height: 1.5;
}

/* ── 10. STAT ROWS (Overview score stats) ──────────────────────────────── */
.stat-row        { display:flex; justify-content:space-between; align-items:center; padding:9px 0; border-bottom:1px solid rgba(148,163,184,0.15); }
.stat-row:last-child { border-bottom:none; }
.stat-row-label  { font-size:0.8em; color:var(--text-color); opacity:0.55; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }
.stat-row-value  { font-size:1.1em; font-weight:700; color:var(--text-color); }
.stat-row-value.val-high { color:#dc2626; }
.stat-row-value.val-low  { color:#16a34a; }
.stat-label { font-size:0.76em; color:var(--text-color); opacity:0.55; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }
.stat-value { font-size:1.55em; font-weight:700; color:var(--text-color); }

/* ── 11. STATUS / NOTICE BOXES ─────────────────────────────────────────── */
.unavailable { background:rgba(245,158,11,0.08); border-left:4px solid #f59e0b; padding:12px 16px; border-radius:0 8px 8px 0; color:var(--text-color); font-size:0.9em; line-height:1.5; }
.unavailable strong { color:var(--text-color); font-weight:700; }
.info-note { background:rgba(37,99,235,0.08); border-left:4px solid #2563eb; padding:12px 16px; border-radius:0 8px 8px 0; color:var(--text-color); font-size:0.88em; line-height:1.55; margin:8px 0 16px 0; }
.info-note b { font-weight:700; color:var(--text-color); }
.disclaimer { background:rgba(148,163,184,0.08); border-left:4px solid rgba(148,163,184,0.5); padding:10px 16px; border-radius:0 6px 6px 0; font-size:0.78em; color:var(--text-color); opacity:0.72; line-height:1.55; margin-top:8px; }

/* ── 12. DEVICE SEARCH — TABLE (theme-safe via CSS classes) ────────────── */
.table-wrap { background:var(--secondary-background-color); border:1px solid rgba(148,163,184,0.22); border-radius:12px; overflow:hidden; margin-bottom:16px; }
.ds-table { border-collapse:collapse; width:100%; font-size:0.88em; }
.ds-table thead tr { background:rgba(148,163,184,0.1); border-bottom:1px solid rgba(148,163,184,0.25); }
.th-cell { padding:10px 12px; text-align:left; color:var(--text-color); opacity:0.55; font-weight:600; font-size:0.78em; text-transform:uppercase; letter-spacing:0.06em; white-space:nowrap; }
.th-cell.right  { text-align:right; }
.th-cell.center { text-align:center; }
.ds-table tbody tr { border-bottom:1px solid rgba(148,163,184,0.12); transition:background 0.12s ease; }
.ds-table tbody tr:last-child { border-bottom:none; }
.ds-table tbody tr:hover { background:rgba(37,99,235,0.04); }
.td-cell { padding:9px 12px; color:var(--text-color); vertical-align:middle; }
.td-cell.muted  { opacity:0.65; font-size:0.87em; }
.td-cell.center { text-align:center; }
.td-cell.right  { text-align:right; font-weight:600; }
.device-link { color:#2563eb; font-weight:600; text-decoration:none; font-family:'JetBrains Mono','Fira Code','Courier New',monospace; font-size:0.92em; }
.device-link:hover { text-decoration:underline; }

/* ── 13. CTA BOX ───────────────────────────────────────────────────────── */
.cta-box       { background:rgba(37,99,235,0.06); border:1px solid rgba(37,99,235,0.2); border-radius:10px; padding:14px 18px; margin-top:10px; }
.cta-box .cta-title { font-size:0.85em; font-weight:600; color:#2563eb; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.06em; }

/* ── 14. HISTORICAL EVENT CARDS ────────────────────────────────────────── */
.event-card   { background:var(--secondary-background-color); border:1px solid rgba(148,163,184,0.22); border-radius:8px; padding:12px 16px; margin-bottom:8px; font-size:0.88em; line-height:1.55; color:var(--text-color) !important; }
.event-card b { color:var(--text-color) !important; font-weight:600; }
.event-card i { color:var(--text-color) !important; opacity:0.72; font-style:italic; }
.event-card small { color:var(--text-color) !important; opacity:0.5; font-size:0.85em; }

/* ── 15. RISK RESULT CARDS ─────────────────────────────────────────────── */
.result-card       { background:var(--secondary-background-color); border:1px solid rgba(148,163,184,0.22); border-radius:12px; padding:22px 24px; margin-bottom:14px; text-align:center; }
.result-card .label { font-size:0.72em; font-weight:600; color:var(--text-color); opacity:0.5; text-transform:uppercase; letter-spacing:0.09em; margin-bottom:10px; }
.result-card .value-HIGH   { font-size:2em; font-weight:700; color:#dc2626; }
.result-card .value-MEDIUM { font-size:2em; font-weight:700; color:#d97706; }
.result-card .value-LOW    { font-size:2em; font-weight:700; color:#16a34a; }
.result-card .value-num    { font-size:1.8em; font-weight:700; color:var(--text-color); }
.result-card .value-sub    { font-size:0.45em; font-weight:400; color:var(--text-color); opacity:0.45; vertical-align:middle; }

/* ── 16. FORM FIELD GROUPS ─────────────────────────────────────────────── */
.field-group-label { font-size:0.72em; font-weight:700; letter-spacing:0.09em; text-transform:uppercase; margin-bottom:8px; }

/* ── 17. LANDING PAGE NAV CARDS ────────────────────────────────────────── */
.nav-card { background:var(--secondary-background-color); border:1px solid rgba(148,163,184,0.22); border-radius:12px; padding:16px 14px 14px 14px; text-align:center; transition:border-color 0.15s ease,box-shadow 0.15s ease,transform 0.12s ease; }
.nav-card:hover { border-color:rgba(37,99,235,0.4); box-shadow:0 4px 16px rgba(37,99,235,0.08); transform:translateY(-2px); }
.nav-card .nc-icon  { font-size:1.8em; margin-bottom:8px; display:block; }
.nav-card .nc-label { font-size:0.82em; font-weight:600; color:var(--text-color); line-height:1.3; }
.nav-card .nc-desc  { font-size:0.72em; color:var(--text-color); opacity:0.5; margin-top:4px; line-height:1.4; }

/* ── 18. MISC ──────────────────────────────────────────────────────────── */
.block-container { padding-top:1.5rem !important; padding-bottom:2rem !important; }
/* Hide Streamlit's default <h1> when we use page_header() */
h1[data-testid="stHeading"] { display:none !important; }
"""


# ══════════════════════════════════════════════════════════════════════════════
#  DARK MODE — force dark slate over Streamlit's light base
# ══════════════════════════════════════════════════════════════════════════════
_DARK_MODE_CSS = """
/* ════════════════════════════════════════════════════════
   DARK MODE THEME — slate-900/800 background, slate-200 text
   ════════════════════════════════════════════════════════ */

/* 1. Redefine Streamlit's CSS custom properties
   Our <style> tag loads after Streamlit's so same-specificity wins */
html {
    --background-color: #0f172a !important;
    --secondary-background-color: #1e293b !important;
    --text-color: #e2e8f0 !important;
    --primary-background-color: #0f172a !important;
}

/* 2. App-level backgrounds */
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
}
section[data-testid="stMain"],
section[data-testid="stMain"] > div {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
}
.block-container { background-color: transparent !important; }

/* 3. Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] > div {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
}

/* 4. Header bar */
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background-color: #0f172a !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12) !important;
}

/* 5. Headings */
h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }

/* 6. Markdown / caption text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li { color: #e2e8f0 !important; }
[data-testid="stCaptionContainer"] p   { color: #94a3b8 !important; }

/* 7. Form inputs */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border-color: rgba(148, 163, 184, 0.3) !important;
    caret-color: #60a5fa !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder { color: #64748b !important; }

/* 8. Select / dropdown */
div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border-color: rgba(148, 163, 184, 0.3) !important;
}
/* Dropdown menu popover */
div[data-baseweb="popover"] > div { background-color: #1e293b !important; }
li[role="option"] {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] { background-color: #334155 !important; }
/* Selected value text */
div[data-baseweb="select"] span { color: #e2e8f0 !important; }

/* 9. Radio / checkbox labels */
[data-testid="stRadio"] label p,
[data-testid="stRadio"] p { color: #e2e8f0 !important; }
[data-testid="stCheckbox"] label p { color: #e2e8f0 !important; }

/* 10. Slider */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p { color: #e2e8f0 !important; }
[data-testid="stSlider"] div[data-baseweb="slider"] div { background-color: #334155 !important; }

/* 11. Metric containers */
[data-testid="metric-container"]     { background: #1e293b !important; border-color: rgba(148,163,184,0.18) !important; }
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] label  { color: #94a3b8 !important; }
[data-testid="stMetricValue"]        { color: #f1f5f9 !important; }
[data-testid="stMetricDelta"] p      { color: #94a3b8 !important; }
[data-testid="stMetricDeltaIcon"]    { filter: none !important; }

/* 12. Expanders */
[data-testid="stExpander"] { background-color: transparent !important; border-color: rgba(148,163,184,0.2) !important; }
[data-testid="stExpander"] summary p { color: #e2e8f0 !important; }
[data-testid="stExpanderDetails"]    { background-color: transparent !important; }

/* 13. Dataframe / st.dataframe */
[data-testid="stDataFrame"]  { background-color: #1e293b !important; }
.dvn-scroller                { background-color: #1e293b !important; }
[data-testid="stDataFrame"] th { color: #94a3b8 !important; background-color: #1e293b !important; }
[data-testid="stDataFrame"] td { color: #e2e8f0 !important; background-color: #0f172a !important; }

/* 14. Form container border */
[data-testid="stForm"] { border-color: rgba(148,163,184,0.18) !important; }

/* 15. Divider */
hr { border-color: rgba(148, 163, 184, 0.15) !important; }

/* 16. Alerts (stSuccess, stError, stWarning, stInfo) */
div[data-testid="stAlert"]      { background-color: #1e293b !important; }
div[data-testid="stInfo"]       { background-color: rgba(37,99,235,0.12) !important; color:#93c5fd !important; }
div[data-testid="stSuccess"]    { background-color: rgba(22,163,74,0.12) !important; color:#86efac !important; }
div[data-testid="stWarning"]    { background-color: rgba(245,158,11,0.12) !important; color:#fcd34d !important; }
div[data-testid="stError"]      { background-color: rgba(220,38,38,0.12) !important; color:#fca5a5 !important; }
[data-testid="stNotification"]  { background-color: #1e293b !important; color: #e2e8f0 !important; }

/* 17. Code blocks */
code { background-color: #1e293b !important; color: #93c5fd !important; border-radius: 4px; padding: 1px 5px; }
pre  { background-color: #1e293b !important; color: #e2e8f0 !important; }

/* 18. Sidebar nav links (Streamlit page navigation) */
[data-testid="stSidebarNavLink"]       { color: #cbd5e1 !important; }
[data-testid="stSidebarNavLink"]:hover { background-color: rgba(148,163,184,0.1) !important; color: #f1f5f9 !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background-color: rgba(37,99,235,0.15) !important; color: #60a5fa !important; }

/* 19. Spinner text */
[data-testid="stStatusWidget"] { color: #94a3b8 !important; }

/* 20. Input label text */
label, [data-testid="stWidgetLabel"] p { color: #e2e8f0 !important; }

/* 21. Selectbox label within dark sidebar */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #e2e8f0 !important; }

/* 22. Plotly chart SVG text (axes, titles) */
.js-plotly-plot .plotly .gtitle         { fill: #cbd5e1 !important; }
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text     { fill: #94a3b8 !important; }
.js-plotly-plot .plotly .xaxislayer-above text,
.js-plotly-plot .plotly .yaxislayer-above text { fill: #94a3b8 !important; }
.js-plotly-plot .plotly .legend text    { fill: #cbd5e1 !important; }
.js-plotly-plot .plotly .xaxis .title,
.js-plotly-plot .plotly .yaxis .title   { fill: #94a3b8 !important; }
"""


# ══════════════════════════════════════════════════════════════════════════════
#  LIGHT MODE — professional slate-based (not Streamlit default grey)
# ══════════════════════════════════════════════════════════════════════════════
_LIGHT_MODE_CSS = """
/* ════════════════════════════════════════════════════════
   LIGHT MODE THEME — slate-50 background, slate-900 text
   ════════════════════════════════════════════════════════ */

html {
    --background-color: #f8fafc !important;
    --secondary-background-color: #ffffff !important;
    --text-color: #0f172a !important;
    --primary-background-color: #f8fafc !important;
}

.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background-color: #f8fafc !important;
    color: #0f172a !important;
}
section[data-testid="stMain"],
section[data-testid="stMain"] > div {
    background-color: #f8fafc !important;
    color: #0f172a !important;
}
.block-container { background-color: transparent !important; }

/* Sidebar — slightly tinted for contrast */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] > div {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background-color: #f8fafc !important;
    border-bottom: 1px solid rgba(148,163,184,0.25) !important;
}

h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li { color: #0f172a !important; }
[data-testid="stCaptionContainer"] p   { color: #64748b !important; }

/* Form inputs — white background, dark text */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-color: rgba(148,163,184,0.4) !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder { color: #94a3b8 !important; }

/* Select */
div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #0f172a !important; border-color: rgba(148,163,184,0.4) !important; }
div[data-baseweb="popover"] > div { background-color: #ffffff !important; }
li[role="option"] { background-color: #ffffff !important; color: #0f172a !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] { background-color: #f1f5f9 !important; }
div[data-baseweb="select"] span { color: #0f172a !important; }

/* Radio / checkbox */
[data-testid="stRadio"] label p,
[data-testid="stRadio"] p { color: #0f172a !important; }
[data-testid="stCheckbox"] label p { color: #0f172a !important; }

/* Slider */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p { color: #0f172a !important; }

/* Metric containers — white cards */
[data-testid="metric-container"]    { background: #ffffff !important; border-color: rgba(148,163,184,0.2) !important; }
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] label { color: #64748b !important; }
[data-testid="stMetricValue"]       { color: #0f172a !important; }
[data-testid="stMetricDelta"] p     { color: #64748b !important; }

/* Expanders */
[data-testid="stExpander"] { background-color: #ffffff !important; border-color: rgba(148,163,184,0.22) !important; }
[data-testid="stExpander"] summary p { color: #0f172a !important; }

/* Dataframe */
[data-testid="stDataFrame"]  { background-color: #ffffff !important; }
[data-testid="stDataFrame"] th { color: #64748b !important; background-color: #f8fafc !important; }
[data-testid="stDataFrame"] td { color: #0f172a !important; background-color: #ffffff !important; }

[data-testid="stForm"] { border-color: rgba(148,163,184,0.2) !important; }
hr { border-color: rgba(148,163,184,0.25) !important; }

/* Alerts in light mode */
div[data-testid="stInfo"]    { background-color: rgba(37,99,235,0.07) !important; color: #1e40af !important; }
div[data-testid="stSuccess"] { background-color: rgba(22,163,74,0.07) !important; color: #15803d !important; }
div[data-testid="stWarning"] { background-color: rgba(245,158,11,0.07) !important; color: #b45309 !important; }
div[data-testid="stError"]   { background-color: rgba(220,38,38,0.07) !important; color: #dc2626 !important; }

code { background-color: #f1f5f9 !important; color: #2563eb !important; }
pre  { background-color: #f1f5f9 !important; color: #0f172a !important; }

/* Sidebar nav links */
[data-testid="stSidebarNavLink"]       { color: #374151 !important; }
[data-testid="stSidebarNavLink"]:hover { background-color: rgba(148,163,184,0.12) !important; color: #0f172a !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background-color: rgba(37,99,235,0.08) !important; color: #2563eb !important; }

label, [data-testid="stWidgetLabel"] p { color: #0f172a !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #0f172a !important; }
[data-testid="stStatusWidget"] { color: #64748b !important; }

/* Plotly chart SVG text */
.js-plotly-plot .plotly .gtitle         { fill: #0f172a !important; }
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text     { fill: #64748b !important; }
.js-plotly-plot .plotly .xaxislayer-above text,
.js-plotly-plot .plotly .yaxislayer-above text { fill: #64748b !important; }
.js-plotly-plot .plotly .legend text    { fill: #0f172a !important; }
"""


def inject(extra_css: str = "") -> str:
    """
    Return the full <style> block for injection via st.markdown(..., unsafe_allow_html=True).

    Reads the current theme from st.session_state["md_theme_toggle"] and
    appends the appropriate dark/light override block after the shared CSS rules.

    Parameters
    ----------
    extra_css : str, optional
        Additional CSS rules to append for page-specific overrides.

    Returns
    -------
    str — HTML <style>…</style> block ready for st.markdown().
    """
    theme = get_theme()
    theme_css = _DARK_MODE_CSS if theme == "dark" else _LIGHT_MODE_CSS
    all_rules = _COMMON_CSS_RULES + "\n" + theme_css + "\n" + (extra_css or "")
    return f"<style>\n{all_rules}\n</style>"


def page_header(
    icon: str,
    title: str,
    subtitle: str,
    eyebrow: str = "Medical Device Risk Dashboard",
) -> None:
    """
    Render a consistent page hero header using st.markdown().

    Parameters
    ----------
    icon     : emoji icon shown before the title
    title    : main page title text (no emoji)
    subtitle : descriptive subtitle / caption line
    eyebrow  : small label above the title (breadcrumb / app name)
    """
    st.markdown(
        f"""<div class='page-header'>
  <div class='ph-eyebrow'>{eyebrow}</div>
  <h1>{icon}&nbsp; {title}</h1>
  <p class='ph-subtitle'>{subtitle}</p>
</div>""",
        unsafe_allow_html=True,
    )


def sidebar_base(active_page: str, show_model: bool = False, show_controls: bool = False) -> None:
    """
    Render consistent sidebar branding + theme toggle + backend status.

    The theme toggle (☀️ Light / 🌙 Dark) is a st.radio widget with
    key="md_theme_toggle". Changing it automatically triggers a Streamlit
    rerun so inject() picks up the new theme on the very next render.

    Parameters
    ----------
    active_page   : label for the currently active page (shown as accent text)
    show_model    : if True, show model version + manifest hash from health check
    show_controls : unused — kept for API compatibility
    """
    # ── Branding block ─────────────────────────────────────────────────────────
    st.markdown(
        """<div class='sidebar-brand'>
  <span class='sb-icon'>🏥</span>
  <div class='sb-text'>
    <span class='sb-title'>Medical Device<br>Risk Dashboard</span>
    <span class='sb-sub'>Cognizant NPN · Healthcare Track</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Theme toggle ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='theme-toggle-label'>Appearance</div>",
        unsafe_allow_html=True,
    )
    # Initialize default to dark if not set
    if _THEME_KEY not in st.session_state:
        st.session_state[_THEME_KEY] = "🌙 Dark"

    st.radio(
        "Theme",
        options=["☀️ Light", "🌙 Dark"],
        index=0 if st.session_state[_THEME_KEY] == "☀️ Light" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key=_THEME_KEY,
    )

    # ── Active page label ──────────────────────────────────────────────────────
    st.markdown(
        f"<span class='sidebar-page-label'>📍 {active_page}</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Backend status ─────────────────────────────────────────────────────────
    health = get_health()
    if health:
        st.markdown(
            "<div class='backend-status'>"
            "<span class='bs-dot online'></span>"
            "<span>Backend connected</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        mv = health.get("model_version", "unknown")
        st.markdown(f"<div class='backend-meta'>Model: {mv}</div>", unsafe_allow_html=True)
        if show_model:
            mh = health.get("data_manifest_hash", "unknown")
            st.markdown(f"<div class='backend-meta'>Manifest: {mh[:12]}…</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='backend-status'>"
            "<span class='bs-dot offline'></span>"
            "<span>Backend unreachable</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='backend-meta'>{BACKEND_URL}</div>", unsafe_allow_html=True)
        st.info("Start: `uvicorn backend.main:app --reload`")

    st.divider()
    st.markdown(
        f"<div class='disclaimer'>{DISCLAIMER}</div>",
        unsafe_allow_html=True,
    )
