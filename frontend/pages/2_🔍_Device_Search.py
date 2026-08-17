"""
frontend/pages/2_🔍_Device_Search.py
======================================
Device Search page — paginated, filtered device list from GET /devices.

Filters: risk level, manufacturer, category, country, free-text search.
Results shown as a styled table with color-coded risk badges.
Selecting a device stores its ID in session_state for the Detail page.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils.api_client import BACKEND_URL, get_devices, get_health

st.set_page_config(
    page_title="Device Search | Medical Device Risk",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.badge-HIGH   { background:#fef2f2;color:#dc2626;padding:2px 10px;border-radius:20px;font-weight:600;font-size:0.82em; }
.badge-MEDIUM { background:#fffbeb;color:#d97706;padding:2px 10px;border-radius:20px;font-weight:600;font-size:0.82em; }
.badge-LOW    { background:#f0fdf4;color:#16a34a;padding:2px 10px;border-radius:20px;font-weight:600;font-size:0.82em; }
.badge-NA     { background:#f1f5f9;color:#64748b;padding:2px 10px;border-radius:20px;font-weight:600;font-size:0.82em; }
.disclaimer   { background:#f8fafc;border-left:4px solid #94a3b8;padding:10px 16px;
                border-radius:4px;font-size:0.78em;color:#64748b; }
table { width:100%; }
</style>
""", unsafe_allow_html=True)

DISCLAIMER = (
    "This system is a decision-support prototype and does not replace qualified "
    "maintenance, biomedical engineering, regulatory, or clinical judgment. "
    "It is not a certified medical device and does not guarantee patient safety outcomes."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Device Search")
    health = get_health()
    if health:
        st.success("🟢 Backend connected")
    else:
        st.error(f"🔴 Backend unreachable\n`{BACKEND_URL}`")
    st.divider()

    st.markdown("#### Filters")
    risk_filter  = st.selectbox("Risk Level",    ["", "HIGH", "MEDIUM", "LOW"], index=0,
                                format_func=lambda x: "All" if x == "" else x)
    mfr_filter   = st.text_input("Manufacturer",  placeholder="e.g. Medtronic")
    cat_filter   = st.text_input("Category",      placeholder="e.g. Recall")
    country_filter = st.text_input("Country",     placeholder="e.g. USA")
    search_filter  = st.text_input("Search (name / ID)", placeholder="device name or ID")
    page_size    = st.selectbox("Rows per page",  [25, 50, 100], index=1)
    st.divider()
    st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)

# ── Pagination state ──────────────────────────────────────────────────────────
if "search_page" not in st.session_state:
    st.session_state["search_page"] = 1

# Reset to page 1 when filters change
filter_key = (risk_filter, mfr_filter, cat_filter, country_filter, search_filter, page_size)
if st.session_state.get("_last_filter_key") != filter_key:
    st.session_state["search_page"] = 1
    st.session_state["_last_filter_key"] = filter_key

current_page = st.session_state["search_page"]

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔍 Device Search")
st.caption("Server-side filtering via GET /devices. Click a Device ID to open its details.")

with st.spinner("Fetching devices..."):
    resp = get_devices(
        risk_level=risk_filter or None,
        manufacturer=mfr_filter or None,
        category=cat_filter or None,
        country=country_filter or None,
        search=search_filter or None,
        page=current_page,
        page_size=page_size,
    )

if resp is None:
    st.error(f"Could not reach the backend API at `{BACKEND_URL}`.")
    st.stop()

pagination = resp.get("pagination", {})
total_items = pagination.get("total_items", 0)
total_pages = pagination.get("total_pages", 1)
items       = resp.get("items", [])

# ── Result count + pagination controls ────────────────────────────────────────
top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown(
        f"**{total_items:,} devices** found"
        + (f" (page {current_page} of {total_pages})" if total_pages > 1 else ""),
    )
with top_r:
    if total_pages > 1:
        nav_l, nav_m, nav_r = st.columns([1, 2, 1])
        if nav_l.button("◀ Prev", disabled=current_page <= 1):
            st.session_state["search_page"] = max(1, current_page - 1)
            st.rerun()
        nav_m.markdown(f"<div style='text-align:center;padding-top:8px'>{current_page}/{total_pages}</div>",
                       unsafe_allow_html=True)
        if nav_r.button("Next ▶", disabled=current_page >= total_pages):
            st.session_state["search_page"] = min(total_pages, current_page + 1)
            st.rerun()

st.divider()

# ── Device table ──────────────────────────────────────────────────────────────
if not items:
    st.info("No devices match the current filters.")
else:
    def _badge(level):
        if not level:
            return "<span class='badge-NA'>N/A</span>"
        return f"<span class='badge-{level}'>{level}</span>"

    def _clean(val):
        if val is None or str(val).lower() in ("nan", "none", ""):
            return "—"
        return str(val)

    rows_html = ""
    for item in items:
        did        = item.get("device_id", "")
        name       = _clean(item.get("device_name"))
        cls        = _clean(item.get("device_classification"))
        country    = _clean(item.get("device_country"))
        mfr        = _clean(item.get("mfr_parent_company") or item.get("mfr_name"))
        score      = item.get("risk_score")
        score_str  = f"{score:.1f}" if score is not None else "—"
        rl         = item.get("risk_level")
        badge      = _badge(rl)

        rows_html += (
            f"<tr>"
            f"<td style='padding:7px 10px;font-weight:600;color:#2563eb'>{did}</td>"
            f"<td style='padding:7px 10px'>{name}</td>"
            f"<td style='padding:7px 10px;color:#475569'>{cls}</td>"
            f"<td style='padding:7px 10px;color:#475569'>{country}</td>"
            f"<td style='padding:7px 10px;color:#475569'>{mfr}</td>"
            f"<td style='padding:7px 10px;text-align:center'>{badge}</td>"
            f"<td style='padding:7px 10px;text-align:right;font-weight:600'>{score_str}</td>"
            f"</tr>"
        )

    table_html = f"""
    <table style='border-collapse:collapse;width:100%;font-size:0.87em;'>
      <thead>
        <tr style='background:#f8fafc;border-bottom:2px solid #e2e8f0;'>
          <th style='padding:8px 10px;text-align:left;color:#64748b;font-weight:600'>Device ID</th>
          <th style='padding:8px 10px;text-align:left;color:#64748b;font-weight:600'>Name</th>
          <th style='padding:8px 10px;text-align:left;color:#64748b;font-weight:600'>Classification</th>
          <th style='padding:8px 10px;text-align:left;color:#64748b;font-weight:600'>Country</th>
          <th style='padding:8px 10px;text-align:left;color:#64748b;font-weight:600'>Manufacturer</th>
          <th style='padding:8px 10px;text-align:center;color:#64748b;font-weight:600'>Risk Level</th>
          <th style='padding:8px 10px;text-align:right;color:#64748b;font-weight:600'>Score</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.divider()

    # Quick-navigate to detail page
    st.markdown("**Open Device Detail**")
    selected_id = st.text_input(
        "Enter a Device ID from the table above:",
        key="quick_device_id",
        placeholder="e.g. 80508",
    )
    if selected_id:
        st.session_state["detail_device_id"] = selected_id.strip()
        st.page_link(
            "pages/3_📋_Device_Detail.py",
            label=f"→ Open details for device {selected_id.strip()}",
            icon="📋",
        )

st.divider()
st.markdown(f"<div class='disclaimer'>⚕️ {DISCLAIMER}</div>", unsafe_allow_html=True)
