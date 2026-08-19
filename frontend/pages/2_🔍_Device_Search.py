"""
frontend/pages/2_🔍_Device_Search.py
======================================
Device Search page — paginated, filtered device list from GET /devices.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from utils.api_client import BACKEND_URL, get_devices, get_health
from utils.styles import DISCLAIMER, inject, page_header, sidebar_base

st.set_page_config(
    page_title="Device Search | Medical Device Risk",
    page_icon="🔍",
    layout="wide",
)

st.markdown(inject(), unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sidebar_base(active_page="Device Search")

    st.markdown("<span style='font-size:0.78em;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-color);opacity:0.55;'>Filters</span>", unsafe_allow_html=True)
    risk_filter    = st.selectbox("Risk Level",    ["", "HIGH", "MEDIUM", "LOW"], index=0,
                                  format_func=lambda x: "All" if x == "" else x)
    mfr_filter     = st.text_input("Manufacturer",  placeholder="e.g. Medtronic")
    cat_filter     = st.text_input("Category",      placeholder="e.g. Cardiovascular")
    country_filter = st.text_input("Country",       placeholder="e.g. USA")
    search_filter  = st.text_input("Search (name / ID)", placeholder="device name or ID")
    page_size      = st.selectbox("Rows per page",  [25, 50, 100], index=1)

# ── Pagination state ──────────────────────────────────────────────────────────
if "search_page" not in st.session_state:
    st.session_state["search_page"] = 1

filter_key = (risk_filter, mfr_filter, cat_filter, country_filter, search_filter, page_size)
if st.session_state.get("_last_filter_key") != filter_key:
    st.session_state["search_page"] = 1
    st.session_state["_last_filter_key"] = filter_key

current_page = st.session_state["search_page"]

# ── Page header ───────────────────────────────────────────────────────────────
page_header(
    icon="🔍",
    title="Device Search",
    subtitle="Server-side filtering via GET /devices. Click a Device ID to open its details.",
)

with st.spinner("Fetching devices…"):
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

pagination  = resp.get("pagination", {})
total_items = pagination.get("total_items", 0)
total_pages = pagination.get("total_pages", 1)
items       = resp.get("items", [])

# ── Result count + pagination controls ────────────────────────────────────────
top_l, top_r = st.columns([3, 1])
with top_l:
    page_info = f" — page {current_page} of {total_pages}" if total_pages > 1 else ""
    st.markdown(
        f"<div style='font-size:0.9em; color:var(--text-color); padding: 6px 0;'>"
        f"<strong>{total_items:,} devices</strong> found{page_info}"
        f"</div>",
        unsafe_allow_html=True,
    )
with top_r:
    if total_pages > 1:
        nav_l, nav_m, nav_r = st.columns([1, 2, 1])
        if nav_l.button("◀", disabled=current_page <= 1, help="Previous page"):
            st.session_state["search_page"] = max(1, current_page - 1)
            st.rerun()
        nav_m.markdown(
            f"<div style='text-align:center;padding-top:8px;font-size:0.85em;color:var(--text-color);opacity:0.7;'>"
            f"{current_page} / {total_pages}</div>",
            unsafe_allow_html=True,
        )
        if nav_r.button("▶", disabled=current_page >= total_pages, help="Next page"):
            st.session_state["search_page"] = min(total_pages, current_page + 1)
            st.rerun()

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
        did       = item.get("device_id", "")
        name      = _clean(item.get("device_name"))
        cls       = _clean(item.get("device_classification"))
        country   = _clean(item.get("device_country"))
        mfr       = _clean(item.get("mfr_parent_company") or item.get("mfr_name"))
        score     = item.get("risk_score")
        score_str = f"{score:.1f}" if score is not None else "—"
        rl        = item.get("risk_level")
        badge     = _badge(rl)

        # All colors use CSS classes — zero hardcoded hex colors
        rows_html += (
            f"<tr>"
            f"<td class='td-cell'><span class='device-link'>{did}</span></td>"
            f"<td class='td-cell'>{name}</td>"
            f"<td class='td-cell muted'>{cls}</td>"
            f"<td class='td-cell muted'>{country}</td>"
            f"<td class='td-cell muted'>{mfr}</td>"
            f"<td class='td-cell center'>{badge}</td>"
            f"<td class='td-cell right'>{score_str}</td>"
            f"</tr>"
        )

    table_html = f"""
    <div class='table-wrap'>
      <table class='ds-table'>
        <thead>
          <tr>
            <th class='th-cell'>Device ID</th>
            <th class='th-cell'>Name</th>
            <th class='th-cell'>Classification</th>
            <th class='th-cell'>Country</th>
            <th class='th-cell'>Manufacturer</th>
            <th class='th-cell center'>Risk Level</th>
            <th class='th-cell right'>Score</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # ── Open Device Detail CTA ─────────────────────────────────────────────────
    st.markdown("<div class='cta-box'><div class='cta-title'>Open Device Detail</div>", unsafe_allow_html=True)
    selected_id = st.text_input(
        "Enter a Device ID from the table above:",
        key="quick_device_id",
        placeholder="e.g. 80508",
        label_visibility="collapsed",
    )
    if selected_id:
        st.session_state["detail_device_id"] = selected_id.strip()
        st.page_link(
            "pages/3_📋_Device_Detail.py",
            label=f"Open details for device {selected_id.strip()}",
            icon="📋",
        )
    else:
        st.markdown(
            "<div style='font-size:0.83em; color:var(--text-color); opacity:0.5;'>Type a Device ID above to navigate to its full profile.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.markdown(f"<div class='disclaimer'>⚕️ {DISCLAIMER}</div>", unsafe_allow_html=True)
