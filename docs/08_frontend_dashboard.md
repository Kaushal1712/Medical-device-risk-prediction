# Stage 8 — Frontend Dashboard

**Project:** Medical Device Failure Risk Prediction System
**Stage:** 8 — Frontend Dashboard
**Date:** 2026-08-17
**Status:** ✅ Complete
**Stack:** Streamlit + Plotly (Python) — consuming the Stage 7 FastAPI backend

---

## Stack Decision

The master prompt (Section 5, Stage 8) specifies React + TypeScript + Tailwind CSS + Recharts. The following trade-off was made explicitly before implementation:

> *"If you determine partway through that a simpler stack (e.g., server-rendered templates) would be materially faster to deliver a polished result within hackathon time, you may propose that trade-off explicitly in docs/ before switching."*

**Rationale for Streamlit:**
- Keeps the entire codebase in Python — no Node/npm toolchain required
- Plotly charts are richer and more interactive than Recharts for data-heavy dashboards
- Significantly faster to build and iterate for a hackathon demo
- No separate build step — `streamlit run frontend/app.py` starts immediately
- All data access patterns (API client, caching, session state) are idiomatic Python

No functionality defined in the master prompt was omitted — all four specified page sections are present.

---

## File Map

```
frontend/
├── requirements.txt          # streamlit>=1.31, plotly>=5.18, requests>=2.31
├── app.py                    # Entry point — landing page + global CSS + sidebar
├── utils/
│   ├── __init__.py
│   ├── api_client.py         # Typed wrappers for all 9 FastAPI endpoints
│   └── charts.py             # Reusable Plotly chart builders
└── pages/
    ├── 1_📊_Overview.py      # Risk summary KPIs + distribution + breakdowns
    ├── 2_🔍_Device_Search.py # Paginated + filtered device table
    ├── 3_📋_Device_Detail.py # Full device profile + SHAP + recommendations + copilot
    └── 4_🧠_Explainability.py # Global importance + per-device local SHAP
```

---

## How to Run

### Prerequisites
Both the FastAPI backend and the Streamlit dashboard must be running simultaneously.

**Terminal 1 — FastAPI backend:**
```bash
cd medical-device-risk
source venv/bin/activate
uvicorn backend.main:app --reload
# → http://localhost:8000
```

**Terminal 2 — Streamlit dashboard:**
```bash
cd medical-device-risk
source venv/bin/activate
streamlit run frontend/app.py
# → http://localhost:8501
```

### Environment variables (optional)
```bash
# Override the backend URL (defaults to http://localhost:8000)
export BACKEND_URL=http://localhost:8000
```

Or add to `.env`:
```
BACKEND_URL=http://localhost:8000
```

---

## Pages

### 1. Overview (`/`)
- **Source:** `GET /risk-summary`
- 6-column KPI row: total devices, scored, unscored, HIGH/MEDIUM/LOW counts + %
- Risk level distribution: horizontal bar chart (Plotly)
- Risk score statistics: min / mean / median / max
- Top-15 category breakdown: stacked horizontal bar chart
- Top-15 manufacturer breakdown: stacked horizontal bar chart

### 2. Device Search
- **Source:** `GET /devices?risk_level=&manufacturer=&category=&country=&search=&page=&page_size=`
- Sidebar filters: risk level (select), manufacturer, category, country, free-text search
- Server-side pagination (25/50/100 rows per page)
- HTML table with color-coded risk badges (red/amber/green)
- Quick-navigate input to open Device Detail for any listed device ID

### 3. Device Details
- **Sources:** `GET /devices/{id}`, `GET /explanation/{id}`, `GET /recommendation/{id}`, `POST /copilot`
- Device attribute info card (8 fields)
- Plotly risk score gauge (0–100) with color-coded risk level
- SHAP waterfall chart (top 5 positive + top 5 negative contributions)
- Historical event summary KPI row (total / Class I / recall events)
- Maintenance recommendation card (priority badge + bulleted actions + rule inputs expander)
- Copilot Q&A panel (free-text question → answer + context expander + provider indicator)
- All "unavailable" states handled explicitly — no fabricated data

### 4. Explainability
- **Sources:** `GET /feature-importance`, `GET /explanation/{id}`
- Global importance: horizontal bar chart (top N features, configurable 5–62 via sidebar slider)
- Full feature importance table (all features, expandable)
- Per-device local SHAP: same waterfall chart as Device Details
- Base value / predicted value delta metrics
- Raw SHAP JSON expander

---

## API Endpoints Consumed

| Endpoint | Page(s) | Cached (TTL) |
|---|---|---|
| `GET /health` | All (sidebar) | 30 s |
| `GET /risk-summary` | Overview | 60 s |
| `GET /feature-importance` | Explainability | 300 s |
| `GET /devices` | Device Search | 30 s |
| `GET /devices/{id}` | Device Detail | 60 s |
| `GET /explanation/{id}` | Device Detail, Explainability | 120 s |
| `GET /recommendation/{id}` | Device Detail | 60 s |
| `POST /copilot` | Device Detail | No cache |

---

## Backend Addition (Stage 8 only)

The `GET /feature-importance` endpoint was added as part of Stage 8:
- **Route file:** `backend/routes/feature_importance.py`
- **Schemas:** `FeatureImportanceItem`, `FeatureImportanceResponse` in `backend/schemas.py`
- **Registered in:** `backend/main.py`
- **Tests:** `tests/api/test_feature_importance.py` (11 tests — all passing)

No other Stage 7 backend code was modified.

---

## Design Conventions

| Element | Convention |
|---|---|
| HIGH risk | `#dc2626` (red-600) |
| MEDIUM risk | `#d97706` (amber-600) |
| LOW risk | `#16a34a` (green-600) |
| Positive SHAP | `#ef4444` (risk-increasing) |
| Negative SHAP | `#3b82f6` (risk-reducing) |
| Font | Inter (Google Fonts) |
| Chart background | Transparent |

---

## Healthcare Disclaimer

The following disclaimer appears on every page (sidebar + page footer):

> *This system is a decision-support prototype and does not replace qualified maintenance, biomedical engineering, regulatory, or clinical judgment. It is not a certified medical device and does not guarantee patient safety outcomes.*

This mirrors the disclaimer in `/health`, `src/config.py::HEALTHCARE_DISCLAIMER`, and `docs/07_recommendations_rules.md`.

---

## Known Limitations

1. **Streamlit caching is process-scoped** — if multiple users run separate Streamlit processes, each has its own cache. Not a concern for a hackathon demo.
2. **SHAP first-call latency** — computing SHAP values for a new device takes 1–3 seconds (model load + TreeExplainer). Subsequent calls serve from `artifacts/explanations/` cache.
3. **No authentication** — appropriate for a hackathon demo; a production deployment would require auth.
4. **Copilot LLM key** — if `LLM_PROVIDER`/`LLM_API_KEY` are not set in `.env`, the copilot uses the deterministic template fallback. This is expected and documented.
