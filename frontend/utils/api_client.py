"""
frontend/utils/api_client.py
==============================
Typed Python wrappers for all 9 FastAPI endpoints.

Backend URL is read from the BACKEND_URL environment variable,
defaulting to http://localhost:8000.

All GET functions are decorated with @st.cache_data(ttl=60) so that
the same API call within a 60-second window is served from Streamlit's
in-memory cache rather than hitting the backend on every rerender.

POST /copilot is intentionally NOT cached — each question should always
reach the backend fresh.

Error handling strategy:
  - ConnectionError  → return None (backend not running)
  - 404              → return None (resource not found)
  - Other HTTP error → raise (unexpected — lets Streamlit show the traceback)
"""

from __future__ import annotations

import os
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

_TIMEOUT_READ = 10    # seconds for read-only endpoints
_TIMEOUT_COPILOT = 45  # seconds for copilot (LLM may be slow)


def _get(path: str, **params) -> Optional[dict | list]:
    """Internal GET helper. Returns parsed JSON or None on connection/404."""
    try:
        resp = requests.get(
            f"{BACKEND_URL}{path}",
            params={k: v for k, v in params.items() if v is not None},
            timeout=_TIMEOUT_READ,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None


def _post(path: str, body: dict) -> Optional[dict]:
    """Internal POST helper."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}{path}",
            json=body,
            timeout=_TIMEOUT_COPILOT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def get_health() -> Optional[dict]:
    """Return the /health response dict, or None if the backend is unreachable."""
    return _get("/health")


# ---------------------------------------------------------------------------
# GET /risk-summary
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_risk_summary() -> Optional[dict]:
    """Return the aggregate risk summary dict."""
    return _get("/risk-summary")


# ---------------------------------------------------------------------------
# GET /feature-importance
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)  # importance doesn't change mid-session
def get_feature_importance() -> Optional[dict]:
    """Return the global feature importance response dict."""
    return _get("/feature-importance")


# ---------------------------------------------------------------------------
# GET /devices  (paginated + filtered list)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def get_devices(
    risk_level: Optional[str] = None,
    manufacturer: Optional[str] = None,
    category: Optional[str] = None,
    country: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> Optional[dict]:
    """Return a paginated DeviceListResponse dict, or None."""
    return _get(
        "/devices",
        risk_level=risk_level or None,
        manufacturer=manufacturer or None,
        category=category or None,
        country=country or None,
        search=search or None,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# GET /devices/{id}
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_device_detail(device_id: str) -> Optional[dict]:
    """Return full DeviceDetail dict, or None if the device is not found."""
    return _get(f"/devices/{device_id}")


# ---------------------------------------------------------------------------
# GET /explanation/{id}
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def get_explanation(device_id: str) -> Optional[dict]:
    """
    Return ExplanationResponse dict.
    The backend caches SHAP computation to disk; first call may be slow.
    Returns None only on connection failure — the response always exists
    (available=False for unscored devices).
    """
    return _get(f"/explanation/{device_id}")


# ---------------------------------------------------------------------------
# GET /recommendation/{id}
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_recommendation(device_id: str) -> Optional[dict]:
    """Return RecommendationResponse dict."""
    return _get(f"/recommendation/{device_id}")


# ---------------------------------------------------------------------------
# POST /copilot  (NOT cached — always fresh)
# ---------------------------------------------------------------------------

def post_copilot(device_id: str, question: str) -> Optional[dict]:
    """Submit a question to the GenAI copilot and return the CopilotResponse dict."""
    return _post("/copilot", {"device_id": device_id, "question": question})


# ---------------------------------------------------------------------------
# POST /assess  (query-driven risk assessment — NOT cached)
# ---------------------------------------------------------------------------

def post_assess(
    device_information: str,
    problem_description: str,
    *,
    device_id: Optional[str] = None,
    device_classification: Optional[str] = None,
    device_risk_class: Optional[str] = None,
    device_implanted: Optional[str] = None,
    country: Optional[str] = None,
) -> Optional[dict]:
    """
    Submit a risk assessment query to POST /assess.

    device_id is optional and is only used for historical evidence retrieval.
    It is NOT used as a predictive ML feature.
    """
    body: dict = {
        "device_information": device_information,
        "problem_description": problem_description,
    }
    if device_id:
        body["device_id"] = device_id
    if device_classification:
        body["device_classification"] = device_classification
    if device_risk_class:
        body["device_risk_class"] = device_risk_class
    if device_implanted:
        body["device_implanted"] = device_implanted
    if country:
        body["country"] = country
    return _post("/assess", body)
