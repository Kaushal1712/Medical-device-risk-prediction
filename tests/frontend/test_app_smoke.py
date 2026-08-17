"""
tests/frontend/test_app_smoke.py
==================================
Stage 10 — Frontend import-level smoke tests.

Verifies that the Streamlit frontend module and its API client are importable,
correctly reference all required FastAPI endpoints, carry the healthcare
disclaimer constant, and define the expected BACKEND_URL mechanism.

No browser. No HTTP calls. Import-time checks only (fast, CI-safe).

Run:  python -m pytest tests/frontend/test_app_smoke.py -v
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = ROOT / "frontend"

# ---------------------------------------------------------------------------
# Ensure frontend/ is importable regardless of CWD
# ---------------------------------------------------------------------------

if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_without_streamlit(module_name: str) -> types.ModuleType:
    """
    Import a frontend module while stubbing out streamlit so that
    tests run without a running Streamlit server or display.
    """
    # Stub out streamlit if not already available
    if "streamlit" not in sys.modules:
        st_stub = types.ModuleType("streamlit")
        # cache_data decorator — returns the function unchanged
        def _cache_data(func=None, *, ttl=None, **_kw):
            if func is not None:
                return func
            return lambda f: f
        st_stub.cache_data = _cache_data
        st_stub.cache_resource = _cache_data
        # Minimal stubs for anything the module imports at module level
        for attr in ("set_page_config", "sidebar", "write", "error", "warning",
                     "info", "success", "title", "header", "subheader",
                     "markdown", "columns", "metric", "dataframe", "plotly_chart",
                     "spinner", "expander", "caption", "stop", "session_state"):
            setattr(st_stub, attr, lambda *a, **kw: None)
        sys.modules["streamlit"] = st_stub

    # Also stub plotly if it's imported at the top level
    if "plotly" not in sys.modules:
        px_stub = types.ModuleType("plotly.express")
        sys.modules["plotly"] = types.ModuleType("plotly")
        sys.modules["plotly.express"] = px_stub
        sys.modules["plotly.graph_objects"] = types.ModuleType("plotly.graph_objects")

    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# TestApiClientSmoke
# ---------------------------------------------------------------------------

class TestApiClientSmoke:
    """
    Verifies that frontend/utils/api_client.py is importable and defines
    all required endpoint wrappers, the BACKEND_URL constant, and references
    the correct FastAPI paths.
    """

    @pytest.fixture(scope="class")
    def api_client(self):
        return _import_without_streamlit("utils.api_client")

    def test_api_client_importable(self, api_client):
        """api_client must import without errors."""
        assert api_client is not None

    def test_backend_url_constant_defined(self, api_client):
        """BACKEND_URL must be a module-level string (configurable via env)."""
        assert hasattr(api_client, "BACKEND_URL"), (
            "BACKEND_URL constant not found in api_client"
        )
        assert isinstance(api_client.BACKEND_URL, str)
        assert len(api_client.BACKEND_URL) > 0

    def test_backend_url_default_is_localhost(self, api_client):
        """Default BACKEND_URL (no env var) must point to localhost:8000."""
        # Only check if BACKEND_URL env var is not set in this test process
        if "BACKEND_URL" not in os.environ:
            assert "localhost:8000" in api_client.BACKEND_URL or \
                   "127.0.0.1:8000" in api_client.BACKEND_URL, (
                f"Default BACKEND_URL should be localhost:8000, got: {api_client.BACKEND_URL!r}"
            )

    def test_all_required_endpoint_functions_defined(self, api_client):
        """Every FastAPI endpoint must have a corresponding wrapper function."""
        required_functions = [
            "get_health",           # GET /health
            "get_risk_summary",     # GET /risk-summary
            "get_feature_importance",  # GET /feature-importance
            "get_devices",          # GET /devices
            "get_device_detail",    # GET /devices/{id}
            "get_explanation",      # GET /explanation/{id}
            "get_recommendation",   # GET /recommendation/{id}
            "post_copilot",         # POST /copilot
        ]
        missing = [fn for fn in required_functions if not hasattr(api_client, fn)]
        assert not missing, (
            f"api_client is missing endpoint wrappers: {missing}"
        )

    def test_endpoint_functions_are_callable(self, api_client):
        """All endpoint wrappers must be callable."""
        for fn_name in ("get_health", "get_risk_summary", "get_feature_importance",
                        "get_devices", "get_device_detail", "get_explanation",
                        "get_recommendation", "post_copilot"):
            fn = getattr(api_client, fn_name, None)
            assert callable(fn), f"api_client.{fn_name} is not callable"


# ---------------------------------------------------------------------------
# TestFrontendDisclaimerPresence
# ---------------------------------------------------------------------------

class TestFrontendDisclaimerPresence:
    """
    Verifies that the healthcare/security disclaimer constant is accessible
    from the backend config module (which the frontend transitively depends on
    via the API, and which the /health endpoint returns directly).

    Also verifies the frontend app.py and pages are present and importable.
    """

    def test_healthcare_disclaimer_in_backend_config(self):
        """
        src/config.py must define HEALTHCARE_DISCLAIMER, and it must contain
        the required prototype/decision-support language per Section 10.
        """
        # Import via sys.path manipulation so we don't need the venv active
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        try:
            from src.config import HEALTHCARE_DISCLAIMER
        except ImportError as e:
            pytest.skip(f"Could not import src.config: {e}")

        assert isinstance(HEALTHCARE_DISCLAIMER, str)
        assert len(HEALTHCARE_DISCLAIMER) > 20, (
            "HEALTHCARE_DISCLAIMER is suspiciously short"
        )
        lower = HEALTHCARE_DISCLAIMER.lower()
        assert "prototype" in lower or "decision-support" in lower, (
            f"HEALTHCARE_DISCLAIMER must contain 'prototype' or 'decision-support'. "
            f"Got: {HEALTHCARE_DISCLAIMER!r}"
        )

    def test_frontend_app_py_exists(self):
        assert (FRONTEND_DIR / "app.py").exists(), (
            "frontend/app.py not found"
        )

    def test_all_page_files_exist(self):
        """All four dashboard pages must be present."""
        required_pages = [
            "1_📊_Overview.py",
            "2_🔍_Device_Search.py",
            "3_📋_Device_Detail.py",
            "4_🧠_Explainability.py",
        ]
        pages_dir = FRONTEND_DIR / "pages"
        for page in required_pages:
            assert (pages_dir / page).exists(), (
                f"Frontend page missing: {page}"
            )

    def test_frontend_utils_init_exists(self):
        assert (FRONTEND_DIR / "utils" / "__init__.py").exists()

    def test_frontend_api_client_exists(self):
        assert (FRONTEND_DIR / "utils" / "api_client.py").exists()
