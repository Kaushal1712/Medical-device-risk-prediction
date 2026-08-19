"""
backend/main.py
================
Stage 7 — FastAPI application entry point.

Artifacts are loaded ONCE at startup via the ModelService singleton.
All 8 endpoints are registered here.

Run:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes import (
    assess,
    copilot,
    devices,
    explanation,
    feature_importance,
    health,
    predict,
    recommendation,
    risk_summary,
)
from backend.services.model_service import get_model_service
from src.config import HEALTHCARE_DISCLAIMER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load all artifacts once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== Medical Device Risk API — startup ===")
    get_model_service()   # triggers singleton load; raises on missing artifacts
    # Pre-warm the inference service (lazy-loaded on first /assess call)
    try:
        from backend.services import inference_service
        inference_service._load_artifacts()
        log.info("InferenceService pre-warmed successfully")
    except Exception as exc:
        log.warning("InferenceService pre-warm failed (will retry on first request): %s", exc)
    log.info("=== Startup complete — all artifacts loaded ===")
    yield
    log.info("=== Medical Device Risk API — shutdown ===")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical Device Failure Risk Prediction API",
    description=(
        "Decision-support API for predicting and explaining medical device failure risk. "
        f"\n\n**Disclaimer:** {HEALTHCARE_DISCLAIMER}"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permissive for hackathon (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url.path)},
    )


@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": str(exc)},
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "disclaimer": HEALTHCARE_DISCLAIMER},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(devices.router)
app.include_router(predict.router)
app.include_router(assess.router)          # NEW: query-driven risk assessment
app.include_router(risk_summary.router)
app.include_router(explanation.router)
app.include_router(feature_importance.router)
app.include_router(recommendation.router)
app.include_router(copilot.router)
