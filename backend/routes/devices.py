"""
backend/routes/devices.py
==========================
GET /devices           — paginated filtered device list
GET /devices/{id}      — full device detail + risk + maintenance priority
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.schemas import (
    DeviceDetail,
    DeviceListItem,
    DeviceListResponse,
    Pagination,
)
from backend.services.model_service import get_model_service
from backend.services.recommendation_service import get_recommendation

log = logging.getLogger(__name__)
router = APIRouter()

_NONE_STR = frozenset({"", "nan", "none", "null", "<na>", "missing"})


def _clean(val) -> Optional[str]:
    """Convert a cell value to a clean string or None."""
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in _NONE_STR:
        return None
    return s


@router.get("/devices", response_model=DeviceListResponse, tags=["Devices"])
def list_devices(
    risk_level: Optional[str] = Query(None, description="Filter by risk level: LOW, MEDIUM, HIGH"),
    manufacturer: Optional[str] = Query(None, description="Partial match on mfr_name or mfr_parent_company"),
    category: Optional[str] = Query(None, description="Partial match on device_classification"),
    country: Optional[str] = Query(None, description="Exact match on device_country"),
    search: Optional[str] = Query(None, description="Partial match on device_name or device_id"),
    page: int = Query(1, ge=1, description="1-indexed page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
) -> DeviceListResponse:
    """
    Paginated device list with optional filters.

    Risk level is served from the pre-computed serving snapshot — never recomputed.
    Devices without a valid snapshot appear in the list with risk_level=null.
    """
    svc = get_model_service()
    device_df = svc.get_all_devices_df()
    risk_df = svc.get_risk_df().set_index("device_id")

    # Apply filters
    df = device_df.copy()

    if risk_level:
        rl = risk_level.strip().upper()
        # Keep devices whose risk_level matches OR devices with no snapshot (if rl==None not requested)
        # Filter: keep only devices in risk_index with matching level
        matching_devices = set(risk_df[risk_df["risk_level"] == rl].index.tolist())
        df = df[df["device_id"].isin(matching_devices)]

    if manufacturer:
        q = manufacturer.strip().lower()
        mfr_mask = (
            df["mfr_name"].fillna("").str.lower().str.contains(q, na=False)
            | df["mfr_parent_company"].fillna("").str.lower().str.contains(q, na=False)
        )
        df = df[mfr_mask]

    if category:
        q = category.strip().lower()
        df = df[df["device_classification"].fillna("").str.lower().str.contains(q, na=False)]

    if country:
        q = country.strip().upper()
        df = df[df["device_country"].fillna("").str.upper() == q]

    if search:
        q = search.strip().lower()
        search_mask = (
            df["device_name"].fillna("").str.lower().str.contains(q, na=False)
            | df["device_id"].fillna("").str.lower().str.contains(q, na=False)
        )
        df = df[search_mask]

    total_items = len(df)
    total_pages = max(1, math.ceil(total_items / page_size))
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    items: list[DeviceListItem] = []
    for _, row in page_df.iterrows():
        did = row["device_id"]
        risk_row = svc.get_device_risk(did)
        items.append(DeviceListItem(
            device_id=did,
            device_name=_clean(row.get("device_name")),
            device_classification=_clean(row.get("device_classification")),
            device_risk_class=_clean(row.get("device_risk_class")),
            device_country=_clean(row.get("device_country")),
            mfr_name=_clean(row.get("mfr_name")),
            mfr_parent_company=_clean(row.get("mfr_parent_company")),
            risk_level=risk_row["risk_level"] if risk_row else None,
            risk_score=float(risk_row["risk_score"]) if risk_row else None,
            calibrated_probability=float(risk_row["calibrated_probability"]) if risk_row else None,
            serving_event_date=str(risk_row["serving_event_date"]) if risk_row else None,
        ))

    return DeviceListResponse(
        items=items,
        pagination=Pagination(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
        ),
    )


@router.get("/devices/{device_id}", response_model=DeviceDetail, tags=["Devices"])
def get_device(device_id: str) -> DeviceDetail:
    """
    Full device record from merged.parquet plus the device's designated
    latest valid serving-snapshot risk score, or an explicit
    'prediction unavailable' indicator if no valid snapshot exists.
    Also includes maintenance priority from the rule-based engine.
    """
    svc = get_model_service()

    device_data = svc.get_device_detail(device_id)
    if device_data is None:
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found.")

    risk_row = svc.get_device_risk(device_id)
    prediction_available = risk_row is not None

    # Maintenance priority (abbreviated)
    rec = get_recommendation(device_id, svc)
    maintenance_priority = rec.maintenance_priority if rec.available else None

    return DeviceDetail(
        device_id=device_id,
        device_name=_clean(device_data.get("device_name")),
        device_description=_clean(device_data.get("device_description")),
        device_classification=_clean(device_data.get("device_classification")),
        device_risk_class=_clean(device_data.get("device_risk_class")),
        device_implanted=_clean(device_data.get("device_implanted")),
        device_country=_clean(device_data.get("device_country")),
        device_number=_clean(device_data.get("device_number")),
        device_distributed_to=_clean(device_data.get("device_distributed_to")),
        manufacturer_id=_clean(device_data.get("manufacturer_id")),
        mfr_name=_clean(device_data.get("mfr_name")),
        mfr_parent_company=_clean(device_data.get("mfr_parent_company")),
        mfr_source=_clean(device_data.get("mfr_source")),
        mfr_address=_clean(device_data.get("mfr_address")),
        prediction_available=prediction_available,
        prediction_unavailable_reason="" if prediction_available else "No valid serving snapshot for this device.",
        risk_level=risk_row["risk_level"] if risk_row else None,
        risk_score=float(risk_row["risk_score"]) if risk_row else None,
        calibrated_probability=float(risk_row["calibrated_probability"]) if risk_row else None,
        serving_event_date=str(risk_row["serving_event_date"]) if risk_row else None,
        model_version=risk_row.get("model_version") if risk_row else None,
        maintenance_priority=maintenance_priority,
    )
