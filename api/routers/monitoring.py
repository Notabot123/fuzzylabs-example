"""Expose logged inference/drift stats for a human or an alerting job to consume."""

from fastapi import APIRouter, HTTPException, Query, Request

from api.state import ModelRegistry
from src.monitoring.drift import drift_summary

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


@router.get("/drift")
def get_drift(request: Request, window: int = Query(200, ge=1, le=5000)):
    registry = get_registry(request)
    if registry.baseline_input_stats is None:
        raise HTTPException(
            503, "no baseline input stats calibrated — run scripts/calibrate_monitoring.py first"
        )
    report = drift_summary(registry.baseline_input_stats, window=window)
    return report.__dict__
