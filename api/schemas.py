"""
api/schemas.py
---------------
Pydantic models describing API request/response payloads. Keeping these
separate from the route handlers gives FastAPI clean OpenAPI docs and lets
downstream consumers (e.g. the hospital's triage dashboard) rely on a
stable, typed contract.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class ScanAnalysisResponse(BaseModel):
    request_id: str
    filename: str
    priority: str = Field(..., description="HIGH | MEDIUM | LOW triage priority")
    top_finding: str
    top_finding_confidence: float = Field(..., ge=0.0, le=1.0)
    flagged_findings: List[str]
    class_scores: Dict[str, float]
    heatmap_url: Optional[str] = None
    processing_time_ms: float
    model_version: str


class ErrorResponse(BaseModel):
    error: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    backbone: str
    checkpoint_found: bool
