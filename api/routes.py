"""
api/routes.py
--------------
REST endpoints for the Radiography Anomaly Detection service.

Primary endpoint:
    POST /api/v1/scan/analyze
        Accepts a multipart file upload (PNG/JPEG/DICOM), runs the anomaly
        classifier + Grad-CAM, and returns a JSON prediction with a URL to
        the annotated heatmap image.
"""

import os
import shutil
import tempfile
import time
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import config
from api.schemas import HealthResponse, ScanAnalysisResponse
from inference import get_model, run_inference
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["scan"])


def _validate_upload(upload: UploadFile) -> None:
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in config.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {sorted(config.ALLOWED_UPLOAD_EXTENSIONS)}"
            ),
        )


@router.post(
    "/scan/analyze",
    response_model=ScanAnalysisResponse,
    responses={400: {"description": "Bad request"}, 415: {"description": "Unsupported media type"}},
    summary="Analyze an X-ray scan for anomalies",
)
async def analyze_scan(file: UploadFile = File(...)):
    """
    Accepts a single radiograph (PNG, JPEG, or DICOM), runs the anomaly
    detection model, and returns:
      - a triage priority (HIGH / MEDIUM / LOW)
      - per-finding confidence scores
      - a URL to a Grad-CAM heatmap overlay highlighting the flagged region
    """
    request_id = uuid.uuid4().hex[:12]
    start_time = time.time()

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    _validate_upload(file)

    # Enforce max upload size by streaming to a temp file with a cap, rather
    # than reading the whole body into memory up front.
    max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    suffix = os.path.splitext(file.filename)[1].lower()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            total_written = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds max upload size of {config.MAX_UPLOAD_SIZE_MB}MB.",
                    )
                tmp.write(chunk)

        if total_written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        logger.info(
            "[%s] Received scan '%s' (%.2f KB) for analysis",
            request_id,
            file.filename,
            total_written / 1024,
        )

        result = run_inference(
            image_path=tmp_path,
            checkpoint_path=config.BEST_MODEL_PATH,
            save_heatmap=True,
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error("[%s] File error: %s", request_id, e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        # Includes: corrupt image, unreadable DICOM, GradCAM hook failure, etc.
        logger.error("[%s] Inference runtime error: %s", request_id, e)
        raise HTTPException(status_code=422, detail=f"Failed to process scan: {e}") from e
    except Exception as e:
        logger.exception("[%s] Unexpected error during scan analysis", request_id)
        raise HTTPException(
            status_code=500, detail=f"Internal server error during analysis: {e}"
        ) from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as cleanup_err:
                logger.warning(
                    "[%s] Failed to clean up temp file '%s': %s",
                    request_id,
                    tmp_path,
                    cleanup_err,
                )

    elapsed_ms = (time.time() - start_time) * 1000

    response = ScanAnalysisResponse(
        request_id=request_id,
        filename=file.filename,
        priority=result["priority"],
        top_finding=result["top_finding"],
        top_finding_confidence=result["top_finding_confidence"],
        flagged_findings=result["flagged_findings"],
        class_scores=result["class_scores"],
        heatmap_url=result.get("heatmap_url"),
        processing_time_ms=round(elapsed_ms, 2),
        model_version=config.BACKBONE,
    )

    logger.info(
        "[%s] Analysis complete: priority=%s top_finding=%s (%.3f) in %.1fms",
        request_id,
        response.priority,
        response.top_finding,
        response.top_finding_confidence,
        elapsed_ms,
    )

    return response


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check():
    checkpoint_found = os.path.exists(config.BEST_MODEL_PATH)
    model_loaded = False
    try:
        get_model(config.BEST_MODEL_PATH)
        model_loaded = True
    except Exception as e:
        logger.error("Health check: model failed to load: %s", e)

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        device=str(config.DEVICE),
        backbone=config.BACKBONE,
        checkpoint_found=checkpoint_found,
    )
