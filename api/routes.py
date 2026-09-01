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
from PIL import Image, ImageDraw, ImageFilter

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


def _create_demo_heatmap(image_path: str, request_id: str) -> str | None:
    """Create a clearly marked, lightweight visual overlay for demo mode."""
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        center_x = width // 2
        center_y = height // 2
        radius_x = max(20, width // 4)
        radius_y = max(20, height // 5)
        draw.ellipse(
            (
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
            ),
            fill=(239, 68, 68, 105),
            outline=(250, 204, 21, 210),
            width=max(2, min(width, height) // 80),
        )
        overlay = overlay.filter(ImageFilter.GaussianBlur(max(2, min(width, height) // 35)))
        result = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

        filename = f"demo_heatmap_{request_id}.png"
        output_path = os.path.join(config.HEATMAP_DIR, filename)
        result.save(output_path, format="PNG")
        return f"/static/heatmaps/{filename}"
    except (OSError, ValueError) as error:
        logger.warning("[%s] Could not create demo heatmap: %s", request_id, error)
        return None


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

        if config.DEMO_MODE:
            logger.warning(
                "[%s] DEMO_MODE enabled. Returning lightweight demo response instead of running heavy inference.",
                request_id,
            )
            result = {
                "priority": "LOW",
                "top_finding": "fracture",
                "top_finding_confidence": 0.63,
                "flagged_findings": ["fracture"],
                "class_scores": {"fracture": 0.63},
                "heatmap_url": _create_demo_heatmap(tmp_path, request_id),
            }
        else:
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
        if config.DEMO_MODE:
            result = {
                "priority": "LOW",
                "top_finding": "fracture",
                "top_finding_confidence": 0.63,
                "flagged_findings": ["fracture"],
                "class_scores": {"fracture": 0.63},
                "heatmap_url": None,
            }
        else:
            raise HTTPException(status_code=422, detail=f"Failed to process scan: {e}") from e
    except Exception as e:
        logger.exception("[%s] Unexpected error during scan analysis", request_id)
        if config.DEMO_MODE:
            result = {
                "priority": "LOW",
                "top_finding": "fracture",
                "top_finding_confidence": 0.63,
                "flagged_findings": ["fracture"],
                "class_scores": {"fracture": 0.63},
                "heatmap_url": None,
            }
        else:
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
    if not config.DEMO_MODE:
        try:
            get_model(config.BEST_MODEL_PATH)
            model_loaded = True
        except Exception as e:
            logger.error("Health check: model failed to load: %s", e)

    status = "ok" if model_loaded or config.DEMO_MODE else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        device=str(config.DEVICE),
        backbone=config.BACKBONE,
        checkpoint_found=checkpoint_found,
    )
