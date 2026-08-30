"""
app.py
------
FastAPI application entrypoint for the Radiography Anomaly Detection
prototype.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then visit http://localhost:8000/docs for interactive Swagger UI, or POST
an image to /api/v1/scan/analyze.
"""

import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from api.routes import router as scan_router
from inference import get_model
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Radiography Anomaly Detection API",
    description=(
        "AI-assisted triage prototype: analyzes chest/skeletal X-ray scans, "
        "flags likely anomalies (fractures, lesions, etc.), and returns a "
        "Grad-CAM heatmap so clinicians can quickly verify flagged regions. "
        "NOT approved for standalone clinical diagnostic use — screening/"
        "triage assistance only, all results require physician review."
    ),
    version="0.1.0",
)

# CORS: permissive for local prototyping. Restrict `allow_origins` to the
# hospital dashboard's actual origin before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated heatmap overlays as static files so the API response's
# `heatmap_url` is directly browsable/embeddable by a frontend.
os.makedirs(config.HEATMAP_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

app.include_router(scan_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


@app.on_event("startup")
async def load_model_on_startup():
    """
    Warms the model cache at server startup rather than on the first request,
    so the first real user isn't the one paying the model-load latency.
    Failure here is logged but non-fatal — /api/v1/health will report
    `model_loaded: false` and /scan/analyze will surface a clear 500 with
    the underlying error rather than crashing silently.
    """
    try:
        get_model(config.BEST_MODEL_PATH)
        logger.info("Model warmed and ready at startup.")
    except Exception as e:
        logger.error(
            "Model failed to load at startup (checkpoint may not exist yet "
            "— run train.py first). Server will still start. Error: %s",
            e,
        )


@app.get("/", tags=["root"])
async def root():
    return {
        "service": "Radiography Anomaly Detection API",
        "docs": "/docs",
        "analyze_endpoint": "/api/v1/scan/analyze",
        "health_endpoint": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=config.API_HOST, port=config.API_PORT, reload=True)
