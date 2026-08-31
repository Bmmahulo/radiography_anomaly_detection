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
    Avoid eager model loading in constrained environments like Render free tiers.
    The model is loaded lazily on first inference request instead of at startup,
    which prevents the process from being killed by an out-of-memory exit (137)
    during boot.
    """
    if not os.path.exists(config.BEST_MODEL_PATH):
        logger.warning(
            "Checkpoint %s not found. The app will stay up but inference will fail "
            "until the model file is present.",
            config.BEST_MODEL_PATH,
        )
        return

    logger.info(
        "Startup complete. Model will be loaded on demand to reduce memory pressure "
        "on constrained deployments."
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
