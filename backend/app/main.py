"""PathSnap FastAPI Application (Section 3).

Lifespan loading, CORS, request logging middleware, and /api endpoints.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .errors import AppError, ErrorCode, app_error_handler
from .jobs import job_manager
from .logging_setup import get_recent_logs, log_request, logger
from .model_loader import ModelBundle, load_hmm_config, load_hmm_decoder, load_model_bundle
from .schemas import (
    HealthResponse,
    JobStatusResponse,
    PredictRequest,
    PredictResponse,
    SafeErrorBody,
    TripInfo,
)

# Startup state
app_state: Dict[str, Any] = {
    "start_time": 0.0,
    "model_bundle": None,
    "hmm_decoder": None,
    "hmm_config": None,
    "hmm_config_hash": None,
    "indexed_trips": {},  # trip_id -> TripInfo dict
}


def _index_trips_directory(trips_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Index trips folder by checking row counts only. Never reads .route."""
    if not trips_dir.exists() or not trips_dir.is_dir():
        raise FileNotFoundError(f"TRIPS_DIR does not exist or is not a directory: {trips_dir}")

    indexed = {}
    for p in sorted(trips_dir.iterdir()):
        if not p.is_dir():
            continue
        trip_id = p.name
        track_p = p / f"{trip_id}.track"
        nodes_p = p / f"{trip_id}.nodes"
        arcs_p = p / f"{trip_id}.arcs"
        if track_p.exists() and nodes_p.exists() and arcs_p.exists():
            with open(track_p, "rb") as f:
                n_gps = sum(1 for _ in f)
            with open(nodes_p, "rb") as f:
                n_nodes = sum(1 for _ in f)
            with open(arcs_p, "rb") as f:
                n_arcs = sum(1 for _ in f)
            indexed[trip_id] = {
                "trip_id": trip_id,
                "n_gps_points": n_gps,
                "n_nodes": n_nodes,
                "n_arcs": n_arcs,
            }
    if not indexed:
        raise ValueError(f"No valid trips found in {trips_dir}")
    return indexed


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, HMM config, and index trips once at startup."""
    app_state["start_time"] = time.perf_counter()
    job_manager.ensure_executor()

    # 1. Load model bundle
    logger.info("Loading classifier model from configured model directory")
    try:
        model_bundle = load_model_bundle(settings.MODEL_DIR)
        app_state["model_bundle"] = model_bundle
    except Exception as e:
        logger.critical(f"FATAL: Failed to load model bundle: {e}")
        raise

    # 2. Load and validate the frozen config once in the backend process.
    app_state["hmm_config"] = None
    app_state["hmm_config_hash"] = None
    if settings.HMM_MODULE or settings.HMM_CONFIG:
        app_state["hmm_config"], app_state["hmm_config_hash"] = load_hmm_config(settings.HMM_CONFIG)
    # The current adapter independently rereads this same file; see README.
    hmm_decoder = load_hmm_decoder(settings.HMM_MODULE)
    app_state["hmm_decoder"] = hmm_decoder

    # 3. Index trips directory
    logger.info("Indexing configured trips folder")
    try:
        indexed_trips = _index_trips_directory(settings.TRIPS_DIR)
        app_state["indexed_trips"] = indexed_trips
        logger.info(f"Indexed {len(indexed_trips)} trips successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to index trips folder: {e}")
        raise

    yield

    # Shutdown
    job_manager.executor.shutdown(wait=False)
    logger.info("PathSnap API shutdown completed.")


app = FastAPI(
    title="PathSnap API",
    description="Live Evaluation API for PathSnap map-matching route reconstruction.",
    version="1.0.0",
    lifespan=lifespan,
)

# Exception handlers
app.add_exception_handler(AppError, app_error_handler)

# CORS configuration
allowed_origins = settings.ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "ngrok-skip-browser-warning"],
)


@app.middleware("http")
async def request_lifecycle_middleware(request: Request, call_next: Callable):
    """Enforce X-Request-ID, timing, and structured console/file request logging."""
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:10]
    request.state.request_id = req_id

    # Try to extract trip_id or job_id for request logging
    trip_id: Optional[str] = None
    job_id: Optional[str] = None
    path = request.url.path

    if "/api/jobs/" in path:
        job_id = path.split("/api/jobs/")[-1].split("/")[0]

    t0 = time.perf_counter()
    try:
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        # Retrieve trip_id attached to request state if any
        trip_id = getattr(request.state, "trip_id", None) or trip_id
        job_id = getattr(request.state, "job_id", None) or job_id

        # Attach X-Request-ID header to response
        response.headers["X-Request-ID"] = req_id
        log_request(
            method=request.method,
            path=path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            trip_id=trip_id,
            job_id=job_id,
            request_id=req_id,
        )
        return response
    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log_request(
            method=request.method,
            path=path,
            status_code=500,
            duration_ms=duration_ms,
            trip_id=trip_id,
            job_id=job_id,
            request_id=req_id,
        )
        raise e


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def get_health(request: Request):
    """Service health, model metadata, HMM status, trip count and uptime."""
    bundle: Optional[ModelBundle] = app_state["model_bundle"]
    if not bundle:
        raise AppError(ErrorCode.MODEL_NOT_READY, "Model not loaded.", request_id=request.state.request_id)

    uptime = time.perf_counter() - app_state["start_time"]
    return HealthResponse(
        status="ok",
        model_name=bundle.name,
        model_version=bundle.version,
        hmm_available=app_state["hmm_decoder"] is not None,
        hmm_config_name=settings.HMM_CONFIG.name if app_state["hmm_config"] is not None else None,
        model_sha256=bundle.model_hash,
        hmm_config_sha256=app_state["hmm_config_hash"],
        n_trips=len(app_state["indexed_trips"]),
        uptime_s=round(uptime, 2),
    )


@app.get("/api/trips", response_model=List[TripInfo])
def list_trips():
    """List available evaluation trips with point and network counts (no route info)."""
    indexed = app_state["indexed_trips"]
    trips = sorted(indexed.values(), key=lambda t: t["trip_id"])
    return [TripInfo(**t) for t in trips]


@app.post("/api/predict", status_code=status.HTTP_202_ACCEPTED, response_model=PredictResponse)
def predict_trip(req: PredictRequest, request: Request):
    """Initiate full pipeline and evaluation for a trip."""
    trip_id = req.trip_id.strip()
    request.state.trip_id = trip_id

    # Strictly disallow path characters (never accept filesystem paths)
    if "/" in trip_id or "\\" in trip_id or ".." in trip_id:
        raise AppError(
            ErrorCode.INVALID_REQUEST,
            "Invalid trip_id format: paths not allowed.",
            trip_id=trip_id,
            request_id=request.state.request_id,
        )

    indexed = app_state["indexed_trips"]
    if trip_id not in indexed:
        raise AppError(
            ErrorCode.TRIP_NOT_FOUND,
            f"Trip '{trip_id}' not found.",
            trip_id=trip_id,
            request_id=request.state.request_id,
            hint="Check GET /api/trips for the list of available trips.",
        )

    bundle: Optional[ModelBundle] = app_state["model_bundle"]
    if not bundle:
        raise AppError(ErrorCode.MODEL_NOT_READY, "Classifier model not ready.", request_id=request.state.request_id)

    job = job_manager.create_job(trip_id=trip_id, request_id=request.state.request_id)
    request.state.job_id = job.job_id

    job_manager.submit_trip(
        job=job,
        model_bundle=bundle,
        hmm_decoder=app_state["hmm_decoder"],
        hmm_config=app_state["hmm_config"],
        hmm_config_hash=app_state["hmm_config_hash"],
        refresh=req.refresh,
    )
    return PredictResponse(job_id=job.job_id)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, request: Request):
    """Poll status, stage progress, elapsed time, and result/error of a prediction job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise AppError(
            ErrorCode.INVALID_REQUEST,
            f"Job '{job_id}' not found.",
            request_id=request.state.request_id,
            status_code=404,
        )

    return JobStatusResponse(
        status=job.status,
        stage=job.stage,
        stage_index=job.stage_index,
        n_stages=job.n_stages,
        elapsed_s=job.elapsed_s(),
        result=job.result,
        error=SafeErrorBody(**job.error) if job.error else None,
    )


@app.get("/api/logs")
def get_logs(limit: int = Query(default=50, ge=1, le=200)):
    """Fetch recent activity log lines for the UI drawer."""
    return {"logs": get_recent_logs(limit=limit)}
