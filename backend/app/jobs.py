"""In-memory Job Store and Background Worker for PathSnap (Section 3.4 & 3.9).

Runs CPU-heavy evaluation jobs in a background thread pool.
Reports real stage progress:
loading_trip -> candidates -> features -> classifier -> hmm -> building_geometry -> scoring.
Caches finished results by (trip_id, model_version, hmm_version).
"""
from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import settings
from .errors import AppError, ErrorCode, sanitize_message
from .logging_setup import log_done, log_error, log_stage
from .model_loader import ModelBundle
from .pipeline import run_pipeline
from .scoring import score_prediction

STAGES = [
    "loading_trip",
    "candidates",
    "features",
    "classifier",
    "hmm",
    "building_geometry",
    "scoring",
]
STAGE_TO_INDEX = {s: i for i, s in enumerate(STAGES)}
N_STAGES = len(STAGES)


class Job:
    """Represents a single prediction job state."""

    def __init__(self, job_id: str, trip_id: str, request_id: Optional[str] = None):
        self.job_id = job_id
        self.trip_id = trip_id
        self.request_id = request_id
        self.status = "queued"  # queued | running | done | failed
        self.stage = "loading_trip"
        self.stage_index = 0
        self.n_stages = N_STAGES
        self.created_at = datetime.datetime.now(datetime.timezone.utc)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[Dict[str, Any]] = None

    def elapsed_s(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return round(end - self.start_time, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "trip_id": self.trip_id,
            "status": self.status,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "n_stages": self.n_stages,
            "elapsed_s": self.elapsed_s(),
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    """Manages thread pool execution and job/result caching."""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        # Cache key: (trip_id, model_version, hmm_version) -> cached result dict
        self.result_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=settings.MAX_WORKERS,
            thread_name_prefix="pathsnap-worker",
        )

    def ensure_executor(self) -> None:
        """Allow application lifespan to restart in a test or embedded server."""
        if getattr(self.executor, "_shutdown", False):
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=settings.MAX_WORKERS, thread_name_prefix="pathsnap-worker")

    def cleanup_old_jobs(self) -> None:
        """Remove jobs older than JOB_TTL_MINUTES."""
        now = datetime.datetime.now(datetime.timezone.utc)
        ttl = datetime.timedelta(minutes=settings.JOB_TTL_MINUTES)
        with self.lock:
            expired = [jid for jid, j in self.jobs.items() if now - j.created_at > ttl]
            for jid in expired:
                del self.jobs[jid]

    def create_job(self, trip_id: str, request_id: Optional[str] = None) -> Job:
        self.cleanup_old_jobs()
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id, trip_id, request_id)
        with self.lock:
            self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self.lock:
            return self.jobs.get(job_id)

    def submit_trip(
        self,
        job: Job,
        model_bundle: ModelBundle,
        hmm_decoder: Optional[Callable],
        hmm_config: Optional[dict] = None,
        hmm_config_hash: Optional[str] = None,
        refresh: bool = False,
    ) -> None:
        hmm_version = hmm_config_hash if hmm_decoder else "none"
        cache_key = (job.trip_id, model_bundle.model_hash, hmm_version, str(settings.TRIPS_DIR))

        config_unchanged = True
        if hmm_decoder is not None and hmm_config is not None:
            try:
                config_unchanged = hashlib.sha256(settings.HMM_CONFIG.read_bytes()).hexdigest()[:12] == hmm_config_hash
            except OSError:
                config_unchanged = False
        if not refresh and config_unchanged:
            with self.lock:
                if cache_key in self.result_cache:
                    cached = dict(self.result_cache[cache_key])
                    cached["from_cache"] = True
                    job.status = "done"
                    job.stage = "scoring"
                    job.stage_index = N_STAGES - 1
                    job.result = cached
                    top1_clf = cached["metrics"]["classifier"]["top1_all"] * 100
                    if cached["hmm_used"] and cached["metrics"]["hmm"]:
                        top1_h = cached["metrics"]["hmm"]["top1_all"] * 100
                        summary = f"hmm Top-1 {top1_h:.1f}% (classifier {top1_clf:.1f}%)"
                    else:
                        summary = f"classifier Top-1 {top1_clf:.1f}%"
                    log_done(job.job_id, job.trip_id, summary, 0.0, from_cache=True)
                    return

        # Submit to worker pool
        self.executor.submit(
            self._run_job,
            job,
            model_bundle,
            hmm_decoder,
            hmm_config,
            cache_key,
        )

    def _run_job(
        self,
        job: Job,
        model_bundle: ModelBundle,
        hmm_decoder: Optional[Callable],
        hmm_config: Optional[dict],
        cache_key: Tuple[str, str, str],
    ) -> None:
        job.status = "running"
        job.start_time = time.perf_counter()

        def on_stage_progress(stage_name: str, duration: float):
            job.stage = stage_name
            job.stage_index = STAGE_TO_INDEX.get(stage_name, 0)
            log_stage(job.job_id, stage_name, duration, trip_id=job.trip_id)

        try:
            # Check timeout wrapper
            t_start = time.perf_counter()

            # The immutable adapter rereads its config on each call. Refuse a
            # changed file rather than report a stale startup fingerprint.
            if hmm_decoder is not None and hmm_config is not None:
                try:
                    current = hashlib.sha256(settings.HMM_CONFIG.read_bytes()).hexdigest()[:12]
                except OSError as exc:
                    raise AppError(ErrorCode.HMM_UNAVAILABLE, "Decoder configuration is unavailable; restart the API.", stage="hmm") from exc
                if current != cache_key[2]:
                    raise AppError(ErrorCode.HMM_UNAVAILABLE, "Decoder configuration changed; restart the API.", stage="hmm")

            # 1. Run inference pipeline
            prediction = run_pipeline(
                trip_id=job.trip_id,
                trips_dir=settings.TRIPS_DIR,
                model_bundle=model_bundle,
                hmm_decoder=hmm_decoder,
                hmm_config=hmm_config,
                strict_hmm=settings.STRICT_HMM,
                max_display_points=settings.MAX_DISPLAY_POINTS,
                stage_callback=on_stage_progress,
            )

            if time.perf_counter() - t_start > settings.TRIP_TIMEOUT_S:
                raise AppError(
                    ErrorCode.TIMEOUT,
                    f"Trip processing exceeded timeout of {settings.TRIP_TIMEOUT_S} s.",
                    stage=job.stage,
                    trip_id=job.trip_id,
                )

            # 2. Run server-side scoring
            on_stage_progress("scoring", 0.0)
            scoring_res = score_prediction(
                prediction=prediction,
                trips_dir=settings.TRIPS_DIR,
            )
            if time.perf_counter() - t_start > settings.TRIP_TIMEOUT_S:
                raise AppError(ErrorCode.TIMEOUT, f"Trip processing exceeded timeout of {settings.TRIP_TIMEOUT_S} s.", stage="scoring", trip_id=job.trip_id)
            on_stage_progress("scoring", prediction.stage_timings.get("scoring", 0.0))

            total_s = time.perf_counter() - job.start_time
            job.end_time = time.perf_counter()

            # Assemble full Result JSON (Section 3.7)
            result = {
                "trip_id": job.trip_id,
                "model": {
                    "name": model_bundle.name,
                    "version": model_bundle.version,
                },
                "hmm_used": prediction.hmm_used,
                "from_cache": False,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_runtime_s": round(total_s, 3),
                "timings_s": {k: round(v, 4) for k, v in prediction.stage_timings.items()},
                "routes": {
                    "classifier": {
                        "arc_ids": prediction.classifier_route.arc_ids,
                        "n_arcs": prediction.classifier_route.n_arcs,
                        "segments": prediction.classifier_route.segments,
                    },
                    "hmm": {
                        "arc_ids": prediction.hmm_route.arc_ids,
                        "n_arcs": prediction.hmm_route.n_arcs,
                        "segments": prediction.hmm_route.segments,
                    } if (prediction.hmm_used and prediction.hmm_route) else None,
                },
                "gps": prediction.display_gps,
                "display_downsample_step": prediction.display_downsample_step,
                # Diagnostic markers retain their real GPS coordinates even
                # when the visible GPS trace is downsampled.
                "diagnostic_points": {
                    key: [[int(i), *prediction.raw_gps[i]] for i in prediction.per_point_comparison[key]]
                    for key in ("hmm_changed_indices", "after_gap_indices", "stationary_indices")
                },
                "gps_endpoints": [prediction.raw_gps[0], prediction.raw_gps[-1]] if prediction.raw_gps else [],
                "metrics": scoring_res["metrics"],
                "error_summary": scoring_res["error_summary"],
                "per_point_comparison": prediction.per_point_comparison,
                "data_quality_context": prediction.data_quality_context,
                "warnings": prediction.warnings,
            }

            job.status = "done"
            job.result = result

            # Do not cache a classifier-only fallback under an HMM cache key.
            if hmm_decoder is None or prediction.hmm_used:
                with self.lock:
                    self.result_cache[cache_key] = result

            top1_clf = scoring_res["metrics"]["classifier"]["top1_all"] * 100
            if prediction.hmm_used and scoring_res["metrics"]["hmm"]:
                top1_h = scoring_res["metrics"]["hmm"]["top1_all"] * 100
                summary = f"hmm Top-1 {top1_h:.1f}% (classifier {top1_clf:.1f}%)"
            else:
                summary = f"classifier Top-1 {top1_clf:.1f}%"
            log_done(job.job_id, job.trip_id, summary, total_s, from_cache=False)

        except AppError as e:
            job.status = "failed"
            job.end_time = time.perf_counter()
            err_dict = {
                "code": e.code,
                "message": e.message,
                "stage": e.stage or job.stage,
                "trip_id": job.trip_id,
                "request_id": job.request_id,
                "hint": e.hint,
            }
            job.error = err_dict
            log_error(job.job_id, job.trip_id, e.code, e.message)

        except Exception as e:
            job.status = "failed"
            job.end_time = time.perf_counter()
            safe_msg = sanitize_message(str(e))
            err_dict = {
                "code": ErrorCode.INTERNAL,
                "message": safe_msg or "An unexpected internal error occurred.",
                "stage": job.stage,
                "trip_id": job.trip_id,
                "request_id": job.request_id,
                "hint": "Check server logs for traceback.",
            }
            job.error = err_dict
            log_error(job.job_id, job.trip_id, ErrorCode.INTERNAL, safe_msg)


job_manager = JobManager()
