"""Logging configuration for PathSnap API.

Outputs formatted events to console and backend/logs/api.log.
Maintains an in-memory ring buffer for the /api/logs activity feed.
Never logs route content, filesystem paths, or secrets.
"""
from __future__ import annotations

import collections
import datetime
import logging
import sys
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "api.log"

# In-memory buffer of recent log events for UI activity drawer
RECENT_LOGS: Deque[Dict[str, Any]] = collections.deque(maxlen=200)

logger = logging.getLogger("pathsnap.api")
logger.setLevel(logging.INFO)
logger.handlers.clear()

# File handler with standard formatting
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)
logger.propagate = False


def _get_time_str() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    trip_id: Optional[str] = None,
    job_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    """Log an HTTP request in the required console format."""
    t_str = _get_time_str()
    parts = [f" {t_str}", "REQUEST", f"{method} {path}"]
    if trip_id:
        parts.append(f"trip={trip_id}")
    if job_id:
        parts.append(f"job={job_id}")
    parts.append(str(status_code))
    parts.append(f"{duration_ms:.1f} ms" if duration_ms < 1000 else f"{duration_ms/1000:.2f} s")
    line = " | ".join(parts)
    logger.info(line)

    RECENT_LOGS.append({
        "type": "REQUEST",
        "time": t_str,
        "method": method,
        "path": path,
        "trip_id": trip_id,
        "job_id": job_id,
        "status": status_code,
        "duration_ms": duration_ms,
        "request_id": request_id,
        "summary": line.strip(),
    })


def log_stage(job_id: str, stage: str, duration_s: float, trip_id: Optional[str] = None) -> None:
    """Log a job pipeline stage execution."""
    t_str = _get_time_str()
    parts = [f" {t_str}", "STAGE  ", f"job={job_id}"]
    if trip_id:
        parts.append(f"trip={trip_id}")
    parts.append(f"{stage}")
    parts.append(f"{duration_s:.2f} s")
    line = " | ".join(parts)
    logger.info(line)

    RECENT_LOGS.append({
        "type": "STAGE",
        "time": t_str,
        "job_id": job_id,
        "trip_id": trip_id,
        "stage": stage,
        "duration_s": duration_s,
        "summary": line.strip(),
    })


def log_done(
    job_id: str,
    trip_id: str,
    top1_summary: str,
    total_s: float,
    from_cache: bool = False,
) -> None:
    """Log job completion with summary metric."""
    t_str = _get_time_str()
    cache_tag = " (cache hit)" if from_cache else ""
    line = f" {t_str} | DONE    | job={job_id} | trip={trip_id}{cache_tag} | {top1_summary} | {total_s:.2f} s"
    logger.info(line)

    RECENT_LOGS.append({
        "type": "DONE",
        "time": t_str,
        "job_id": job_id,
        "trip_id": trip_id,
        "summary": line.strip(),
        "total_s": total_s,
        "from_cache": from_cache,
    })


def log_error(job_id: Optional[str], trip_id: Optional[str], error_code: str, message: str) -> None:
    """Log job or request failure safely."""
    t_str = _get_time_str()
    parts = [f" {t_str}", "ERROR  "]
    if job_id:
        parts.append(f"job={job_id}")
    if trip_id:
        parts.append(f"trip={trip_id}")
    parts.append(error_code)
    parts.append(message)
    line = " | ".join(parts)
    logger.error(line)

    RECENT_LOGS.append({
        "type": "ERROR",
        "time": t_str,
        "job_id": job_id,
        "trip_id": trip_id,
        "error_code": error_code,
        "summary": line.strip(),
    })


def get_recent_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent log events for the activity drawer."""
    logs = list(RECENT_LOGS)
    return logs[-limit:]
