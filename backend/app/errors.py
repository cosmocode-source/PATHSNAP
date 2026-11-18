"""Safe error definitions and exception handlers for PathSnap API.

Never exposes filesystem paths, stack traces, reference routes, or labels.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional
from fastapi import Request
from fastapi.responses import JSONResponse


class ErrorCode:
    """Stable error codes returned by the API."""
    TRIP_NOT_FOUND = "TRIP_NOT_FOUND"       # 404
    INVALID_REQUEST = "INVALID_REQUEST"     # 422
    MODEL_NOT_READY = "MODEL_NOT_READY"     # 503
    HMM_UNAVAILABLE = "HMM_UNAVAILABLE"     # 400 or 503
    PIPELINE_FAILED = "PIPELINE_FAILED"     # 500
    TIMEOUT = "TIMEOUT"                     # 504
    OUT_OF_MEMORY = "OUT_OF_MEMORY"         # 507
    INTERNAL = "INTERNAL"                   # 500


HTTP_STATUS_MAP = {
    ErrorCode.TRIP_NOT_FOUND: 404,
    ErrorCode.INVALID_REQUEST: 422,
    ErrorCode.MODEL_NOT_READY: 503,
    ErrorCode.HMM_UNAVAILABLE: 400,
    ErrorCode.PIPELINE_FAILED: 500,
    ErrorCode.TIMEOUT: 504,
    ErrorCode.OUT_OF_MEMORY: 507,
    ErrorCode.INTERNAL: 500,
}


def sanitize_message(msg: str) -> str:
    """Strip filesystem paths, tracebacks, or potential sensitive leakages from error strings."""
    if not msg:
        return ""
    # Strip Windows or Unix file paths: e.g. C:\... or /usr/...
    cleaned = re.sub(r"[a-zA-Z]:\\[^\s:\"']+", "[path]", msg)
    cleaned = re.sub(r"/(?:[a-zA-Z0-9_\-\.]+/)+[a-zA-Z0-9_\-\.]+", "[path]", cleaned)
    # Strip Traceback mentions
    if "Traceback" in cleaned:
        cleaned = cleaned.split("Traceback")[0].strip() or "An internal error occurred."
    return cleaned


class AppError(Exception):
    """Base application exception with sanitized output."""

    def __init__(
        self,
        code: str,
        message: str,
        stage: Optional[str] = None,
        trip_id: Optional[str] = None,
        request_id: Optional[str] = None,
        hint: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = sanitize_message(message)
        self.stage = stage
        self.trip_id = trip_id
        self.request_id = request_id
        self.hint = hint or ""
        self.status_code = status_code or HTTP_STATUS_MAP.get(code, 500)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "stage": self.stage,
                "trip_id": self.trip_id,
                "request_id": self.request_id,
                "hint": self.hint,
            }
        }


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handler for custom AppError exceptions."""
    req_id = getattr(request.state, "request_id", None) or exc.request_id
    if not exc.request_id and req_id:
        exc.request_id = req_id
    headers = {"X-Request-ID": req_id} if req_id else {}
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict(), headers=headers)
