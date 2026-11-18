"""Stable, public route definitions for the PathSnap REST API.

Route handlers remain in ``backend/app/main.py`` so application startup,
middleware, and endpoint registration stay together.  Consumers can use this
module as the canonical inventory of supported API paths.
"""

API_PREFIX = "/api"

ROUTES = {
    "health": {"method": "GET", "path": f"{API_PREFIX}/health"},
    "trips": {"method": "GET", "path": f"{API_PREFIX}/trips"},
    "predict": {"method": "POST", "path": f"{API_PREFIX}/predict"},
    "job_status": {"method": "GET", "path": f"{API_PREFIX}/jobs/{{job_id}}"},
    "logs": {"method": "GET", "path": f"{API_PREFIX}/logs"},
}
