"""Public API contract for the PathSnap backend.

The FastAPI application is implemented in :mod:`app.main`; this package keeps
the externally consumed endpoint contract in one obvious backend location.
"""

from .routes import API_PREFIX, ROUTES

__all__ = ["API_PREFIX", "ROUTES"]
