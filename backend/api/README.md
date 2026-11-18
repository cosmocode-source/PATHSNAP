# API package

This folder is the public API contract for the backend.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check service, model, HMM, and trip-index status. |
| `GET` | `/api/trips` | List the evaluation trips available to the UI. |
| `POST` | `/api/predict` | Start an evaluation job for a trip. |
| `GET` | `/api/jobs/{job_id}` | Poll a job until it completes or fails. |
| `GET` | `/api/logs` | Read recent backend activity for the UI log drawer. |

The FastAPI route handlers are registered in `backend/app/main.py`, and the
interactive OpenAPI specification is available while the service is running:

- `http://localhost:8000/docs`
- `http://localhost:8000/openapi.json`
