# PathSnap Backend API

FastAPI backend for PathSnap live map-matching evaluation.

---

## 1. Quick Start (Windows Laptop 1)

### 1.1 Prerequisites
- Python 3.10+ (tested on Python 3.12)
- [ngrok](https://ngrok.com/download) installed (or on PATH)

### 1.2 Setup Environment
```bat
cd backend
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 1.3 Configuration
Copy the template configuration file:
```bat
copy .env.example .env
```
Key settings in `.env`:
- `TRIPS_DIR`: Path to the evaluation trips directory (default `../Dataset/student_test`; on demo day point to the evaluator folder).
- `MODEL_DIR`: Path to the classifier model folder (default `../trained_models/model4`).
- `HMM_MODULE`: Python path of the HMM decoder (format `module.path:function_name`; leave blank if HMM is not available).
- `PORT`: Port to listen on (default `8000`).

### 1.4 Start the Backend
Double-click `start_backend.bat` or run:
```bat
start_backend.bat
```
The server will start on `http://localhost:8000`.

### 1.5 Expose with ngrok
In a separate terminal on Laptop 1:
```bat
ngrok http 8000
```
Copy the public `https://...ngrok-free.app` URL and paste it into the UI on Laptop 2.

---

## 2. API Endpoints

All endpoints are prefixed with `/api`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Service health, model info, HMM status, trip count, uptime |
| `GET` | `/api/trips` | Sorted list of trips with GPS point and network counts (no routes) |
| `POST` | `/api/predict` | Initiate evaluation job: `{"trip_id": "...", "refresh": false}` -> 202 `{job_id}` |
| `GET` | `/api/jobs/{id}` | Poll job status: `queued`, `running`, `done`, `failed` + stage + result/error |
| `GET` | `/api/logs?limit=50`| Recent console and stage events for the UI activity drawer |

---

## 3. Connecting the HMM Module Later

The HMM decoder is dynamically plugged in via the `HMM_MODULE` setting in `backend/.env`.

### 3.1 Contract
The target function must match the following signature:
```python
def decode_trip(
    candidates: pd.DataFrame,    # Columns: [gps_row, arc_id, dist_m, rank]
    probs: np.ndarray,           # (M,) normalised emission probabilities summing to 1 per GPS fix
    network: dict,               # {"arcs": (A,2), "node_xy": (V,2), "arc_a": (A,2), "arc_b": (A,2), "arc_len": (A,)}
    gps_xy: np.ndarray,          # (N,2) UTM projected GPS coordinates
    times: np.ndarray,           # (N,) timestamps in seconds
    config: dict,                # Preprocessing and HMM parameters
) -> dict:
    return {
        "arc_per_point": list_or_array_of_int,   # (N,) local arc_id chosen for each GPS fix
        "predicted_route": list_of_int,          # Collapsed sequence of unique consecutive arc_ids
        "diagnostics": {
            "fallbacks_unreachable": 0,          # Count of fallback events due to unreachable candidates
            "fallbacks_gap": 0,                  # Count of fallback events due to long GPS gaps
        }
    }
```
**CRITICAL**: The function must NEVER take or use any reference route parameter.

### 3.2 Activation
1. Place the module in Python's path (or in `backend/app/` or `trained_models/`).
2. In `backend/.env`, set:
   ```env
   HMM_MODULE=my_hmm_package.decoder:decode_trip
   ```
3. Restart the backend. `/api/health` will report `"hmm_available": true` and HMM metrics and routes will appear automatically in the UI.

---

## 4. Switching to the Final Tuned Model

When the hyperparameter tuning study completes:
1. In `backend/.env`, update:
   ```env
   MODEL_DIR=../trained_models/final_model
   ```
2. Ensure `final_model/` contains `config.json` and `lightgbm_model.txt`.
3. Restart the backend. No other code changes are required.

---

## 5. Demo-Day Checklist (Laptop 1)

1. [ ] **Update Trips Folder**: Set `TRIPS_DIR=C:/path/to/evaluator_trips` in `backend/.env`.
2. [ ] **Start Backend**: Run `start_backend.bat`. Keep console window visible for the evaluator.
3. [ ] **Start ngrok**: Run `ngrok http 8000`. Note the HTTPS URL.
4. [ ] **Check Health**: Open `https://<ngrok-url>/api/health` in a browser or curl. Ensure `status: ok` and `n_trips: 10`.
5. [ ] **Provide URL**: Provide the ngrok HTTPS URL to the evaluator or enter it on Laptop 2.
6. [ ] **Monitor Console**: Observe the formatted request and stage log lines flowing in real time.

---

## 6. Troubleshooting

- **CORS Error in Browser**: Ensure `ALLOWED_ORIGINS=*` in `backend/.env`.
- **ngrok Interstitial Page**: The frontend fetch wrapper automatically includes `ngrok-skip-browser-warning: true`.
- **Port 8000 In Use**: Change `PORT=8001` in `.env` and run `ngrok http 8001`.
- **Model Failed to Load**: Verify `MODEL_DIR` points to a folder containing `config.json` and a valid LightGBM model text file.
- **HMM Unavailable**: This is normal if `HMM_MODULE` is empty. The API operates cleanly in classifier-only mode.
