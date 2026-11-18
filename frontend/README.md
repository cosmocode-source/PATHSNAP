# PathSnap Evaluator Web UI

Single-page web application for live evaluation of PathSnap map-matching route reconstruction.

---

## 1. Quick Start (Laptop 2)

### 1.1 Serve the Static Frontend
The frontend requires no build steps, bundlers, or npm dependencies.
From the repository root or inside `frontend/`:
```bat
cd frontend
python -m http.server 5500
```

### 1.2 Access the UI
Open your browser and navigate to:
```
http://localhost:5500
```

### 1.3 Connect to the Backend
1. In the top bar input field, paste the public ngrok URL generated on Laptop 1 (e.g. `https://xxxx-xx-xx.ngrok-free.app`) or `http://localhost:8000` for local testing.
2. Click **Connect** or **Test connection**.
3. The live status pill will turn:
   - **Green** (`Connected - LightGBM + HMM`) if the full pipeline including HMM decoder is active.
   - **Amber** (`Connected - classifier only`) if running in classifier-only mode.
4. The remembered URL is saved in `localStorage` and will persist across browser reloads.

---

## 2. Features

- **Sequential Batch Evaluation ("Run all 10")**: Runs the 10 evaluation trips in order, tracking real-time stage progress (`loading_trip`, `candidates`, `features`, `classifier`, `hmm`, `building_geometry`, `scoring`) with a live progress bar. Can be cancelled at any time without data loss.
- **Single Trip Runner ("Run trip")**: Evaluate any individual trip with a single click.
- **Interactive Leaflet Map**:
  - Raw GPS observations (small blue dots, hover for fix index, click for candidate and arc details).
  - Classifier route (dashed amber).
  - HMM route (thick teal solid polyline).
  - Violet rings highlighting points where HMM changed the classifier's prediction.
  - Purple dots indicating points immediately following long GPS gaps ($dt > 5s$).
  - Grey dots indicating stationary vehicle clusters.
  - Multi-segment rendering preventing false lines across disconnected road network breaks.
- **Metric Cards & Dual Bars**: Inline SVG visual comparison for all headline metrics (Top-1 all, Top-1 in candidates, Top-1 clean, Continuity, Miss rate, Sequence F1, Mismatch fraction).
- **Error & Data Quality Summary**: Table breakdown of correct/incorrect fixes, broken transitions, HMM fallbacks, and label quality shares.
- **CSV & JSON Export**: One-click download of the complete metrics CSV or individual trip result JSONs.
- **API Activity Drawer**: Real-time console feed displaying requests and backend stage durations.
- **Privacy & Safety Guarantees**: Reference routes, ground truth arc IDs, and per-point correctness flags are never exposed or rendered.

---

## 3. Demo-Day Checklist (Laptop 2)

1. [ ] Start the HTTP server: `python -m http.server 5500` inside `frontend/`.
2. [ ] Open `http://localhost:5500` in Google Chrome or modern browser.
3. [ ] Paste the ngrok HTTPS URL provided by Laptop 1 and click **Connect**.
4. [ ] Verify the status pill is green or amber.
5. [ ] Click **Run all 10** to evaluate the entire benchmark dataset.
6. [ ] Expand any completed row to review map geometry, metric cards, and comparison charts.
7. [ ] Export the CSV report using **Export CSV** in the summary strip.

---

## 4. Troubleshooting

- **`Failed to connect to backend` / CORS error**:
  - Verify Laptop 1 is running `ngrok http 8000` and the tunnel has not expired.
  - Ensure backend `.env` has `ALLOWED_ORIGINS=*`.
- **ngrok Interstitial Warning Page**:
  - All fetches from the frontend automatically include the `ngrok-skip-browser-warning: true` header. If accessing endpoints directly in browser tabs, accept the one-time ngrok disclaimer.
- **Map Tiles Loading Issue**:
  - Ensure Laptop 2 has internet access to reach the CartoDB/OpenStreetMap tile servers.
