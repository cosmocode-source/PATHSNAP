/**
 * Main application coordinator for PathSnap Evaluator.
 */
import {
  fetchBackendLogs,
  fetchHealth,
  fetchJob,
  fetchTrips,
  postPredict,
  setRequestLogListener,
} from "./api.js";
import { state } from "./state.js";
import {
  exportResultsCsv,
  renderConnectionStatus,
  renderSummaryStrip,
  renderTripRows,
  showToast,
} from "./ui.js";

// Polling interval IDs
let healthPollInterval = null;
let activityLogInterval = null;
let hasAutoExpandedFirst = false;

document.addEventListener("DOMContentLoaded", () => {
  initApp();
});

async function initApp() {
  bindUiElements();

  // Connect request logger to state
  setRequestLogListener((logEntry) => {
    state.addUiLog(logEntry);
    renderActivityLogs();
  });

  // Subscribe state changes
  state.subscribe((event, payload) => {
    if (event === "connectionChanged") {
      renderConnectionStatus(payload.status, payload.healthData);
      updateConnectionBanner(payload.status);
    } else if (event === "tripsLoaded") {
      populateTripDropdown(payload);
      renderTripRows(state.tripRows, onRowToggle, onRunSingleTrip);
    } else if (event === "tripRowUpdated" || event === "rowExpansionChanged") {
      renderTripRows(state.tripRows, onRowToggle, onRunSingleTrip);
      renderSummaryStrip(state.getSummaryStats());
    }
  });

  // Set initial API URL in input
  const urlInput = document.getElementById("api-url-input");
  if (urlInput) {
    urlInput.value = state.apiUrl;
  }

  // Initial connection
  await checkConnectionAndLoad(true);

  // Periodic health check (every 10s)
  healthPollInterval = setInterval(() => checkConnectionAndLoad(false), 10000);

  // Periodic backend logs poll (every 5s)
  activityLogInterval = setInterval(pollBackendLogs, 5000);
}

function bindUiElements() {
  const urlInput = document.getElementById("api-url-input");
  const btnConnect = document.getElementById("btn-connect");
  const btnTestConn = document.getElementById("btn-test-conn");
  const tripSelect = document.getElementById("trip-select");
  const btnRunTrip = document.getElementById("btn-run-trip");
  const btnRunAll = document.getElementById("btn-run-all");
  const skipCacheToggle = document.getElementById("skip-cache-toggle");
  const drawerToggle = document.getElementById("drawer-toggle");
  const btnExportCsv = document.getElementById("summary-strip");

  // Connect Button
  if (btnConnect) {
    btnConnect.addEventListener("click", () => {
      const url = urlInput.value.trim();
      if (!url) return;
      state.setApiUrl(url);
      checkConnectionAndLoad(true);
    });
  }

  // Test Connection Button
  if (btnTestConn) {
    btnTestConn.addEventListener("click", async () => {
      const url = urlInput.value.trim();
      state.setApiUrl(url);
      btnTestConn.disabled = true;
      btnTestConn.textContent = "Testing...";
      try {
        const health = await fetchHealth();
        showToast(`Connected! Model: ${health.model_name} (${health.model_version})`, "success");
        state.setConnection(health.hmm_available ? "connected_full" : "connected_classifier", health);
      } catch (err) {
        showToast(`Connection failed: ${err.message}`, "error");
        state.setConnection("unreachable");
      } finally {
        btnTestConn.disabled = false;
        btnTestConn.textContent = "Test connection";
      }
    });
  }

  // Trip Dropdown
  if (tripSelect) {
    tripSelect.addEventListener("change", (e) => {
      state.selectedTripId = e.target.value;
    });
  }

  // Skip Cache Toggle
  if (skipCacheToggle) {
    skipCacheToggle.addEventListener("change", (e) => {
      state.skipCache = e.target.checked;
    });
  }

  // Run Single Trip Button
  if (btnRunTrip) {
    btnRunTrip.addEventListener("click", () => {
      if (!state.selectedTripId) {
        showToast("Please select a trip first.", "warning");
        return;
      }
      onRunSingleTrip(state.selectedTripId);
    });
  }

  // Run All 10 / Cancel Button
  if (btnRunAll) {
    btnRunAll.addEventListener("click", () => {
      if (state.isRunningAll) {
        // Cancel run
        state.runAllCancelled = true;
        btnRunAll.textContent = "Cancelling...";
        btnRunAll.disabled = true;
      } else {
        runAllTrips();
      }
    });
  }

  // Export CSV Delegation
  document.addEventListener("click", (e) => {
    if (e.target.closest("#btn-export-csv")) {
      exportResultsCsv(state.tripRows);
    }
  });

  // Drawer Toggle
  if (drawerToggle) {
    drawerToggle.addEventListener("click", () => {
      const drawer = document.getElementById("activity-drawer");
      if (drawer) {
        const isOpen = drawer.classList.toggle("open");
        drawerToggle.setAttribute("aria-expanded", isOpen);
        drawerToggle.querySelector(".drawer-icon").textContent = isOpen ? "\u25BC" : "\u25B2";
      }
    });
  }
}

async function checkConnectionAndLoad(initial = false) {
  try {
    const health = await fetchHealth();
    const connType = health.hmm_available ? "connected_full" : "connected_classifier";
    state.setConnection(connType, health);

    // If trips list is empty, fetch it
    if (!state.trips.length) {
      const trips = await fetchTrips();
      state.setTrips(trips);
    }
    if (initial) {
      showToast(`Connected to PathSnap API (${health.n_trips} trips indexed)`, "success");
    }
  } catch (err) {
    state.setConnection("unreachable");
    if (initial) {
      showToast("Cannot connect to backend. Please check URL and server status.", "error");
    }
  }
}

function updateConnectionBanner(status) {
  let banner = document.getElementById("connection-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "connection-banner";
    document.body.prepend(banner);
  }

  if (status === "unreachable" || status === "disconnected") {
    banner.style.display = "block";
    banner.className = "banner-alert banner-error";
    banner.innerHTML = `
      <span>&#x26A0; <strong>Backend unreachable.</strong> Ensure the FastAPI server is running on Laptop 1 and ngrok URL is active.</span>
    `;
  } else {
    banner.style.display = "none";
  }
}

function populateTripDropdown(trips) {
  const select = document.getElementById("trip-select");
  if (!select) return;

  select.innerHTML = trips.map((t) => `
    <option value="${t.trip_id}" ${t.trip_id === state.selectedTripId ? "selected" : ""}>
      ${t.trip_id} &mdash; ${t.n_gps_points.toLocaleString()} points (${t.n_arcs.toLocaleString()} arcs)
    </option>
  `).join("");
}

function onRowToggle(tripId) {
  state.toggleRowExpansion(tripId);
}

async function onRunSingleTrip(tripId) {
  const row = state.tripRows.get(tripId);
  if (row && row.status === "running") {
    showToast(`Trip ${tripId} is already running.`, "warning");
    return;
  }

  state.updateTripRow(tripId, {
    status: "queued",
    stage: "queued",
    stage_index: 0,
    elapsed_s: 0,
    error: null,
  });

  try {
    const postRes = await postPredict(tripId, state.skipCache);
    const jobId = postRes.job_id;

    // Auto-poll job
    await pollJobUntilDone(jobId, tripId);
  } catch (err) {
    state.updateTripRow(tripId, {
      status: "failed",
      stage: err.stage || "predict_request",
      error: {
        code: err.code || "REQUEST_FAILED",
        message: err.message,
        stage: err.stage || "predict_request",
        hint: err.hint,
      },
    });
    showToast(`Trip ${tripId} failed: ${err.message}`, "error");
  }
}

async function pollJobUntilDone(jobId, tripId) {
  const pollIntervalMs = 600;
  const maxPolls = 300; // ~3 mins max

  for (let i = 0; i < maxPolls; i++) {
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));

    try {
      const job = await fetchJob(jobId);
      state.updateTripRow(tripId, {
        status: job.status,
        stage: job.stage,
        stage_index: job.stage_index,
        n_stages: job.n_stages,
        elapsed_s: job.elapsed_s,
        result: job.result,
        error: job.error,
      });

      if (job.status === "done") {
        // Auto-expand first finished trip
        if (!hasAutoExpandedFirst) {
          hasAutoExpandedFirst = true;
          state.toggleRowExpansion(tripId, true);
        }
        showToast(`Trip ${tripId} completed successfully!`, "success");
        return job.result;
      } else if (job.status === "failed") {
        showToast(`Trip ${tripId} failed at stage ${job.stage}`, "error");
        return null;
      }
    } catch (pollErr) {
      console.warn("Poll attempt failed:", pollErr);
    }
  }

  throw new Error("Job polling timed out.");
}

async function runAllTrips() {
  if (!state.trips.length) {
    showToast("No trips available to run.", "warning");
    return;
  }

  state.isRunningAll = true;
  state.runAllCancelled = false;

  const btnRunAll = document.getElementById("btn-run-all");
  const progressWrap = document.getElementById("batch-progress-wrap");
  const progressText = document.getElementById("batch-progress-text");
  const progressBar = document.getElementById("batch-progress-bar");

  btnRunAll.textContent = "Cancel Run All";
  btnRunAll.classList.add("btn-danger");
  if (progressWrap) progressWrap.style.display = "flex";

  const total = state.trips.length;
  let succeeded = 0;
  let failed = 0;

  for (let idx = 0; idx < total; idx++) {
    if (state.runAllCancelled) {
      showToast("Run all cancelled by user.", "warning");
      break;
    }

    const trip = state.trips[idx];
    const tripNum = idx + 1;
    if (progressText) progressText.textContent = `${tripNum} / ${total} (Running ${trip.trip_id}...)`;
    if (progressBar) progressBar.style.width = `${((tripNum - 1) / total) * 100}%`;

    try {
      await onRunSingleTrip(trip.trip_id);
      const row = state.tripRows.get(trip.trip_id);
      if (row && row.status === "done") {
        succeeded++;
      } else {
        failed++;
      }
    } catch (e) {
      failed++;
    }
  }

  if (progressBar) progressBar.style.width = "100%";
  if (progressText) progressText.textContent = `Completed: ${succeeded} succeeded, ${failed} failed`;

  state.isRunningAll = false;
  btnRunAll.textContent = "Run all 10";
  btnRunAll.classList.remove("btn-danger");
  btnRunAll.disabled = false;

  // Show summary modal/toast
  if (failed > 0) {
    showToast(`Batch completed: ${succeeded} succeeded, ${failed} failed. Click 'Retry failed' to re-run.`, "warning");
    renderRetryFailedButton();
  } else {
    showToast(`All ${succeeded} trips evaluated successfully!`, "success");
  }
}

function renderRetryFailedButton() {
  let btnRetryFailed = document.getElementById("btn-retry-failed");
  if (!btnRetryFailed) {
    const controlActions = document.querySelector(".control-actions");
    if (controlActions) {
      btnRetryFailed = document.createElement("button");
      btnRetryFailed.id = "btn-retry-failed";
      btnRetryFailed.className = "btn btn-secondary btn-sm";
      btnRetryFailed.textContent = "Retry failed";
      btnRetryFailed.addEventListener("click", retryFailedTrips);
      controlActions.appendChild(btnRetryFailed);
    }
  } else {
    btnRetryFailed.style.display = "inline-flex";
  }
}

async function retryFailedTrips() {
  const failedTrips = Array.from(state.tripRows.values())
    .filter((r) => r.status === "failed")
    .map((r) => r.trip_id);

  if (!failedTrips.length) {
    showToast("No failed trips to retry.", "info");
    return;
  }

  for (const tid of failedTrips) {
    await onRunSingleTrip(tid);
  }
}

async function pollBackendLogs() {
  try {
    const data = await fetchBackendLogs(50);
    if (data && data.logs) {
      state.setBackendLogs(data.logs);
      renderActivityLogs();
    }
  } catch (err) {
    // Silently ignore log polling errors
  }
}

function renderActivityLogs() {
  const listEl = document.getElementById("activity-logs-list");
  if (!listEl) return;

  const logs = [...state.uiLogs, ...state.backendLogs];
  // Sort by time or display combined
  if (!logs.length) {
    listEl.innerHTML = `<div class="log-empty">No activity yet. Initiate an evaluation to see live events.</div>`;
    return;
  }

  listEl.innerHTML = logs.slice(0, 60).map((l) => {
    const summary = l.summary || `${l.time} | ${l.method || ""} ${l.path || ""} | Status: ${l.status || "-"}`;
    let typeClass = "log-info";
    if (l.type === "DONE" || l.status === 200 || l.status === 202) typeClass = "log-success";
    if (l.type === "ERROR" || (l.status && l.status >= 400)) typeClass = "log-error";
    if (l.type === "STAGE") typeClass = "log-stage";

    return `
      <div class="log-line ${typeClass}">
        <span class="log-mono">${summary}</span>
      </div>
    `;
  }).join("");
}
