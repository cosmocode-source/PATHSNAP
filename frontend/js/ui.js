/**
 * UI Rendering and DOM manipulation for PathSnap Evaluator.
 */
import { renderMetricComparisonBar } from "./charts.js";
import { TripMap } from "./map.js";

// Active Leaflet map instances keyed by tripId
export const activeMaps = new Map();

// Helper to format percentage
export function formatPct(val) {
  if (val == null || isNaN(val)) return "-";
  return `${(val * 100).toFixed(1)}%`;
}

// Helper to format float
export function formatFloat(val, decimals = 3) {
  if (val == null || isNaN(val)) return "-";
  return val.toFixed(decimals);
}

// Copy text to clipboard helper
export async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied to clipboard!", "success");
  } catch (err) {
    showToast("Failed to copy to clipboard", "error");
  }
}

// Download JSON helper
export function downloadJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Export CSV of results across finished trips
export function exportResultsCsv(rows) {
  const finished = Array.from(rows.values()).filter((r) => r.status === "done" && r.result);
  if (!finished.length) {
    showToast("No finished trips to export.", "warning");
    return;
  }

  const headers = [
    "trip_id",
    "total_runtime_s",
    "hmm_used",
    "clf_top1_all",
    "clf_top1_in_cands",
    "clf_top1_clean",
    "clf_miss_rate",
    "clf_route_continuity",
    "clf_arc_precision",
    "clf_arc_recall",
    "clf_arc_f1",
    "clf_mismatch_frac",
    "hmm_top1_all",
    "hmm_top1_in_cands",
    "hmm_top1_clean",
    "hmm_route_continuity",
    "hmm_arc_precision",
    "hmm_arc_recall",
    "hmm_arc_f1",
    "hmm_mismatch_frac",
    "delta_top1_all",
    "delta_continuity",
    "delta_f1",
  ];

  const lines = [headers.join(",")];

  for (const r of finished) {
    const res = r.result;
    const m = res.metrics;
    const c = m.classifier;
    const h = m.hmm || {};
    const d = m.delta || {};

    const row = [
      res.trip_id,
      res.total_runtime_s,
      res.hmm_used ? "true" : "false",
      c.top1_all,
      c.top1_in_cands,
      c.top1_clean,
      c.candidate_miss_rate,
      c.route_continuity,
      c.sequence_arc_precision,
      c.sequence_arc_recall,
      c.sequence_arc_f1,
      c.sequence_length_weighted_mismatch,
      h.top1_all ?? "",
      h.top1_in_cands ?? "",
      h.top1_clean ?? "",
      h.route_continuity ?? "",
      h.sequence_arc_precision ?? "",
      h.sequence_arc_recall ?? "",
      h.sequence_arc_f1 ?? "",
      h.sequence_length_weighted_mismatch ?? "",
      d.top1_all ?? "",
      d.route_continuity ?? "",
      d.sequence_arc_f1 ?? "",
    ];
    lines.push(row.map((v) => (v != null ? String(v) : "")).join(","));
  }

  const csvContent = lines.join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `pathsnap_evaluation_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Toast notification
export function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("fade-out");
    setTimeout(() => toast.remove(), 400);
  }, 3500);
}

// Render Top Bar Connection Pill
export function renderConnectionStatus(status, healthData) {
  const pill = document.getElementById("connection-pill");
  if (!pill) return;

  pill.className = "status-pill";
  if (status === "connected_full") {
    pill.classList.add("connected-full");
    pill.innerHTML = `<span class="dot"></span> Connected - LightGBM + HMM`;
  } else if (status === "connected_classifier") {
    pill.classList.add("connected-classifier");
    pill.innerHTML = `<span class="dot"></span> Connected - classifier only`;
  } else if (status === "unreachable") {
    pill.classList.add("unreachable");
    pill.innerHTML = `<span class="dot"></span> Unreachable`;
  } else {
    pill.classList.add("disconnected");
    pill.innerHTML = `<span class="dot"></span> Not connected`;
  }
}

// Render Summary Strip
export function renderSummaryStrip(stats) {
  const strip = document.getElementById("summary-strip");
  if (!strip) return;

  if (!stats) {
    strip.style.display = "none";
    return;
  }

  strip.style.display = "flex";
  const avg = stats.averages;
  const hasHmm = stats.hasHmm;

  const top1Clf = formatPct(avg.top1_all.classifier);
  const top1Hmm = hasHmm ? formatPct(avg.top1_all.hmm) : null;
  const contClf = formatPct(avg.route_continuity.classifier);
  const contHmm = hasHmm ? formatPct(avg.route_continuity.hmm) : null;
  const f1Clf = formatFloat(avg.sequence_arc_f1.classifier);
  const f1Hmm = hasHmm ? formatFloat(avg.sequence_arc_f1.hmm) : null;

  let deltaTop1Html = "";
  if (hasHmm && avg.top1_all.delta != null) {
    const d = avg.top1_all.delta;
    const sign = d > 0 ? "+" : "";
    const pClass = d > 0 ? "delta-pill positive" : d < 0 ? "delta-pill negative" : "delta-pill neutral";
    deltaTop1Html = `<span class="${pClass}">${sign}${(d * 100).toFixed(1)}%</span>`;
  }

  strip.innerHTML = `
    <div class="summary-metric-card">
      <div class="summary-label">Mean Top-1 Accuracy</div>
      <div class="summary-val-wrap">
        <span class="summary-value">${hasHmm ? top1Hmm : top1Clf}</span>
        ${deltaTop1Html}
      </div>
      <div class="summary-sub">${hasHmm ? `Classifier: ${top1Clf}` : "Headline unweighted"}</div>
    </div>

    <div class="summary-metric-card">
      <div class="summary-label">Mean Route Continuity</div>
      <div class="summary-val-wrap">
        <span class="summary-value">${hasHmm ? contHmm : contClf}</span>
      </div>
      <div class="summary-sub">${hasHmm ? `Classifier: ${contClf}` : "Connected transitions"}</div>
    </div>

    <div class="summary-metric-card">
      <div class="summary-label">Mean Sequence F1</div>
      <div class="summary-val-wrap">
        <span class="summary-value">${hasHmm ? f1Hmm : f1Clf}</span>
      </div>
      <div class="summary-sub">${hasHmm ? `Classifier: ${f1Clf}` : "Arc-level overlap"}</div>
    </div>

    <div class="summary-metric-card">
      <div class="summary-label">Trips Evaluated</div>
      <div class="summary-val-wrap">
        <span class="summary-value">${stats.nDone} / ${stats.totalTrips}</span>
      </div>
      <div class="summary-sub">Total time: ${stats.totalTime}s</div>
    </div>

    <div class="summary-actions">
      <button id="btn-export-csv" class="btn btn-secondary btn-sm">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Export CSV
      </button>
    </div>
  `;
}

// Render Trip Rows in Accordion
export function renderTripRows(tripRows, onRowToggle, onRerunTrip) {
  const container = document.getElementById("results-accordion");
  if (!container) return;

  const rows = Array.from(tripRows.values());
  if (!rows.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">&#x1F6E3;</div>
        <h3>No evaluation trips loaded</h3>
        <p>Connect to the PathSnap API backend to load the student evaluation trips.</p>
      </div>
    `;
    return;
  }

  // Preserve existing map nodes or re-render
  container.innerHTML = "";

  for (const row of rows) {
    const rowEl = createTripRowElement(row, onRowToggle, onRerunTrip);
    container.appendChild(rowEl);

    // If expanded and has result, initialize map after DOM mount
    if (row.isExpanded && row.result) {
      setTimeout(() => {
        let tripMap = activeMaps.get(row.trip_id);
        if (!tripMap) {
          tripMap = new TripMap(`map-${row.trip_id}`);
          activeMaps.set(row.trip_id, tripMap);
        }
        tripMap.init(row.result);
      }, 50);
    }
  }
}

function createTripRowElement(row, onRowToggle, onRerunTrip) {
  const rowEl = document.createElement("div");
  rowEl.className = `trip-row-card ${row.isExpanded ? "expanded" : ""} ${row.status}`;
  rowEl.id = `row-${row.trip_id}`;

  const res = row.result;
  const isDone = row.status === "done" && res;
  const isFailed = row.status === "failed";
  const isRunning = row.status === "running";
  const isQueued = row.status === "queued";

  // Headline metrics
  let top1Display = "-";
  let deltaPill = "";
  let continuityDisplay = "-";
  let runtimeDisplay = "-";

  if (isDone) {
    const m = res.metrics;
    const clfM = m.classifier;
    const hmmM = m.hmm;
    const d = m.delta;

    if (res.hmm_used && hmmM) {
      top1Display = formatPct(hmmM.top1_all);
      continuityDisplay = formatPct(hmmM.route_continuity);
      if (d) {
        const sign = d.top1_all > 0 ? "+" : "";
        const pClass = d.top1_all > 0 ? "delta-pill positive" : d.top1_all < 0 ? "delta-pill negative" : "delta-pill neutral";
        deltaPill = `<span class="${pClass}">${sign}${(d.top1_all * 100).toFixed(1)}%</span>`;
      }
    } else {
      top1Display = formatPct(clfM.top1_all);
      continuityDisplay = formatPct(clfM.route_continuity);
      deltaPill = `<span class="badge badge-amber">classifier only</span>`;
    }
    runtimeDisplay = `${res.total_runtime_s}s`;
  }

  // Status chip
  let statusChip = "";
  if (isRunning) {
    statusChip = `<span class="chip chip-running"><span class="spinner-dot"></span> ${row.stage} (${row.stage_index + 1}/${row.n_stages})</span>`;
  } else if (isQueued) {
    statusChip = `<span class="chip chip-queued"><span class="dot"></span> Queued</span>`;
  } else if (isDone) {
    statusChip = `<span class="chip chip-done"><span class="dot"></span> Done</span>`;
  } else if (isFailed) {
    statusChip = `<span class="chip chip-failed"><span class="dot"></span> Failed</span>`;
  } else {
    statusChip = `<span class="chip chip-idle"><span class="dot"></span> Ready</span>`;
  }

  // Collapsed Header
  rowEl.innerHTML = `
    <button class="row-header" type="button" aria-expanded="${row.isExpanded}">
      <div class="row-left">
        <span class="trip-id-badge">${row.trip_id}</span>
        ${statusChip}
      </div>
      <div class="row-metrics">
        <div class="metric-col">
          <span class="metric-caption">Top-1 Accuracy</span>
          <span class="metric-val-lead">${top1Display} ${deltaPill}</span>
        </div>
        <div class="metric-col">
          <span class="metric-caption">Route Continuity</span>
          <span class="metric-val">${continuityDisplay}</span>
        </div>
        <div class="metric-col">
          <span class="metric-caption">Runtime</span>
          <span class="metric-val mono">${runtimeDisplay}</span>
        </div>
      </div>
      <div class="row-chevron">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
      </div>
    </button>
  `;

  // Attach accordion click handler
  const headerBtn = rowEl.querySelector(".row-header");
  headerBtn.addEventListener("click", () => onRowToggle(row.trip_id));

  // Expanded Content Panel
  if (row.isExpanded) {
    const bodyEl = document.createElement("div");
    bodyEl.className = "row-body";

    if (isFailed) {
      bodyEl.innerHTML = renderFailedPanel(row, onRerunTrip);
    } else if (isDone) {
      bodyEl.innerHTML = renderExpandedPanelContent(res, onRerunTrip);
      attachExpandedPanelEvents(bodyEl, res, onRerunTrip);
    } else if (isRunning) {
      bodyEl.innerHTML = `
        <div class="panel-loading">
          <div class="spinner-large"></div>
          <h4>Running Evaluation Pipeline for Trip ${row.trip_id}</h4>
          <p>Stage: <strong>${row.stage}</strong> (${row.stage_index + 1} of ${row.n_stages})</p>
          <div class="progress-bar-wrap">
            <div class="progress-bar-fill" style="width: ${((row.stage_index + 1) / row.n_stages) * 100}%"></div>
          </div>
        </div>
      `;
    } else {
      bodyEl.innerHTML = `
        <div class="panel-empty">
          <p>Trip not yet run. Click below to execute the evaluation pipeline.</p>
          <button class="btn btn-primary btn-sm btn-run-single" data-trip-id="${row.trip_id}">Run Trip ${row.trip_id}</button>
        </div>
      `;
      bodyEl.querySelector(".btn-run-single").addEventListener("click", () => onRerunTrip(row.trip_id));
    }

    rowEl.appendChild(bodyEl);
  }

  return rowEl;
}

function renderFailedPanel(row, onRerunTrip) {
  const err = row.error || { message: "Unknown pipeline error", code: "INTERNAL" };
  return `
    <div class="panel-error-alert">
      <div class="error-badge-icon">&#x26A0;</div>
      <div class="error-details">
        <h4>Pipeline Failed at stage: <code>${err.stage || "unknown"}</code></h4>
        <div class="error-code-line">Error Code: <strong>${err.code}</strong> ${err.request_id ? `| Request ID: <code>${err.request_id}</code>` : ""}</div>
        <p class="error-msg">${err.message}</p>
        ${err.hint ? `<div class="error-hint">&#x1F4A1; ${err.hint}</div>` : ""}
      </div>
      <div class="error-actions">
        <button class="btn btn-secondary btn-sm btn-retry-trip" data-trip-id="${row.trip_id}">Retry</button>
        <button class="btn btn-outline btn-sm btn-copy-error" data-error='${JSON.stringify(err)}'>Copy Error</button>
      </div>
    </div>
  `;
}

function renderExpandedPanelContent(res, onRerunTrip) {
  const m = res.metrics;
  const clf = m.classifier;
  const hmm = m.hmm;
  const d = m.delta;
  const errSum = res.error_summary || {};
  const dq = res.data_quality_context || {};
  const warnings = res.warnings || [];

  const headlineTop1 = res.hmm_used && hmm ? formatPct(hmm.top1_all) : formatPct(clf.top1_all);
  const headlineLabel = res.hmm_used ? "HMM Top-1 vs reference route" : "Classifier Top-1 vs reference route";

  let deltaHeadline = "";
  if (res.hmm_used && d) {
    const sign = d.top1_all > 0 ? "+" : "";
    const pClass = d.top1_all > 0 ? "delta-pill positive" : d.top1_all < 0 ? "delta-pill negative" : "delta-pill neutral";
    deltaHeadline = `<span class="${pClass}">${sign}${(d.top1_all * 100).toFixed(1)}% vs classifier</span>`;
  }

  // Downsample notice
  const downsampleNotice = res.display_downsample_step > 1
    ? `<span class="downsample-badge">Showing every ${res.display_downsample_step}th GPS point</span>`
    : "";

  return `
    <!-- a. Headline -->
    <div class="expanded-headline">
      <div class="headline-metric-box">
        <span class="headline-number">${headlineTop1}</span>
        ${deltaHeadline}
        <span class="headline-label">${headlineLabel}</span>
      </div>
      <div class="model-info-tag">
        <span>Model: <strong>${res.model.name} (${res.model.version})</strong></span>
        ${res.from_cache ? '<span class="badge badge-cache">cache hit</span>' : ""}
      </div>
    </div>

    <!-- b. Map -->
    <div class="map-card-container">
      <div class="map-toolbar">
        <span class="map-title">Route Visualization & GPS Diagnostics</span>
        ${downsampleNotice}
        <button class="btn btn-outline btn-xs btn-fit-map" data-trip-id="${res.trip_id}">Fit Bounds</button>
      </div>
      <div id="map-${res.trip_id}" class="trip-map-leaflet"></div>
    </div>

    <!-- c. Metric Cards -->
    <div class="metric-cards-section">
      <h5 class="section-title">Aggregate Metrics <span class="label-muted">(vs reference route)</span></h5>
      <div class="metric-cards-grid">
        <div class="metric-stat-card" title="Observation-level Top-1 accuracy over all GPS points">
          <span class="stat-title">Top-1 (All)</span>
          <span class="stat-val">${formatPct(res.hmm_used && hmm ? hmm.top1_all : clf.top1_all)}</span>
          <span class="stat-sub">Classifier: ${formatPct(clf.top1_all)}</span>
        </div>
        <div class="metric-stat-card" title="Top-1 accuracy over points whose reference arc is in the candidate set">
          <span class="stat-title">Top-1 (In Candidates)</span>
          <span class="stat-val">${formatPct(res.hmm_used && hmm ? hmm.top1_in_cands : clf.top1_in_cands)}</span>
          <span class="stat-sub">Classifier: ${formatPct(clf.top1_in_cands)}</span>
        </div>
        <div class="metric-stat-card" title="Top-1 accuracy on points with clean alignment labels">
          <span class="stat-title">Top-1 (Clean Points)</span>
          <span class="stat-val">${formatPct(res.hmm_used && hmm ? hmm.top1_clean : clf.top1_clean)}</span>
          <span class="stat-sub">Classifier: ${formatPct(clf.top1_clean)}</span>
        </div>
        <div class="metric-stat-card" title="Share of points where the reference arc was not in the candidate set">
          <span class="stat-title">Candidate Miss Rate</span>
          <span class="stat-val">${formatPct(clf.candidate_miss_rate)}</span>
          <span class="stat-sub">Upper bound: ${formatPct(1 - clf.candidate_miss_rate)}</span>
        </div>
        <div class="metric-stat-card" title="Share of consecutive predicted arcs connected in road network">
          <span class="stat-title">Route Continuity</span>
          <span class="stat-val">${formatPct(res.hmm_used && hmm ? hmm.route_continuity : clf.route_continuity)}</span>
          <span class="stat-sub">Classifier: ${formatPct(clf.route_continuity)}</span>
        </div>
        <div class="metric-stat-card" title="Arc precision against reference route">
          <span class="stat-title">Sequence Arc Precision</span>
          <span class="stat-val">${formatPct(res.hmm_used && hmm ? hmm.sequence_arc_precision : clf.sequence_arc_precision)}</span>
          <span class="stat-sub">Classifier: ${formatPct(clf.sequence_arc_precision)}</span>
        </div>
        <div class="metric-stat-card" title="Arc recall against reference route">
          <span class="stat-title">Sequence Arc Recall</span>
          <span class="stat-val">${formatPct(res.hmm_used && hmm ? hmm.sequence_arc_recall : clf.sequence_arc_recall)}</span>
          <span class="stat-sub">Classifier: ${formatPct(clf.sequence_arc_recall)}</span>
        </div>
        <div class="metric-stat-card" title="Arc F1 score against reference route">
          <span class="stat-title">Sequence Arc F1</span>
          <span class="stat-val">${formatFloat(res.hmm_used && hmm ? hmm.sequence_arc_f1 : clf.sequence_arc_f1)}</span>
          <span class="stat-sub">Classifier: ${formatFloat(clf.sequence_arc_f1)}</span>
        </div>
        <div class="metric-stat-card" title="Length-weighted route mismatch fraction vs reference route">
          <span class="stat-title">Mismatch Fraction</span>
          <span class="stat-val">${formatFloat(res.hmm_used && hmm ? hmm.sequence_length_weighted_mismatch : clf.sequence_length_weighted_mismatch)}</span>
          <span class="stat-sub">Classifier: ${formatFloat(clf.sequence_length_weighted_mismatch)}</span>
        </div>
      </div>
    </div>

    <!-- d. Classifier vs HMM Comparison Bars -->
    <div class="comparison-bars-section">
      <h5 class="section-title">Classifier vs HMM Comparison</h5>
      <div class="bars-grid">
        ${renderMetricComparisonBar("Top-1 (All Points)", clf.top1_all, hmm ? hmm.top1_all : null, true)}
        ${renderMetricComparisonBar("Top-1 (In Candidates)", clf.top1_in_cands, hmm ? hmm.top1_in_cands : null, true)}
        ${renderMetricComparisonBar("Top-1 (Clean Points)", clf.top1_clean, hmm ? hmm.top1_clean : null, true)}
        ${renderMetricComparisonBar("Route Continuity", clf.route_continuity, hmm ? hmm.route_continuity : null, true)}
        ${renderMetricComparisonBar("Sequence Arc F1", clf.sequence_arc_f1, hmm ? hmm.sequence_arc_f1 : null, false)}
        ${renderMetricComparisonBar("Route Mismatch Fraction", clf.sequence_length_weighted_mismatch, hmm ? hmm.sequence_length_weighted_mismatch : null, false, true)}
      </div>
    </div>

    <!-- e. Error Summary Table -->
    <div class="error-summary-section">
      <h5 class="section-title">Error & Transition Summary <span class="label-muted">(Aggregates Only)</span></h5>
      <div class="table-responsive">
        <table class="data-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Classifier Only</th>
              <th>HMM Decoder</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Correct Points</td>
              <td>${errSum.n_correct ? errSum.n_correct.classifier : "-"} / ${errSum.n_points || "-"}</td>
              <td>${errSum.n_correct && errSum.n_correct.hmm != null ? errSum.n_correct.hmm : "-"} / ${errSum.n_points || "-"}</td>
              <td>Matches reference label arc</td>
            </tr>
            <tr>
              <td>Incorrect Points</td>
              <td>${errSum.n_incorrect ? errSum.n_incorrect.classifier : "-"}</td>
              <td>${errSum.n_incorrect && errSum.n_incorrect.hmm != null ? errSum.n_incorrect.hmm : "-"}</td>
              <td>Differs from reference label</td>
            </tr>
            <tr>
              <td>Broken Transitions</td>
              <td>${errSum.broken_transitions ? errSum.broken_transitions.classifier : "-"}</td>
              <td>${errSum.broken_transitions && errSum.broken_transitions.hmm != null ? errSum.broken_transitions.hmm : "-"}</td>
              <td>Disconnected consecutive arcs</td>
            </tr>
            <tr>
              <td>HMM Fallbacks</td>
              <td>-</td>
              <td>${errSum.hmm_fallbacks ? `${errSum.hmm_fallbacks.unreachable_candidates} unreachable, ${errSum.hmm_fallbacks.long_gaps} gaps` : "-"}</td>
              <td>Transitions where fallback applied</td>
            </tr>
            <tr>
              <td>Points Changed by HMM</td>
              <td>-</td>
              <td>${errSum.n_points_hmm_changed != null ? errSum.n_points_hmm_changed : "-"}</td>
              <td>Points where HMM picked different arc</td>
            </tr>
            <tr>
              <td>Label Quality Breakdown</td>
              <td colspan="3">
                ${errSum.label_quality_share ? `Clean: ${(errSum.label_quality_share.clean * 100).toFixed(1)}% | Ambiguous: ${(errSum.label_quality_share.ambiguous * 100).toFixed(1)}% | Far: ${(errSum.label_quality_share.far * 100).toFixed(1)}%` : "-"}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- f. Stage Timing Strip -->
    <div class="stage-timing-section">
      <h5 class="section-title">Stage Timings <span class="label-muted">(Total: ${res.total_runtime_s}s)</span></h5>
      <div class="timings-strip">
        ${Object.entries(res.timings_s || {}).map(([stage, sec]) => `
          <div class="timing-box">
            <span class="timing-stage">${stage}</span>
            <span class="timing-val">${sec.toFixed(3)}s</span>
          </div>
        `).join("")}
      </div>
    </div>

    <!-- g. Data Quality Context & Warnings -->
    <div class="context-warnings-section">
      <div class="data-quality-pills">
        <span class="dq-pill">Mean candidates/point: <strong>${dq.n_candidates_mean}</strong> (max ${dq.n_candidates_max})</span>
        <span class="dq-pill">After long gap: <strong>${(dq.share_after_long_gap * 100).toFixed(1)}%</strong></span>
        <span class="dq-pill">Stationary fixes: <strong>${(dq.share_stationary * 100).toFixed(1)}%</strong></span>
      </div>
      ${warnings.length > 0 ? `
        <div class="warnings-box">
          ${warnings.map((w) => `<div class="warning-item">&#x26A0; ${w}</div>`).join("")}
        </div>
      ` : ""}
    </div>

    <!-- h. Action Buttons -->
    <div class="panel-actions-footer">
      <button class="btn btn-secondary btn-sm btn-rerun-this" data-trip-id="${res.trip_id}">Re-run Trip</button>
      <button class="btn btn-outline btn-sm btn-copy-json" data-trip-id="${res.trip_id}">Copy Result JSON</button>
      <button class="btn btn-outline btn-sm btn-download-json" data-trip-id="${res.trip_id}">Download Result JSON</button>
    </div>
  `;
}

function attachExpandedPanelEvents(bodyEl, res, onRerunTrip) {
  // Re-run
  const rerunBtn = bodyEl.querySelector(".btn-rerun-this");
  if (rerunBtn) {
    rerunBtn.addEventListener("click", () => onRerunTrip(res.trip_id));
  }

  // Fit Map
  const fitBtn = bodyEl.querySelector(".btn-fit-map");
  if (fitBtn) {
    fitBtn.addEventListener("click", () => {
      const tripMap = activeMaps.get(res.trip_id);
      if (tripMap) tripMap.fitBounds();
    });
  }

  // Copy JSON
  const copyBtn = bodyEl.querySelector(".btn-copy-json");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => copyToClipboard(JSON.stringify(res, null, 2)));
  }

  // Download JSON
  const dlBtn = bodyEl.querySelector(".btn-download-json");
  if (dlBtn) {
    dlBtn.addEventListener("click", () => downloadJson(res, `pathsnap_${res.trip_id}_result.json`));
  }

  // Error retry
  const retryBtn = bodyEl.querySelector(".btn-retry-trip");
  if (retryBtn) {
    retryBtn.addEventListener("click", () => onRerunTrip(res.trip_id));
  }
}
