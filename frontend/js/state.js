/**
 * Application State management for PathSnap Evaluator.
 */
import { getSavedApiUrl, saveApiUrl } from "../config.js";

class AppState {
  constructor() {
    this.apiUrl = getSavedApiUrl();
    this.connectionStatus = "disconnected"; // disconnected | connected_full | connected_classifier | unreachable
    this.healthData = null;
    this.trips = [];
    this.selectedTripId = "";
    this.skipCache = false;
    
    // Map: trip_id -> row state
    // row state: { status, stage, stage_index, n_stages, elapsed_s, result, error, isExpanded }
    this.tripRows = new Map();

    // Batch run state
    this.isRunningAll = false;
    this.runAllCancelled = false;
    this.runAllProgress = { current: 0, total: 0 };

    // Log feeds
    this.uiLogs = [];
    this.backendLogs = [];

    // Subscribed render callbacks
    this.listeners = new Set();
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify(eventType, payload) {
    for (const listener of this.listeners) {
      try {
        listener(eventType, payload, this);
      } catch (err) {
        console.error("State listener error:", err);
      }
    }
  }

  setApiUrl(url) {
    this.apiUrl = url;
    saveApiUrl(url);
    this.notify("apiUrlChanged", url);
  }

  setConnection(status, healthData = null) {
    this.connectionStatus = status;
    this.healthData = healthData;
    this.notify("connectionChanged", { status, healthData });
  }

  setTrips(tripsList) {
    this.trips = tripsList;
    if (tripsList.length && !this.selectedTripId) {
      this.selectedTripId = tripsList[0].trip_id;
    }
    // Initialize row states if not present
    for (const trip of tripsList) {
      if (!this.tripRows.has(trip.trip_id)) {
        this.tripRows.set(trip.trip_id, {
          trip_id: trip.trip_id,
          status: "idle", // idle | queued | running | done | failed
          stage: "ready",
          stage_index: 0,
          n_stages: 7,
          elapsed_s: 0,
          result: null,
          error: null,
          isExpanded: false,
        });
      }
    }
    this.notify("tripsLoaded", tripsList);
  }

  updateTripRow(tripId, updates) {
    const existing = this.tripRows.get(tripId) || {
      trip_id: tripId,
      status: "idle",
      stage: "",
      stage_index: 0,
      n_stages: 7,
      elapsed_s: 0,
      result: null,
      error: null,
      isExpanded: false,
    };
    const merged = { ...existing, ...updates };
    this.tripRows.set(tripId, merged);
    this.notify("tripRowUpdated", merged);
  }

  toggleRowExpansion(tripId, forceState = null) {
    const row = this.tripRows.get(tripId);
    if (!row) return;
    const isExpanded = forceState !== null ? forceState : !row.isExpanded;
    row.isExpanded = isExpanded;
    this.notify("rowExpansionChanged", { tripId, isExpanded });
  }

  addUiLog(entry) {
    this.uiLogs.unshift(entry);
    if (this.uiLogs.length > 100) this.uiLogs.pop();
    this.notify("uiLogsUpdated", this.uiLogs);
  }

  setBackendLogs(logs) {
    this.backendLogs = logs;
    this.notify("backendLogsUpdated", this.backendLogs);
  }

  // Summary statistics across completed trips
  getSummaryStats() {
    const finishedRows = Array.from(this.tripRows.values()).filter(
      (r) => r.status === "done" && r.result && r.result.metrics
    );
    if (!finishedRows.length) return null;

    const nDone = finishedRows.length;
    let totalTime = 0;
    const totals = {
      top1_all: { clf: 0, hmm: 0 },
      top1_in_cands: { clf: 0, hmm: 0 },
      top1_clean: { clf: 0, hmm: 0 },
      route_continuity: { clf: 0, hmm: 0 },
      sequence_arc_f1: { clf: 0, hmm: 0 },
      sequence_length_weighted_mismatch: { clf: 0, hmm: 0 },
    };

    let hmmCount = 0;

    for (const r of finishedRows) {
      totalTime += r.result.total_runtime_s || 0;
      const m = r.result.metrics;
      const c = m.classifier;
      const h = m.hmm;

      for (const k of Object.keys(totals)) {
        totals[k].clf += c[k] || 0;
        if (h && typeof h[k] === "number" && !isNaN(h[k])) {
          totals[k].hmm += h[k];
        }
      }
      if (h) hmmCount++;
    }

    const averages = {};
    for (const k of Object.keys(totals)) {
      averages[k] = {
        classifier: totals[k].clf / nDone,
        hmm: hmmCount ? totals[k].hmm / hmmCount : null,
        delta: hmmCount ? (totals[k].hmm / hmmCount) - (totals[k].clf / nDone) : null,
      };
    }

    return {
      nDone,
      totalTrips: this.trips.length,
      totalTime: totalTime.toFixed(2),
      averages,
      hasHmm: hmmCount > 0,
    };
  }
}

export const state = new AppState();
