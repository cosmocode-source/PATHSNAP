/**
 * PathSnap API client with ngrok interstitial skip and structured error handling.
 */
import { getSavedApiUrl } from "../config.js";

export class ApiError extends Error {
  constructor(message, status, code, stage, hint) {
    super(message);
    this.status = status;
    this.code = code || "INTERNAL";
    this.stage = stage || "";
    this.hint = hint || "";
  }
}

// Global hook for logging UI requests into the activity drawer
let requestLogListener = null;

export function setRequestLogListener(fn) {
  requestLogListener = fn;
}

export async function request(endpoint, options = {}) {
  const baseUrl = getSavedApiUrl().replace(/\/+$/, "");
  const url = `${baseUrl}${endpoint.startsWith("/") ? endpoint : "/" + endpoint}`;
  const method = options.method || "GET";

  const headers = {
    "ngrok-skip-browser-warning": "true",
    "Accept": "application/json",
    ...(options.headers || {}),
  };

  if (options.body && typeof options.body === "object" && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.body);
  }

  const timeoutMs = options.timeout || 25000;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  const t0 = performance.now();
  let status = 0;
  try {
    const res = await fetch(url, {
      ...options,
      headers,
      signal: controller.signal,
    });
    clearTimeout(id);
    status = res.status;
    const durationMs = Math.round(performance.now() - t0);

    // Notify activity drawer
    if (requestLogListener) {
      requestLogListener({
        time: new Date().toLocaleTimeString(),
        method,
        path: endpoint,
        status,
        durationMs,
        trip_id: options.trip_id || null,
      });
    }

    if (!res.ok) {
      let errorData = {};
      try {
        errorData = await res.json();
      } catch (e) {
        errorData = { message: res.statusText || `HTTP Error ${res.status}` };
      }
      const err = errorData.error || errorData;
      throw new ApiError(
        err.message || `Request failed with status ${res.status}`,
        res.status,
        err.code || "REQUEST_FAILED",
        err.stage || "",
        err.hint || ""
      );
    }

    return await res.json();
  } catch (err) {
    clearTimeout(id);
    const durationMs = Math.round(performance.now() - t0);
    if (requestLogListener) {
      requestLogListener({
        time: new Date().toLocaleTimeString(),
        method,
        path: endpoint,
        status: status || 0,
        durationMs,
        error: err.message,
        trip_id: options.trip_id || null,
      });
    }

    if (err instanceof ApiError) {
      throw err;
    }
    if (err.name === "AbortError") {
      throw new ApiError(`Request timeout after ${timeoutMs / 1000}s`, 504, "TIMEOUT", "", "Check backend connectivity.");
    }
    throw new ApiError(
      err.message || "Failed to connect to backend",
      status || 0,
      "NETWORK_ERROR",
      "",
      "Ensure backend is running and ngrok URL is valid."
    );
  }
}

export async function fetchHealth() {
  return await request("/api/health", { timeout: 8000 });
}

export async function fetchTrips() {
  return await request("/api/trips", { timeout: 10000 });
}

export async function postPredict(tripId, refresh = false) {
  return await request("/api/predict", {
    method: "POST",
    body: { trip_id: tripId, refresh },
    trip_id: tripId,
  });
}

export async function fetchJob(jobId) {
  return await request(`/api/jobs/${jobId}`, { timeout: 10000 });
}

export async function fetchBackendLogs(limit = 50) {
  return await request(`/api/logs?limit=${limit}`, { timeout: 5000 });
}
