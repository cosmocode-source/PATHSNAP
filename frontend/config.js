/**
 * PathSnap Evaluator UI Configuration.
 */

// Default backend URL (overridden by localStorage or UI input)
export const DEFAULT_API_URL = "https://febriferous-unfudged-antonietta.ngrok-free.dev";

// Storage key for remembered URL
export const STORAGE_KEY_API_URL = "pathsnap_api_url";

export function getSavedApiUrl() {
  return localStorage.getItem(STORAGE_KEY_API_URL) || DEFAULT_API_URL;
}

export function saveApiUrl(url) {
  if (url) {
    localStorage.setItem(STORAGE_KEY_API_URL, url.replace(/\/+$/, ""));
  }
}
