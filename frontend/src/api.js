// Tiny fetch wrapper shared by every tab - calls the SAME FastAPI service
// that serves this bundle, at the same origin (no base-URL config, no
// CORS, no Docker container-networking gotcha - same reasoning as the
// Aug 26 2026 vanilla-JS frontend this replaces). Throws an Error with a
// `.status` property on a non-2xx response so callers can branch on
// e.g. 404 vs 503 the way the old frontend/app.js did.

// Sep 13 2026 addition (point 4 of the "final version of the
// application" plan): the admin token, if the person has unlocked one
// via the Settings panel (see StatusStrip.jsx), lives in THIS browser's
// localStorage only - never sent anywhere but this same-origin API, never
// synced, never visible to anyone else viewing this deployment. Reading
// it here (rather than threading it through every caller) means every
// existing fetchJSON call site keeps working unchanged - the token just
// rides along automatically once one is saved, and gated routes ignore
// it entirely when the backend has no ADMIN_TOKEN configured.
const ADMIN_TOKEN_KEY = "f1qp_admin_token";

// localStorage can throw (private browsing, blocked site data) - never
// let a storage hiccup break the app itself, same spirit as every other
// best-effort boundary in this project.
export function getAdminToken() {
  try {
    return localStorage.getItem(ADMIN_TOKEN_KEY) || "";
  } catch (e) {
    return "";
  }
}

export function setAdminToken(token) {
  try {
    if (token) {
      localStorage.setItem(ADMIN_TOKEN_KEY, token);
    } else {
      localStorage.removeItem(ADMIN_TOKEN_KEY);
    }
  } catch (e) {
    /* token just won't persist across reloads - not fatal */
  }
}

export async function fetchJSON(path, opts) {
  const token = getAdminToken();
  const headers = { ...(opts && opts.headers) };
  if (token) headers["X-Admin-Token"] = token;

  const res = await fetch(path, { ...opts, headers });
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body.detail || "";
    } catch (e) {
      /* non-JSON error body - fall through with empty detail */
    }
    const err = new Error(detail || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// Polls GET /data/jobs/{jobId} (a background download/build-features/
// extract-results job - see f1qp.serving.data_fetch) every `intervalMs`
// until it reports "done" or "error". Shared by the Prediction tab's
// data-fetch flow and the History tab's "Check for official result"
// button (Sep 2026, Damiano: "the app should be able to do everything"
// instead of running scripts by hand). `onUpdate`, if given, is called
// with each raw status payload so the caller can show the current stage
// while polling; resolves with the final payload once finished.
export async function pollJob(jobId, onUpdate, intervalMs = 2000) {
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const status = await fetchJSON(`/data/jobs/${jobId}`);
    if (onUpdate) onUpdate(status);
    if (status.status === "done" || status.status === "error") {
      return status;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
