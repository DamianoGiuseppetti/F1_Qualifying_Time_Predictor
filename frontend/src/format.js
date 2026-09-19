// Display-only time formatting. Ported 1:1 (Aug 26 2026, then carried
// into the Aug 30 2026 React rebuild unchanged) from
// dashboard/formatting.py, which stays the dependency-free Python
// source of truth for the sourcing/legibility notes - see that file's
// docstring for why Predicted/Official render at 3 decimals (mm:ss.ddd)
// while the Typical range column's low/high bounds stay at 1 decimal
// (mm:ss.d): Damiano's Aug 26 2026 design-review requirement. One
// function, one rounding/carry implementation, two call sites choosing
// precision via the `decimals` argument.

export function formatMmss(seconds, decimals = 3) {
  if (seconds == null || Number.isNaN(seconds) || seconds < 0) {
    return "--:--." + "-".repeat(decimals);
  }
  const unit = 10 ** decimals;
  const totalUnits = Math.round(seconds * unit);
  const minutes = Math.floor(totalUnits / (60 * unit));
  const remUnits = totalUnits - minutes * 60 * unit;
  const secs = Math.floor(remUnits / unit);
  const frac = remUnits - secs * unit;
  const mm = String(minutes).padStart(2, "0");
  const ss = String(secs).padStart(2, "0");
  const ff = String(frac).padStart(decimals, "0");
  return `${mm}:${ss}.${ff}`;
}

// 0.041 -> "0.041s" (or "+0.041s" with signed=true) - the plain-decimal
// "s.ddd" format for a margin/error/gap (never mm:ss). "--" for null/NaN.
export function formatDelta(seconds, signed = false) {
  if (seconds == null || Number.isNaN(seconds)) return "--";
  if (signed) {
    const sign = seconds >= 0 ? "+" : "-";
    return `${sign}${Math.abs(seconds).toFixed(3)}s`;
  }
  return `${Math.abs(seconds).toFixed(3)}s`;
}

// Sep 2026 addition: display label for a background job's current
// `stage` (see f1qp.serving.data_fetch) - shared by the Prediction tab's
// data-fetch flow and the History tab's "Check for official result".
export function formatStage(stage) {
  const labels = {
    queued: "Queued…",
    downloading: "Downloading practice laps…",
    building_features: "Building features…",
    extracting_results: "Fetching official result…",
    done: "Done.",
  };
  return labels[stage] || stage;
}

// "2026-08-25T21:20:00+00:00" -> "25 AUG 2026 21:20 UTC"
export function formatTimestamp(iso) {
  if (!iso) return "--";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const day = String(d.getUTCDate()).padStart(2, "0");
    const month = d.toLocaleString("en-US", { month: "short", timeZone: "UTC" }).toUpperCase();
    const year = d.getUTCFullYear();
    const hh = String(d.getUTCHours()).padStart(2, "0");
    const mm = String(d.getUTCMinutes()).padStart(2, "0");
    return `${day} ${month} ${year} ${hh}:${mm} UTC`;
  } catch (e) {
    return iso;
  }
}
