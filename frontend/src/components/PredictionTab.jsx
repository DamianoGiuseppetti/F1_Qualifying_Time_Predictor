import { useEffect, useRef, useState } from "react";
import { fetchJSON, pollJob } from "../api.js";
import { formatStage, formatTimestamp } from "../format.js";
import { PredictionsTable, ExcludedNote } from "./PredictionsTable.jsx";

// Aug 30 2026 (Damiano): "the predictions of 2026 already done must be
// in the history not in the prediction section" - this tab never
// remembers or auto-loads any past launch (the old "switcher" between
// the two most recently launched rounds is gone for good, carried over
// unchanged from the vanilla-JS frontend into this React rebuild). It
// only ever reflects the ONE action just taken in this visit - Preview
// or Launch - and starts from a blank slate on every page load. Every
// already-launched round lives in the History tab, full stop.
//
// Default round: bumped to 15 (Sep 18 2026) - Round 14 is the latest
// one already launched (see App.jsx, now defaulting to the History tab
// for exactly that round), so the Prediction tab's own default should
// point at the next round nobody has launched yet, same reasoning as
// the original Aug 30 2026 choice of 12 - just carried forward instead
// of left stale.
//
// Sep 2026 addition (Damiano: "I don't want to run the py scripts
// outside everytime. The app should be able to do everything"): a
// missing round no longer just shows a "run these scripts" error message
// - it offers to fetch the data itself. Because a fetch started too soon
// after practice can silently find nothing or an incomplete session
// (Damiano: "we need to be sure that practices are over with a temporal
// range of security"), a 404 first checks /data/readiness and shows that
// warning before asking to confirm, rather than fetching immediately.
//
// Sep 13 2026 addition (point 4 of the "final version of the
// application" plan): `canWrite` comes from App.jsx (true when this
// deployment has no admin gating at all, or when this browser's admin
// token checks out - see its own comments). Preview stays available
// regardless; Launch and the fetch-now flow below both check it.
export default function PredictionTab({ active, onLaunched, canWrite }) {
  const [season, setSeason] = useState(2026);
  const [round, setRound] = useState(15);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [previewPayload, setPreviewPayload] = useState(null);
  const [launchedPayload, setLaunchedPayload] = useState(null);
  // null | {phase:"confirm", launch, readiness} | {phase:"fetching", launch, stage} | {phase:"error", launch, message}
  const [fetchState, setFetchState] = useState(null);
  const [dots, setDots] = useState("");

  useEffect(() => {
    if (!fetchState || fetchState.phase !== "fetching") return;
    const id = setInterval(() => setDots((d) => (d.length >= 3 ? "" : d + ".")), 450);
    return () => clearInterval(id);
  }, [fetchState]);

  async function handle(launch) {
    if (!season || !round) return;
    setBusy(true);
    setFeedback("");
    setFetchState(null);
    await runPredict(launch);
    setBusy(false);
  }

  async function runPredict(launch) {
    const endpoint = launch ? `/predict/${season}/${round}/launch` : `/predict/${season}/${round}`;
    try {
      const payload = await fetchJSON(endpoint, { method: "POST" });
      if (launch) {
        setPreviewPayload(null);
        setLaunchedPayload(payload);
        setFeedback(`Launched ${season} Round ${round}.`);
        // Already-launched rounds only ever live in History/Performance
        // from now on - tell the parent so it can invalidate both tabs'
        // caches, matching the old historyLoaded/performanceLoaded reset.
        onLaunched();
      } else {
        setLaunchedPayload(null);
        setPreviewPayload(payload);
      }
      setFetchState(null);
    } catch (e) {
      if (e.status === 404 && canWrite) {
        const readiness = await fetchJSON(`/data/readiness/${season}/${round}`).catch(() => null);
        setFetchState({ phase: "confirm", launch, readiness });
      } else if (e.status === 404) {
        // Guest (or an admin token that just expired) - no fetch-now
        // offer, since /data/fetch would 401 anyway (see require_admin
        // in f1qp.api.main). Just say plainly that it isn't there yet.
        setFeedback(`No data for ${season} Round ${round} yet - check back once practice sessions are done.`);
      } else if (e.status === 401) {
        setFeedback("Admin token missing or incorrect - unlock it via the GUEST/ADMIN chip in the status bar first.");
      } else if (e.status === 503) {
        setFeedback(`Model not loaded: ${e.message}`);
      } else {
        setFeedback(`Request failed: ${e.message}`);
      }
    }
  }

  async function startFetch(launch) {
    setFetchState({ phase: "fetching", launch, stage: "queued" });
    setBusy(true);
    try {
      const { job_id } = await fetchJSON(`/data/fetch/${season}/${round}`, { method: "POST" });
      const final = await pollJob(job_id, (status) =>
        setFetchState({ phase: "fetching", launch, stage: status.stage })
      );
      if (final.status === "error") {
        setFetchState({ phase: "error", launch, message: final.error || "Fetch failed." });
        setBusy(false);
        return;
      }
      // Data's there now - retry the original Preview/Launch automatically.
      await runPredict(launch);
      setBusy(false);
    } catch (e) {
      setFetchState({ phase: "error", launch, message: e.message });
      setBusy(false);
    }
  }

  function eventTitle(payload) {
    return payload.event_name
      ? `${payload.year} Round ${payload.round_number} — ${payload.event_name}`
      : `${payload.year} Round ${payload.round_number}`;
  }

  return (
    <section className={`view${active ? " active" : ""}`} id="view-prediction">
      <div className="panel launch-card">
        <div className="section-head">
          <div className="section-label">Launch a prediction</div>
          <div className="section-title">Run the model for one weekend</div>
        </div>
        <div className="launch-row">
          <div className="field">
            <div className="field-label">Season</div>
            <input
              className="field-box"
              type="number"
              // Sep 16 2026 (Damiano: "give the opportunity to launch
              // preview only on 2026 and not before"): 2023-2025 are
              // training seasons (see f1qp.config.TRAINING_SEASONS) -
              // "predicting" one of those through the live app isn't a
              // real prediction, so the input itself no longer offers
              // them. The backend enforces the same floor independently
              // (see main.py's _require_live_season) for anyone calling
              // the API directly instead of through this form.
              min="2026"
              max="2030"
              value={season}
              onChange={(e) => setSeason(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <div className="field-label">Round</div>
            <input
              className="field-box"
              type="number"
              min="1"
              max="24"
              value={round}
              onChange={(e) => setRound(Number(e.target.value))}
            />
          </div>
          <div className="spacer"></div>
          <button className="launch-btn launch-btn--ghost" type="button" disabled={busy} onClick={() => handle(false)}>
            Preview (not saved)
          </button>
          <button
            className="launch-btn"
            type="button"
            disabled={busy || !canWrite}
            title={canWrite ? undefined : "Admin only - unlock via the GUEST/ADMIN chip in the status bar to Launch"}
            onClick={() => handle(true)}
          >
            Launch Prediction
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4">
              <path d="M5 12h14"></path>
              <path d="M13 6l6 6-6 6"></path>
            </svg>
          </button>
        </div>
        <div className="run-meta">{feedback}</div>
      </div>

      {fetchState && fetchState.phase === "confirm" && (
        <div className="panel launch-card">
          <div className="section-head">
            <div className="section-label">No data yet for {season} Round {round}</div>
            <div className="section-title">Fetch it now?</div>
          </div>
          <p className="placeholder-note">
            {fetchState.readiness
              ? fetchState.readiness.message
              : "Couldn't check session timing, but this round's data hasn't been downloaded yet."}
          </p>
          <div className="launch-row">
            <div className="spacer"></div>
            <button className="launch-btn launch-btn--ghost" type="button" onClick={() => setFetchState(null)}>
              Wait
            </button>
            <button className="launch-btn" type="button" onClick={() => startFetch(fetchState.launch)}>
              Fetch data now
            </button>
          </div>
        </div>
      )}

      {fetchState && fetchState.phase === "fetching" && (
        <div className="panel launch-card">
          <div className="section-head">
            <div className="section-label">Fetching {season} Round {round}</div>
            <div className="section-title">
              {formatStage(fetchState.stage)}
              {dots}
            </div>
          </div>
          <p className="placeholder-note">
            Downloading practice laps and rebuilding features - this can take up to a couple of minutes. The result
            will predict automatically once it's ready.
          </p>
        </div>
      )}

      {fetchState && fetchState.phase === "error" && (
        <div className="panel launch-card">
          <div className="section-head">
            <div className="section-label">Fetch failed</div>
            <div className="section-title">{fetchState.message}</div>
          </div>
          <div className="launch-row">
            <div className="spacer"></div>
            <button className="launch-btn launch-btn--ghost" type="button" onClick={() => setFetchState(null)}>
              Dismiss
            </button>
            <button className="launch-btn" type="button" onClick={() => startFetch(fetchState.launch)}>
              Try again
            </button>
          </div>
        </div>
      )}

      <div className="panel results">
        {previewPayload ? (
          <>
            <div className="results-head">
              <div className="pending-tag">PREVIEW &mdash; NOT SAVED</div>
              <div className="results-title">{eventTitle(previewPayload)}</div>
              <div className="results-meta">{previewPayload.predictions.length} drivers scored</div>
            </div>
            <ExcludedNote excluded={previewPayload.excluded_test_drivers} />
            <PredictionsTable
              rows={previewPayload.predictions}
              year={previewPayload.year}
              roundNumber={previewPayload.round_number}
              scored={previewPayload.predictions.some((p) => p.official_results_available)}
            />
          </>
        ) : launchedPayload ? (
          <>
            <div className="results-head">
              <div className="badge badge--good">LAUNCHED</div>
              <div className="results-title">{eventTitle(launchedPayload)}</div>
              <div className="results-meta">
                {launchedPayload.predictions.length} drivers &middot; launched{" "}
                {formatTimestamp(launchedPayload.launched_at_utc)}
              </div>
            </div>
            <ExcludedNote excluded={launchedPayload.excluded_test_drivers} />
            <PredictionsTable
              rows={launchedPayload.predictions}
              year={launchedPayload.year}
              roundNumber={launchedPayload.round_number}
              scored={launchedPayload.predictions.some((p) => p.official_results_available)}
            />
            <p className="placeholder-note">
              Saved &mdash; find it in the History tab from now on, scored automatically once the official result is
              in.
            </p>
          </>
        ) : (
          <p className="placeholder-note">
            Enter a season and round above, then Preview or Launch Prediction. Every already-launched round lives in
            the History tab, not here.
          </p>
        )}
      </div>
    </section>
  );
}
