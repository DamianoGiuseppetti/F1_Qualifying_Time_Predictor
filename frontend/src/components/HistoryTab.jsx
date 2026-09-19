import { useCallback, useEffect, useRef, useState } from "react";
import { fetchJSON, pollJob } from "../api.js";
import { driverIdentity, roundOverridesFor } from "../teamColors.js";
import { formatDelta, formatStage } from "../format.js";
import { PredictionsTable } from "./PredictionsTable.jsx";

// `/history` itself already only ever returns MIN_SEASON (2026)-onward
// rows (src/f1qp/serving/history.py) - nothing to filter client-side.
//
// Sep 13 2026 addition: `canWrite` (see App.jsx / PredictionTab.jsx's own
// comments) - viewing History stays open to everyone; only the "Check
// for official result" action below is restricted.
export default function HistoryTab({ active, dataVersion, canWrite }) {
  const [rows, setRows] = useState([]);
  const [selectedYear, setSelectedYear] = useState(null);
  const [selectedRound, setSelectedRound] = useState(null);
  // -1 never equals a real dataVersion, so the first time this tab is
  // opened it always fetches; after that it only re-fetches when
  // dataVersion changes (bumped by PredictionTab after a launch) - same
  // "load once, invalidate on launch" behavior as the old
  // historyLoaded/historyLoaded=false pattern, without needing a
  // separate loaded flag.
  const loadedVersionRef = useRef(-1);

  // Sep 2026: pulled out of the effect below so the "Check for official
  // result" action (in HistoryBody) can re-fetch /history itself once its
  // background job finishes, without waiting for dataVersion to change -
  // that prop only ever gets bumped by a Launch on the Prediction tab.
  const loadHistory = useCallback(async (keepSelection) => {
    let fetched = [];
    try {
      const resp = await fetchJSON("/history");
      fetched = resp.rows || [];
    } catch (e) {
      fetched = [];
    }
    setRows(fetched);
    if (!fetched.length) {
      setSelectedYear(null);
      setSelectedRound(null);
      return;
    }
    if (keepSelection && selectedYear != null && selectedRound != null) {
      return; // rows already updated above; keep the round the user is looking at
    }
    const years = [...new Set(fetched.map((r) => r.year))].sort((a, b) => b - a);
    const year = years[0];
    const roundsForYear = [...new Set(fetched.filter((r) => r.year === year).map((r) => r.round_number))].sort(
      (a, b) => a - b
    );
    setSelectedYear(year);
    setSelectedRound(roundsForYear[roundsForYear.length - 1]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedYear, selectedRound]);

  useEffect(() => {
    if (!active) return;
    if (loadedVersionRef.current === dataVersion) return;
    loadedVersionRef.current = dataVersion;
    let cancelled = false;
    (async () => {
      if (cancelled) return;
      await loadHistory(false);
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, dataVersion]);

  return (
    <section className={`view${active ? " active" : ""}`} id="view-history">
      {!rows.length ? (
        <p className="placeholder-note">
          No launched rounds yet. Use the Prediction tab's Launch Prediction button &mdash; each launch adds one
          entry here, scored automatically once the official result is available.
        </p>
      ) : (
        <HistoryBody
          rows={rows}
          selectedYear={selectedYear}
          selectedRound={selectedRound}
          onSelectYear={setSelectedYear}
          onSelectRound={setSelectedRound}
          onResultsFetched={() => loadHistory(true)}
          canWrite={canWrite}
        />
      )}
    </section>
  );
}

function HistoryBody({ rows, selectedYear, selectedRound, onSelectYear, onSelectRound, onResultsFetched, canWrite }) {
  const years = [...new Set(rows.map((r) => r.year))].sort((a, b) => b - a);
  const rounds = [...new Set(rows.filter((r) => r.year === selectedYear).map((r) => r.round_number))].sort(
    (a, b) => a - b
  );
  // Sep 16 2026 fix: `official_results_available` (round-level - true
  // once every driver has an official-result ROW, whether or not that
  // row itself has a time) is what decides "scored", NOT
  // `has_target` (per-driver - false forever for a driver who DNS'd/
  // DSQ'd/crashed with no lap time, e.g. Round 14's BEA/STR). The old
  // `every(r => r.has_target)` meant a round with even one such driver
  // could never flip out of "Awaiting official result" no matter how
  // many times "Check for official result" succeeded - see
  // f1qp.serving.history.is_scored's docstring.
  const roundHasScore = new Set(
    rows.filter((r) => r.official_results_available).map((r) => `${r.year}:${r.round_number}`)
  );
  const roundRows = rows.filter((r) => r.year === selectedYear && r.round_number === selectedRound);
  const scored = roundRows.length > 0 && roundRows[0].official_results_available;
  // Aug 30 2026 (Damiano): "Add the name of the GP not only the round" -
  // every row for a round already carries the same event_name (see
  // src/f1qp/serving/history.py's prediction_history()), so any row's
  // value works; None on a schedule-lookup miss just shows nothing extra.
  const eventName = roundRows[0]?.event_name;

  // Sep 2026 (Damiano: "the app should be able to do everything"):
  // null | {phase:"confirm"} | {phase:"fetching", stage} | {phase:"error", message}
  // | {phase:"warning", message}
  // No readiness pre-check here on purpose (unlike the Prediction tab's
  // practice-data fetch) - see f1qp.config.session_readiness's docstring
  // on why guessing the Qualifying session's timing on a sprint weekend
  // is more likely to be confidently wrong than simply absent.
  //
  // Sep 16 2026 addition: "warning" - the official result itself fetched
  // fine, but the automatic rebuild-dataset/retrain chain
  // f1qp.serving.data_fetch.start_results_job now runs afterward hit a
  // problem (e.g. a broken local torch install). Deliberately NOT the
  // same as "error" - the thing this button promised (the official
  // result) did land; the retrain chain is a best-effort bonus riding
  // along on top of it, surfaced here so it's still visible somewhere
  // rather than silently lost (see that function's own docstring).
  const [resultsFetch, setResultsFetch] = useState(null);
  const [dots, setDots] = useState("");

  useEffect(() => {
    if (!resultsFetch || resultsFetch.phase !== "fetching") return;
    const id = setInterval(() => setDots((d) => (d.length >= 3 ? "" : d + ".")), 450);
    return () => clearInterval(id);
  }, [resultsFetch]);

  async function startResultsFetch() {
    setResultsFetch({ phase: "fetching", stage: "queued" });
    try {
      const { job_id } = await fetchJSON(`/data/fetch-results/${selectedYear}/${selectedRound}`, { method: "POST" });
      const final = await pollJob(job_id, (status) => setResultsFetch({ phase: "fetching", stage: status.stage }));
      if (final.status === "error") {
        setResultsFetch({ phase: "error", message: final.error || "Fetch failed." });
        return;
      }
      if (final.warnings && final.warnings.length) {
        setResultsFetch({
          phase: "warning",
          message:
            "Official result saved. The automatic retrain that normally follows didn't complete - check the " +
            "Performance tab, or run it by hand later: " +
            final.warnings.join(" "),
        });
      } else {
        setResultsFetch(null);
      }
      onResultsFetched();
    } catch (e) {
      setResultsFetch({ phase: "error", message: e.message });
    }
  }

  function selectYear(y) {
    onSelectYear(y);
    const roundsForYear = [...new Set(rows.filter((r) => r.year === y).map((r) => r.round_number))].sort(
      (a, b) => a - b
    );
    onSelectRound(roundsForYear[roundsForYear.length - 1]);
  }

  let outcome = null;
  if (scored) {
    const avgErr = roundRows.reduce((s, r) => s + r.abs_error_seconds, 0) / roundRows.length;
    const nWithin = roundRows.filter((r) => r.within_interval).length;
    const closest = roundRows.reduce((a, b) => (a.abs_error_seconds <= b.abs_error_seconds ? a : b));
    const biggest = roundRows.reduce((a, b) => (a.abs_error_seconds >= b.abs_error_seconds ? a : b));
    const closestId = driverIdentity(closest.driver, closest.year, closest.round_number);
    const biggestId = driverIdentity(biggest.driver, biggest.year, biggest.round_number);
    outcome = (
      <div className="outcome-row">
        <div className="outcome-tile">
          <div className="outcome-label">Average error</div>
          <div className="outcome-value">{formatDelta(avgErr)}</div>
          <div className="outcome-caption">across all {roundRows.length} drivers</div>
        </div>
        <div className="outcome-tile">
          <div className="outcome-label">Inside typical range</div>
          <div className="outcome-value">
            {nWithin} / {roundRows.length}
          </div>
          <div className="outcome-caption">this round's shipped range</div>
        </div>
        <div className="outcome-tile">
          <div className="outcome-label">Closest call</div>
          <div className="outcome-driver">
            <div className="team-bar" style={{ background: closestId.color }}></div>
            <div className="outcome-value" style={{ fontSize: 20 }}>
              {closestId.number} {closest.driver}
            </div>
          </div>
          <div className="outcome-caption">out by {formatDelta(closest.abs_error_seconds)}</div>
        </div>
        <div className="outcome-tile">
          <div className="outcome-label">Biggest miss</div>
          <div className="outcome-driver">
            <div className="team-bar" style={{ background: biggestId.color }}></div>
            <div className="outcome-value" style={{ fontSize: 20 }}>
              {biggestId.number} {biggest.driver}
            </div>
          </div>
          <div className="outcome-caption">out by {formatDelta(biggest.abs_error_seconds)}</div>
        </div>
      </div>
    );
  }

  const overrides = roundOverridesFor(selectedYear, selectedRound);
  const spread = roundRows.length
    ? Math.max(...roundRows.map((r) => r.predicted_quali_time_seconds)) -
      Math.min(...roundRows.map((r) => r.predicted_quali_time_seconds))
    : null;
  const contextBits = [`Field spread (predicted): ${formatDelta(spread)}`];
  if (overrides) {
    contextBits.push(
      "Line-up change this round: " + Object.entries(overrides).map(([code, id]) => `${code} to ${id.team}`).join(", ")
    );
  }

  return (
    <>
      <div className="panel filters">
        <div className="filter-row">
          <div className="filter-label">Season</div>
          <div className="pill-group">
            {years.map((y) => (
              <div key={y} className={`pill${y === selectedYear ? " active" : ""}`} onClick={() => selectYear(y)}>
                {y}
              </div>
            ))}
          </div>
        </div>
        <div className="filter-row">
          <div className="filter-label">Round</div>
          <div className="pill-group">
            {rounds.map((r) => (
              <div
                key={r}
                className={`pill${r === selectedRound ? " active" : ""}`}
                onClick={() => onSelectRound(r)}
              >
                {r}
                {roundHasScore.has(`${selectedYear}:${r}`) ? <span className="s-dot"></span> : null}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="round-header">
        <div className="round-title">
          ROUND {selectedRound}
          {eventName ? <> &middot; {eventName}</> : null} &middot; <b>{selectedYear} SEASON</b>
        </div>
        <div className={`badge ${scored ? "badge--good" : "badge--neutral"}`}>
          {scored ? "Official result in" : "Awaiting official result"}
        </div>
      </div>

      {!scored && resultsFetch === null && canWrite && (
        <div className="launch-row">
          <div className="spacer"></div>
          <button className="launch-btn launch-btn--ghost" type="button" onClick={() => setResultsFetch({ phase: "confirm" })}>
            Check for official result
          </button>
        </div>
      )}
      {resultsFetch && resultsFetch.phase === "confirm" && (
        <div className="panel launch-card">
          <p className="placeholder-note">
            Fetch the official qualifying result for Round {selectedRound} now? Only works once that round's
            qualifying session has actually happened - this can take a moment.
          </p>
          <div className="launch-row">
            <div className="spacer"></div>
            <button className="launch-btn launch-btn--ghost" type="button" onClick={() => setResultsFetch(null)}>
              Cancel
            </button>
            <button className="launch-btn" type="button" onClick={startResultsFetch}>
              Fetch now
            </button>
          </div>
        </div>
      )}
      {resultsFetch && resultsFetch.phase === "fetching" && (
        <div className="panel launch-card">
          <p className="placeholder-note">
            {formatStage(resultsFetch.stage)}
            {dots}
          </p>
        </div>
      )}
      {resultsFetch && resultsFetch.phase === "error" && (
        <div className="panel launch-card">
          <p className="placeholder-note">{resultsFetch.message}</p>
          <div className="launch-row">
            <div className="spacer"></div>
            <button className="launch-btn launch-btn--ghost" type="button" onClick={() => setResultsFetch(null)}>
              Dismiss
            </button>
            <button className="launch-btn" type="button" onClick={startResultsFetch}>
              Try again
            </button>
          </div>
        </div>
      )}
      {resultsFetch && resultsFetch.phase === "warning" && (
        <div className="panel launch-card">
          <p className="placeholder-note">{resultsFetch.message}</p>
          <div className="launch-row">
            <div className="spacer"></div>
            <button className="launch-btn launch-btn--ghost" type="button" onClick={() => setResultsFetch(null)}>
              Dismiss
            </button>
          </div>
        </div>
      )}

      {outcome}
      <div className="cond-note-text">{contextBits.join(" · ")}</div>
      <div className="panel">
        <div className="compare-head">
          <div className="compare-title">Predicted{scored ? " vs. official result" : " qualifying order"}</div>
          <div className="compare-meta">ordered by predicted time</div>
        </div>
        <PredictionsTable rows={roundRows} year={selectedYear} roundNumber={selectedRound} scored={scored} />
      </div>
    </>
  );
}
