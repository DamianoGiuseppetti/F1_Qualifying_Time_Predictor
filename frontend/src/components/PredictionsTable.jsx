import { useState } from "react";
import { driverIdentity } from "../teamColors.js";
import { formatMmss, formatDelta } from "../format.js";

function DriverCell({ driver, year, roundNumber }) {
  const id = driverIdentity(driver, year, roundNumber);
  return (
    <div className="driver-id">
      <div className="team-bar" style={{ background: id.color }}></div>
      <div className="driver-num">{id.number}</div>
      <div className="driver-code">{driver}</div>
    </div>
  );
}

// Shared by the Prediction tab and the History tab - one table, `scored`
// controls whether the official-result columns (Off. Pos/Official/Delta/
// Result) render at all. Mirrors the retired dashboard/app.py's
// `_render_predictions_table` (same base columns, same Typical range
// decimals=1 exception), plus the Aug 30 2026 addition below.
//
// Aug 30 2026 (Damiano): "Give the possibility to order also for official
// results. Make two columns, one for prediction position (as the one
// already there) and one for official position, both in prediction page
// than history page." Two independent rankings are computed up front -
// "Pos" (by predicted time, always available) and "Off. Pos" (by official
// time, only among drivers who have a result yet) - and shown as their
// own columns regardless of which one the table happens to be sorted by;
// clicking either column header re-sorts the table by that ranking.
export function PredictionsTable({ rows, year, roundNumber, scored }) {
  const [sortBy, setSortBy] = useState("predicted"); // "predicted" | "official"

  if (!rows || rows.length === 0) {
    return <p className="placeholder-note">No rows.</p>;
  }

  const byPredicted = [...rows].sort((a, b) => a.predicted_quali_time_seconds - b.predicted_quali_time_seconds);
  const predictedPos = new Map(byPredicted.map((r, i) => [r.driver, i + 1]));

  // Official position: rank by official time among scored drivers only -
  // a driver with no result yet has no official position (shown "--") and
  // always sorts after every scored driver when sorting by this column.
  const byOfficial = rows
    .filter((r) => r.has_target)
    .sort((a, b) => a.final_quali_time - b.final_quali_time);
  const officialPos = new Map(byOfficial.map((r, i) => [r.driver, i + 1]));

  const displayRows =
    sortBy === "official"
      ? [...rows].sort((a, b) => {
          const pa = officialPos.get(a.driver);
          const pb = officialPos.get(b.driver);
          if (pa == null && pb == null) return predictedPos.get(a.driver) - predictedPos.get(b.driver);
          if (pa == null) return 1;
          if (pb == null) return -1;
          return pa - pb;
        })
      : byPredicted;

  const sortHint = { cursor: "pointer", userSelect: "none" };

  return (
    <table className="timing">
      <thead>
        <tr>
          <th
            style={sortHint}
            title="Sort by predicted position"
            onClick={() => setSortBy("predicted")}
          >
            Pos{sortBy === "predicted" ? " ▾" : ""}
          </th>
          <th>Driver</th>
          <th>
            <span className="th-swatch">
              <i style={{ background: "#3987e5" }}></i>Predicted
            </span>
          </th>
          <th>Typical range</th>
          {scored && (
            <>
              <th
                style={sortHint}
                title="Sort by official position"
                onClick={() => setSortBy("official")}
              >
                Off. Pos{sortBy === "official" ? " ▾" : ""}
              </th>
              <th>
                <span className="th-swatch">
                  <i style={{ background: "#d95926" }}></i>Official
                </span>
              </th>
              <th>&Delta;</th>
              <th>Result</th>
            </>
          )}
        </tr>
      </thead>
      <tbody>
        {displayRows.map((r) => (
          <tr key={r.driver}>
            <td className="pos-cell">{predictedPos.get(r.driver)}</td>
            <td>
              <DriverCell driver={r.driver} year={year} roundNumber={roundNumber} />
            </td>
            <td className="pred-cell">{formatMmss(r.predicted_quali_time_seconds)}</td>
            <td className="range-cell">
              {formatMmss(r.interval_low_seconds, 1)} &ndash; {formatMmss(r.interval_high_seconds, 1)}
            </td>
            {scored &&
              (r.has_target ? (
                <>
                  <td className="pos-cell">{officialPos.get(r.driver)}</td>
                  <td className="official-cell">{formatMmss(r.final_quali_time)}</td>
                  <td className={`delta-cell ${r.within_interval ? "good" : "critical"}`}>
                    {formatDelta(r.abs_error_seconds)}
                  </td>
                  <td>
                    {r.within_interval ? (
                      <div className="result-cell good">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4">
                          <path d="M20 6L9 17l-5-5"></path>
                        </svg>
                        <span>Within range</span>
                      </div>
                    ) : (
                      <div className="result-cell critical">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4">
                          <path d="M18 6L6 18"></path>
                          <path d="M6 6l12 12"></path>
                        </svg>
                        <span>Missed</span>
                      </div>
                    )}
                  </td>
                </>
              ) : (
                <>
                  <td className="pos-cell">--</td>
                  <td className="official-cell">--:--.---</td>
                  <td className="delta-cell">--</td>
                  <td>--</td>
                </>
              ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ExcludedNote({ excluded }) {
  if (!excluded || !excluded.length) return null;
  return (
    <div className="panel excluded-note">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 9v4"></path>
        <path d="M12 17h.01"></path>
        <path d="M10.3 3.9L2.5 17a2 2 0 0 0 1.7 3h15.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"></path>
      </svg>
      <div className="excluded-body">
        <div className="excluded-title">
          {excluded.length} {excluded.length === 1 ? "entry" : "entries"} excluded from this run &mdash;{" "}
          <b>FP1-only appearances</b>
        </div>
        <div className="excluded-chips">
          {excluded.map((code) => (
            <div className="excluded-chip" key={code}>
              {code}
            </div>
          ))}
        </div>
        <div className="excluded-explain">
          A driver is scored only once they've run more than one session this weekend, or any session besides FP1. A
          single FP1-only outing is a mandatory rookie or reserve appearance &mdash; that seat's real occupant
          returns for FP2/FP3 and is who actually qualifies.
        </div>
      </div>
    </div>
  );
}
