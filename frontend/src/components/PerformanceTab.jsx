import { useCallback, useEffect, useRef, useState } from "react";
import { fetchJSON } from "../api.js";
import { formatTimestamp } from "../format.js";

// Sep 16 2026 addition (Damiano's "auto-stage, one-click to promote"
// design): `canWrite` (see App.jsx / PredictionTab.jsx's own comments) -
// viewing Performance stays open to everyone; only the "Promote to
// production" action below is restricted, same gating as Launch and the
// History tab's "Check for official result".
export default function PerformanceTab({ active, dataVersion, canWrite }) {
  const [info, setInfo] = useState(null);
  const [historyRows, setHistoryRows] = useState([]);
  const [runs, setRuns] = useState([]);
  const [staged, setStaged] = useState(null);
  const [loaded, setLoaded] = useState(false);
  const loadedVersionRef = useRef(-1);
  // Bumped locally after a successful Promote, alongside the normal
  // dataVersion-driven load below - independent of any Launch elsewhere.
  const [localVersion, setLocalVersion] = useState(0);

  const loadAll = useCallback(async () => {
    let infoResp = null;
    let historyResp = null;
    let runsResp = [];
    let stagedResp = null;
    try {
      infoResp = await fetchJSON("/model/info");
    } catch (e) {
      /* handled below via null info */
    }
    try {
      historyResp = await fetchJSON("/history");
    } catch (e) {
      /* handled below via empty rows */
    }
    try {
      runsResp = await fetchJSON("/model/runs");
    } catch (e) {
      runsResp = [];
    }
    try {
      stagedResp = await fetchJSON("/model/staged");
    } catch (e) {
      stagedResp = null;
    }
    setInfo(infoResp);
    setHistoryRows((historyResp && historyResp.rows) || []);
    setRuns(runsResp || []);
    setStaged(stagedResp);
    setLoaded(true);
  }, []);

  useEffect(() => {
    if (!active) return;
    if (loadedVersionRef.current === dataVersion) return;
    loadedVersionRef.current = dataVersion;
    let cancelled = false;
    (async () => {
      if (cancelled) return;
      await loadAll();
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, dataVersion, localVersion]);

  return (
    <section className={`view${active ? " active" : ""}`} id="view-performance">
      {!loaded ? (
        <p className="placeholder-note">Loading&hellip;</p>
      ) : (
        <PerformanceBody
          info={info}
          historyRows={historyRows}
          runs={runs}
          staged={staged}
          canWrite={canWrite}
          onPromoted={() => setLocalVersion((v) => v + 1)}
        />
      )}
    </section>
  );
}

const _COMPARISON_LABELS = {
  n_train: "Training rows",
  pooled_mape: "MAPE",
  pooled_r2: "R²",
  epoch_count: "Epochs",
  interval_50pct: "50% range",
};

function _formatComparisonValue(key, value) {
  if (value == null) return "?";
  if (key === "pooled_mape") return `${Number(value).toFixed(3)}%`;
  if (key === "pooled_r2") return Number(value).toFixed(3);
  if (key === "interval_50pct") return `±${Number(value).toFixed(3)}s`;
  return String(value);
}

// Damiano's own spec: "The app shows you a before/after comparison...
// One explicit action ('Promote to production') swaps the live model. If
// you don't act, last week's model keeps serving." This card is that
// checkpoint - only rendered when the automatic post-results chain
// (f1qp.serving.data_fetch.start_results_job -> prepare_phase3_dataset.py
// --include-holdout -> retrain_pipeline.py) has actually produced a
// staged candidate.
function StagedModelCard({ staged, canWrite, onPromoted }) {
  const [promoting, setPromoting] = useState(false);
  const [error, setError] = useState("");

  if (!staged || !staged.has_staged) return null;

  async function promote() {
    setPromoting(true);
    setError("");
    try {
      await fetchJSON("/model/promote", { method: "POST" });
      onPromoted();
    } catch (e) {
      setError(e.message);
      setPromoting(false);
    }
  }

  return (
    <div className="panel section staged-model-card">
      <div className="section-head">
        <div className="section-label">Ready to review</div>
        <div className="section-title">A new model is staged</div>
        <div className="section-sub">
          Trained {formatTimestamp(staged.trained_at_utc)} from this weekend's official result &mdash; production
          keeps serving the current model until you promote this one.
        </div>
      </div>
      {staged.comparison && staged.comparison.length > 0 && (
        <table className="timing">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Current (live)</th>
              <th>Staged (candidate)</th>
            </tr>
          </thead>
          <tbody>
            {staged.comparison.map((row) => (
              <tr key={row.key}>
                <td>{_COMPARISON_LABELS[row.key] || row.key}</td>
                <td>{_formatComparisonValue(row.key, row.previous)}</td>
                <td>{_formatComparisonValue(row.key, row.current)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {error && <p className="placeholder-note">{error}</p>}
      <div className="launch-row">
        <div className="spacer"></div>
        {!canWrite ? (
          <span className="run-meta">Admin only - unlock via the GUEST/ADMIN chip in the status bar to promote.</span>
        ) : (
          <button className="launch-btn" type="button" disabled={promoting} onClick={promote}>
            {promoting ? "Promoting…" : "Promote to production"}
          </button>
        )}
      </div>
    </div>
  );
}

function StatDelta({ latest, previous, lowerIsBetter }) {
  if (latest == null || previous == null) return null;
  const diff = latest - previous;
  if (diff === 0) return null;
  const improved = lowerIsBetter ? diff < 0 : diff > 0;
  return (
    <div className={`stat-delta ${improved ? "good" : "critical"}`}>
      {diff < 0 ? (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
          <path d="M6 9l6 6 6-6"></path>
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
          <path d="M6 15l6-6 6 6"></path>
        </svg>
      )}
      {Math.abs(diff).toFixed(3)}
    </div>
  );
}

function PerformanceBody({ info, historyRows, runs, staged, canWrite, onPromoted }) {
  const sortedRuns = [...runs].sort((a, b) => new Date(a.start_time || 0) - new Date(b.start_time || 0));
  // Sep 16 2026 fix: with staged candidates now possible, the last LOGGED
  // MLflow run is no longer necessarily what's actually being SERVED - a
  // run can sit staged (not yet promoted) for a while. `info.mlflow_run_id`
  // (set by f1qp.api.main's /model/info from the live production
  // metadata) is the source of truth for what "CURRENT" means; fall back
  // to the last run if it's missing (a production model trained before
  // this field existed) rather than showing no CURRENT badge at all.
  const currentRunIndex = info && info.mlflow_run_id
    ? sortedRuns.findIndex((r) => r.run_id === info.mlflow_run_id)
    : -1;
  const latestRunIdx = currentRunIndex >= 0 ? currentRunIndex : sortedRuns.length - 1;
  const latestRun = sortedRuns[latestRunIdx];
  const prevRun = sortedRuns[latestRunIdx - 1];

  const scoredRows = historyRows.filter((r) => r.has_target);
  const grouped = {};
  scoredRows.forEach((r) => {
    const key = `${r.year}:${r.round_number}`;
    if (!grouped[key]) grouped[key] = { year: r.year, round_number: r.round_number, errors: [] };
    grouped[key].errors.push(r.abs_error_seconds);
  });
  const bars = Object.values(grouped)
    .map((g) => ({ ...g, mean: g.errors.reduce((s, e) => s + e, 0) / g.errors.length }))
    .sort((a, b) => a.year - b.year || a.round_number - b.round_number);
  const maxMean = Math.max(...bars.map((b) => b.mean), 0.001);

  return (
    <>
      <StagedModelCard staged={staged} canWrite={canWrite} onPromoted={onPromoted} />
      {!info ? (
        <p className="placeholder-note">Model info unavailable &mdash; check the API connection.</p>
      ) : (
        <>
          <div className="hero-stats">
            <div className="stat-tile">
              <div className="stat-label">Trained</div>
              <div className="stat-value" style={{ fontSize: 16 }}>
                {formatTimestamp(info.trained_at_utc)}
              </div>
            </div>
            <div className="stat-tile">
              <div className="stat-label">MAPE</div>
              <div className="stat-value">{info.reference_leave_one_round_out_pooled_mape.toFixed(3)}%</div>
              {prevRun && latestRun && (
                <StatDelta latest={latestRun.metrics.pooled_mape} previous={prevRun.metrics.pooled_mape} lowerIsBetter={true} />
              )}
            </div>
            <div className="stat-tile">
              <div className="stat-label">R&sup2;</div>
              <div className="stat-value">{info.reference_leave_one_round_out_pooled_r2.toFixed(3)}</div>
              {prevRun && latestRun && (
                <StatDelta latest={latestRun.metrics.pooled_r2} previous={prevRun.metrics.pooled_r2} lowerIsBetter={false} />
              )}
            </div>
            <div className="stat-tile">
              <div className="stat-label">Typical range</div>
              <div className="stat-value">&plusmn;{info.deployment_quantile_seconds.toFixed(3)}s</div>
              <div className="stat-caption">{info.interval_level_pct.toFixed(0)}% coverage</div>
            </div>
            <div className="stat-tile">
              <div className="stat-label">Training rows</div>
              <div className="stat-value">{info.n_train.toLocaleString()}</div>
            </div>
            <div className="stat-tile">
              <div className="stat-label">Epochs</div>
              <div className="stat-value">{info.n_epochs}</div>
            </div>
          </div>
          <div className="measure-note">
            MAPE and R&sup2; are measured leave-one-round-out: every 2026 round is scored by a model that never
            trained on it, then pooled. It's the honest stand-in for a round the model hasn't seen &mdash; not a
            score on its own training data.
          </div>
          {info.note && <div className="measure-note">{info.note}</div>}
        </>
      )}

      <div className="panel section">
        <div className="section-head">
          <div className="section-title">Per-round accuracy &mdash; launched &amp; scored rounds</div>
          <div className="section-sub">Each bar is a real launched round, scored once its official qualifying result is in.</div>
        </div>
        {!scoredRows.length ? (
          <p className="placeholder-note">
            No scored rounds yet from launched predictions &mdash; launch a round from the Prediction tab and wait
            for its official result to populate this chart.
          </p>
        ) : (
          <div className="chart-wrap">
            <div className="chart-plot">
              {bars.map((b) => (
                <div className="bar-col" key={`${b.year}:${b.round_number}`}>
                  <div className="bar good" style={{ height: `${Math.max(4, (b.mean / maxMean) * 190)}px` }}></div>
                </div>
              ))}
            </div>
            <div className="chart-axis">
              {bars.map((b) => (
                <div className="axis-label" key={`${b.year}:${b.round_number}`}>
                  {b.year} R{b.round_number}
                </div>
              ))}
            </div>
            <div className="chart-note">
              Average absolute prediction error per scored round (seconds) &mdash; lower is better. Built from real
              launches through this app, distinct from the pooled leave-one-round-out MAPE above, and empty until at
              least one launched round has been scored.
            </div>
          </div>
        )}
      </div>

      <div className="panel section">
        <div className="section-head">
          <div className="section-title">Retrain history</div>
          <div className="section-sub">
            One entry per completed <code>retrain_pipeline.py</code> run, read from MLflow.
          </div>
        </div>
        {!sortedRuns.length ? (
          <p className="placeholder-note">
            No retrain runs logged yet. Run <code>python scripts/retrain_pipeline.py</code> on the host at least once
            to populate this section &mdash; each run adds one entry here.
          </p>
        ) : (
          <div className="timeline">
            {sortedRuns.map((run, i) => {
              const isCurrent = i === latestRunIdx;
              const m = run.metrics || {};
              const p = run.params || {};
              const subBits = [];
              if (p.n_train) subBits.push(`${p.n_train} rows`);
              if (p.n_epochs) subBits.push(`${p.n_epochs} epochs`);
              return (
                <div className={`timeline-row${isCurrent ? " current" : ""}`} key={run.start_time || i}>
                  <div className="timeline-rail">
                    <div className="timeline-dot"></div>
                    {i < sortedRuns.length - 1 && <div className="timeline-line"></div>}
                  </div>
                  <div className="timeline-card">
                    <div className="timeline-name">
                      <div className="timeline-title">
                        {formatTimestamp(run.start_time)} {isCurrent && <span className="current-badge">CURRENT</span>}
                      </div>
                      <div className="timeline-sub">{subBits.join(" · ")}</div>
                    </div>
                    <div className="timeline-metrics">
                      {m.pooled_mape != null && (
                        <div className="tm">
                          <div className="tm-label">MAPE</div>
                          <div className="tm-value">{Number(m.pooled_mape).toFixed(3)}%</div>
                        </div>
                      )}
                      {m.pooled_r2 != null && (
                        <div className="tm">
                          <div className="tm-label">R&sup2;</div>
                          <div className="tm-value">{Number(m.pooled_r2).toFixed(3)}</div>
                        </div>
                      )}
                      {m.interval_50pct != null && (
                        <div className="tm">
                          <div className="tm-label">Range</div>
                          <div className="tm-value">&plusmn;{Number(m.interval_50pct).toFixed(3)}s</div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}
