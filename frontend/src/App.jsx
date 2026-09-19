import { useCallback, useEffect, useState } from "react";
import StatusStrip from "./components/StatusStrip.jsx";
import PredictionTab from "./components/PredictionTab.jsx";
import HistoryTab from "./components/HistoryTab.jsx";
import PerformanceTab from "./components/PerformanceTab.jsx";
import { fetchJSON, getAdminToken, setAdminToken } from "./api.js";

const TABS = [
  { key: "prediction", label: "Prediction" },
  { key: "history", label: "History" },
  { key: "performance", label: "Performance" },
];

// Aug 30 2026 React rebuild (Damiano's own choice, via AskUserQuestion,
// over keeping the Aug 26 2026 plain HTML/CSS/vanilla-JS frontend it
// replaces - see Task_List.txt and project memory for the full history).
// All three tabs stay mounted at once (never conditionally unmounted) so
// each keeps its own state - e.g. the History tab's selected season/
// round - across tab switches, matching the old vanilla-JS frontend's
// DOM-based .view/.view.active show/hide instead of React's usual
// conditional-render pattern.
// Sep 18 2026 (Damiano: "the application keeps opening without showing
// the predictions already done... a guest will never see anything since
// only me can launch a prediction"): default landing tab is History, not
// Prediction. History is a public read (GET /history, no admin gating -
// see HistoryTab.jsx) and already auto-selects the latest launched
// year/round with zero extra logic, so this one-line change is enough to
// make the app open on the latest launched round (Round 14 as of today)
// with its predictions already visible, to guests included. Prediction
// stays reachable via its tab for the one person who can Launch.
export default function App() {
  const [activeTab, setActiveTab] = useState("history");
  // Bumped every time a launch succeeds - History/Performance both watch
  // this to know their cached data is stale and re-fetch, mirroring the
  // old historyLoaded/performanceLoaded = false reset in app.js.
  const [dataVersion, setDataVersion] = useState(0);
  const bumpDataVersion = useCallback(() => setDataVersion((v) => v + 1), []);

  // Sep 13 2026 addition (point 4 of the "final version of the
  // application" plan): whether THIS deployment gates Launch/data-fetch
  // behind an admin token at all (false for local docker-compose, no
  // ADMIN_TOKEN set - see f1qp.api.main.AuthStatusResponse's docstring),
  // and whether the token already saved in this browser (if any) is
  // currently valid. `canWrite` is the single value everything below
  // actually branches on: true when gating is off, OR when it's on and
  // this browser's token checks out.
  const [adminRequired, setAdminRequired] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  const canWrite = !adminRequired || isAdmin;

  // Re-validates whatever token is currently in localStorage against
  // POST /auth/check - shared by the initial mount check below and by
  // StatusStrip's Settings panel after the person pastes a new one, so
  // there is exactly one place that decides "is this browser an admin
  // right now".
  const revalidateAdmin = useCallback(async () => {
    if (!getAdminToken()) {
      setIsAdmin(false);
      return false;
    }
    try {
      await fetchJSON("/auth/check", { method: "POST" });
      setIsAdmin(true);
      return true;
    } catch (e) {
      setIsAdmin(false);
      return false;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      let required = false;
      try {
        const status = await fetchJSON("/auth/status");
        required = !!status.admin_required;
      } catch (e) {
        required = false; // can't tell - default to the frictionless local behavior
      }
      if (cancelled) return;
      setAdminRequired(required);
      if (required) {
        await revalidateAdmin();
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [revalidateAdmin]);

  // Passed to StatusStrip's Settings panel - saves (or clears) the token
  // in this browser and re-checks it immediately, so unlocking/locking
  // takes effect on every tab's buttons right away, not on next reload.
  const applyAdminToken = useCallback(
    async (token) => {
      setAdminToken(token);
      return revalidateAdmin();
    },
    [revalidateAdmin]
  );

  return (
    <div className="app">
      <div className="topbar">
        <div className="topbar-row">
          <div className="brand">
            <svg className="brand-mark" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <circle cx="12" cy="13" r="8"></circle>
              <path d="M9 3h6"></path>
              <path d="M12 8v5l3.2 2"></path>
            </svg>
            <div className="brand-text">
              QUALIFYING <span>PREDICTOR</span>
            </div>
          </div>
          <div className="tabs">
            {TABS.map((t) => (
              <div
                key={t.key}
                className={`tab${activeTab === t.key ? " active" : ""}`}
                onClick={() => setActiveTab(t.key)}
              >
                {t.label}
              </div>
            ))}
          </div>
        </div>
        <StatusStrip adminRequired={adminRequired} isAdmin={isAdmin} onApplyToken={applyAdminToken} />
      </div>

      <div className="content">
        <PredictionTab active={activeTab === "prediction"} onLaunched={bumpDataVersion} canWrite={canWrite} />
        <HistoryTab active={activeTab === "history"} dataVersion={dataVersion} canWrite={canWrite} />
        <PerformanceTab active={activeTab === "performance"} dataVersion={dataVersion} canWrite={canWrite} />
      </div>
    </div>
  );
}
