import { useEffect, useState } from "react";
import { fetchJSON } from "../api.js";
import { formatTimestamp } from "../format.js";

// Polls /health every 20s, same cadence as the retired vanilla-JS
// frontend's setInterval(refreshStatus, 20000).
//
// Sep 13 2026 addition: `adminRequired`/`isAdmin`/`onApplyToken` come
// from App.jsx (see its own comments) - this component only renders the
// admin chip and the small token-entry panel; App.jsx owns whether a
// token is actually valid. Nothing here shows up at all when
// `adminRequired` is false (local docker-compose, no ADMIN_TOKEN set) -
// same status strip as before this existed.
export default function StatusStrip({ adminRequired, isAdmin, onApplyToken }) {
  const [online, setOnline] = useState(null); // null = still checking for the first time
  const [trainedAt, setTrainedAt] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const health = await fetchJSON("/health");
        if (cancelled) return;
        setOnline(true);
        setTrainedAt(health.model_trained_at_utc);
      } catch (e) {
        if (cancelled) return;
        setOnline(false);
      }
    }
    check();
    const id = setInterval(check, 20000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="status-strip">
      <div className="status-chip">
        <span className={`live-dot${online === false ? " offline" : ""}`}></span> ENGINE{" "}
        <b>{online == null ? "CHECKING…" : online ? "ONLINE" : "OFFLINE"}</b>
      </div>
      <div className="divider-dot"></div>
      <div className="status-chip">
        TRAINED <b>{formatTimestamp(trainedAt)}</b>
      </div>
      {adminRequired ? (
        <>
          <div className="divider-dot"></div>
          <AdminControl isAdmin={isAdmin} onApplyToken={onApplyToken} />
        </>
      ) : null}
    </div>
  );
}

// Small unlock-with-a-token control, only ever mounted when this
// deployment actually has ADMIN_TOKEN configured (see StatusStrip
// above). Guests never see anything to click beyond the chip itself;
// Damiano pastes his token here once per browser (localStorage, see
// api.js) and it's remembered on that browser from then on.
function AdminControl({ isAdmin, onApplyToken }) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState("");

  async function unlock() {
    if (!draft) return;
    setChecking(true);
    setError("");
    const ok = await onApplyToken(draft);
    setChecking(false);
    if (ok) {
      setOpen(false);
      setDraft("");
    } else {
      setError("Incorrect token.");
    }
  }

  async function lock() {
    await onApplyToken("");
    setOpen(false);
    setDraft("");
    setError("");
  }

  return (
    <div className="admin-control">
      <div
        className={`status-chip admin-chip${isAdmin ? " admin-chip--unlocked" : ""}`}
        onClick={() => setOpen((v) => !v)}
      >
        {isAdmin ? "ADMIN" : "GUEST"}
      </div>
      {open && (
        <div className="admin-panel">
          {isAdmin ? (
            <>
              <div className="admin-panel-label">Signed in as admin on this browser.</div>
              <button className="launch-btn launch-btn--ghost" type="button" onClick={lock}>
                Sign out
              </button>
            </>
          ) : (
            <>
              <div className="admin-panel-label">Admin token</div>
              <input
                className="field-box"
                type="password"
                autoFocus
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && unlock()}
                placeholder="Paste your token"
              />
              {error && <div className="admin-panel-error">{error}</div>}
              <button className="launch-btn" type="button" disabled={checking || !draft} onClick={unlock}>
                {checking ? "Checking…" : "Unlock"}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
