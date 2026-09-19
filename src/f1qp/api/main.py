"""Phase 4, API step: FastAPI inference endpoint for the production LSTM.

Loads the current production artifacts ONCE at startup (see
`f1qp.serving.predict.load_production_artifacts`) and reuses them for every
request - this is what keeps `/predict` fast enough for the <500ms latency
target in Task_List.txt's success criteria; reloading the model/scaler from
disk per request would be pure waste since nothing about them changes
between requests (only a retrain - a separate manual step, per the Aug 24
2026 AskUserQuestion decision - changes them).

Run locally from the repo root:

    uvicorn f1qp.api.main:app --reload

then e.g. `curl -X POST http://127.0.0.1:8000/predict/2026/13`, or open
http://127.0.0.1:8000/docs for the interactive Swagger UI FastAPI builds
automatically from the response models below.

Retrain trigger (superseded Sep 16 2026 - see f1qp.modeling.promote's
module docstring): the Aug 24 2026 "manual CLI step only" decision above
is no longer current. `f1qp.serving.data_fetch.start_results_job` now
chains straight into `scripts/prepare_phase3_dataset.py --include-holdout`
+ `scripts/retrain_pipeline.py` automatically once a round's official
result is fetched - but retraining automatically does NOT mean serving
the result automatically: retrain_pipeline.py only ever writes a STAGED
candidate under `models/lstm/staged/<timestamp>/`. This module's cached
`_state["artifacts"]` only changes when `POST /model/promote` (below) is
called - Damiano's own explicit "one-click to promote" checkpoint.

Sep 13 2026 addition (point 4 of the "final version of the application"
plan - Damiano: "only me can make predictions on the application while
guests... will see the application but not be able to make the
predictions", answered "allow Preview only" via AskUserQuestion): a
single shared ADMIN_TOKEN secret gates Launch and the on-demand data-fetch
triggers - see require_admin()'s docstring below for exactly how. Preview
(`POST /predict/...` without `/launch`) and every GET route stay open to
anyone, unchanged.

Sep 13 2026 addition, same day (the other open item from that plan -
Damiano: "storage should be into hugging face without be in my local
storage anymore"): `lifespan` below now pulls the latest launches/
features/targets from a Hugging Face dataset repo before serving starts,
and f1qp.serving.history/f1qp.serving.data_fetch push their own writes
back out to it - see f1qp.serving.hf_sync's module docstring. A no-op
everywhere this isn't explicitly configured (local docker-compose,
every test in this repo).
"""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from f1qp.config import get_event_name, session_readiness
from f1qp.serving import hf_sync
from f1qp.serving import history as history_mod
from f1qp.serving import predict as predict_mod
from f1qp.serving.predict import (
    MODELS_DIR,
    ArtifactsNotFoundError,
    DriverPrediction,
    EventNotFoundError,
    ProductionArtifacts,
    load_production_artifacts,
    predict_event_from_disk,
    test_drivers_for_event_from_disk,
)
from f1qp.serving.history import (
    latest_two_launches,
    prediction_history,
    record_launch,
    score_predictions,
)
from f1qp.serving.data_fetch import job_status, start_fetch_job, start_results_job
from f1qp.modeling.tracking import MLRUNS_DIR as _DEFAULT_MLRUNS_DIR, search_runs
from f1qp.modeling.promote import NoStagedModelError, latest_staged, promote_staged
from f1qp.modeling.retrain import comparison_rows, format_retrain_comparison

# Sep 16 2026 addition: same F1QP_MLRUNS_DIR resolution `/model/runs`
# already did inline (`search_runs`'s own `mlruns_dir=None` fallback) -
# pulled out to a module constant so `lifespan()`/`model_promote()` below
# can hand hf_sync a real Path too (hf_sync.push_mlruns/pull_all need an
# actual directory to read from/write to, not None).
_MLRUNS_DIR = Path(os.environ.get("F1QP_MLRUNS_DIR") or _DEFAULT_MLRUNS_DIR)


class DriverPredictionResponse(BaseModel):
    driver: str
    era: int
    is_sprint: bool
    n_practice_sessions: int
    predicted_quali_time_seconds: float
    interval_low_seconds: float
    interval_high_seconds: float
    interval_width_seconds: float
    interval_level_pct: float
    interval_exact: bool
    interval_label: str
    # Aug 30 2026 (Damiano): "Give the possibility to order also for
    # official results... both in prediction page than history page" -
    # every prediction (preview OR launch, not just an already-launched
    # History row) is now scored against qualifying_targets.parquet at
    # response time (see f1qp.serving.history.score_predictions), so the
    # Prediction tab can show the same official-position/delta/result
    # columns History always could, e.g. when re-previewing an
    # already-happened round. All 4 stay unset (None/False) until an
    # official result exists for this driver - never a crash either way.
    final_quali_time: Optional[float] = None
    has_target: bool = False
    abs_error_seconds: Optional[float] = None
    within_interval: Optional[bool] = None
    # Sep 16 2026 addition - round-level "has this round's official
    # result actually been extracted at all" (independent of whether THIS
    # driver individually has a time - see
    # f1qp.serving.history._score_predictions_df's docstring). This is
    # what the frontend should use to decide whether to show the
    # official-result columns at all; `has_target` stays purely a per-
    # driver "does THIS row have a time" flag.
    official_results_available: bool = False


class PredictResponse(BaseModel):
    year: int
    round_number: int
    n_drivers: int
    model_trained_at_utc: str
    excluded_test_drivers: List[str]
    predictions: List[DriverPredictionResponse]
    # Aug 30 2026 (Damiano): "Add the name of the GP not only the round."
    # Best-effort (None on a schedule lookup failure) - see
    # f1qp.config.get_event_name's docstring.
    event_name: Optional[str] = None


class LaunchResponse(PredictResponse):
    """Same shape as `PredictResponse`, plus the UTC timestamp this launch
    was recorded at - what `POST /predict/{year}/{round_number}/launch`
    returns once it has persisted the launch (see f1qp.serving.history)."""

    launched_at_utc: str


class HistoryRowResponse(BaseModel):
    """One driver's row in a launched (and possibly scored) round - same
    columns `models/lstm/holdout_evaluation.csv` uses, generalized to any
    launched round rather than just the one-off Round 12 offline test."""

    year: int
    round_number: int
    launched_at_utc: str
    driver: str
    era: int
    is_sprint: bool
    n_practice_sessions: int
    predicted_quali_time_seconds: float
    interval_low_seconds: float
    interval_high_seconds: float
    interval_width_seconds: float
    interval_level_pct: float
    interval_exact: bool
    interval_label: str
    final_quali_time: Optional[float] = None
    has_target: bool
    abs_error_seconds: Optional[float] = None
    within_interval: Optional[bool] = None
    # Sep 16 2026 addition - see DriverPredictionResponse's own field of
    # the same name above; this is what the History tab's round-header
    # badge and round-pill "scored" dot should use instead of
    # `every(r => r.has_target)` (see Task_List.txt / f1qp.serving.
    # history.is_scored's docstring for the Round 14 bug this fixes).
    official_results_available: bool = False
    # Aug 30 2026 (Damiano): "Add the name of the GP not only the round" -
    # same field/source as PredictResponse.event_name above, repeated onto
    # every driver row for a round (see prediction_history()'s docstring).
    event_name: Optional[str] = None


class HistoryResponse(BaseModel):
    rows: List[HistoryRowResponse]


class CurrentPredictionResponse(BaseModel):
    """One of the (up to 2) most recently launched rounds, for the
    Prediction tab's switcher between "the new weekend" and "the last one,
    now scored" - see f1qp.serving.history.latest_two_launches's
    docstring."""

    year: int
    round_number: int
    launched_at_utc: str
    model_trained_at_utc: str
    excluded_test_drivers: List[str]
    scored: bool
    rows: List[HistoryRowResponse]


class ModelInfoResponse(BaseModel):
    trained_at_utc: str
    n_train: int
    n_epochs: int
    reference_leave_one_round_out_pooled_mape: float
    reference_leave_one_round_out_pooled_r2: float
    deployment_quantile_seconds: float
    deployment_quantile_exact: bool
    interval_level_pct: float
    interval_label: str
    note: str
    # Sep 16 2026 addition: which MLflow run produced the model CURRENTLY
    # loaded/serving - lets the Performance tab's "Retrain history"
    # timeline mark the right entry "CURRENT" (a run can sit in MLflow for
    # a while as a staged, not-yet-promoted candidate - see
    # f1qp.modeling.promote - so "the last logged run" is no longer always
    # the same thing as "what's actually live"). None for a production
    # model trained before this field existed, or if MLflow logging
    # failed for it - never an error either way.
    mlflow_run_id: Optional[str] = None


class ModelRunResponse(BaseModel):
    """One retrain_pipeline.py run, read back from the local MLflow
    tracking store (f1qp.modeling.tracking.search_runs) - the Performance
    tab's "Retrain history" timeline. Empty params/metrics dicts when a
    run logged neither (shouldn't normally happen, but never raises)."""

    run_id: str
    start_time: Optional[str] = None
    params: Dict[str, str] = {}
    metrics: Dict[str, float] = {}


class RetrainComparisonRow(BaseModel):
    """One field of the before/after comparison between the currently
    live production model and a staged candidate - see
    f1qp.modeling.retrain.comparison_rows's docstring for the fixed set of
    5 keys (n_train, pooled_mape, pooled_r2, epoch_count, interval_50pct).
    `previous` is None when there's no production model yet at all (the
    very first retrain of the season)."""

    key: str
    previous: Optional[float] = None
    current: Optional[float] = None


class StagedModelResponse(BaseModel):
    """Whether the automatic post-results retrain chain
    (f1qp.serving.data_fetch.start_results_job -> scripts/
    prepare_phase3_dataset.py --include-holdout -> scripts/
    retrain_pipeline.py) has produced a candidate model that hasn't been
    promoted yet - Damiano's "auto-stage, one-click to promote" design.
    `has_staged=False` is the normal steady state between race weekends,
    not an error - the Performance tab simply shows nothing extra then."""

    has_staged: bool
    trained_at_utc: Optional[str] = None
    holdout_included: Optional[bool] = None
    mlflow_run_id: Optional[str] = None
    comparison_text: Optional[str] = None
    comparison: List[RetrainComparisonRow] = []


class PromoteResponse(BaseModel):
    promoted: bool
    staged_trained_at_utc: str
    model_trained_at_utc: str


class HealthResponse(BaseModel):
    status: str
    model_trained_at_utc: str


# Sep 2026 addition (Damiano: "I don't want to run the py scripts outside
# everytime. The app should be able to do everything") - see
# f1qp.serving.data_fetch's module docstring and f1qp.config.
# session_readiness's docstring for the reasoning behind each piece.
class DataReadinessResponse(BaseModel):
    """The Prediction tab's pre-flight check before offering to fetch a
    missing round's data - never an error response on its own; `checked`
    is False (not `ready`) when the check itself couldn't run."""

    ready: bool
    checked: bool
    message: str
    minutes_since_session_start: Optional[float] = None
    buffer_minutes: Optional[int] = None
    recommended_fetch_at_utc: Optional[str] = None


class FetchJobStartedResponse(BaseModel):
    job_id: str


class FetchJobStatusResponse(BaseModel):
    status: str  # "running" | "done" | "error"
    stage: str
    year: int
    round_number: int
    error: Optional[str] = None
    # Sep 16 2026 addition: non-fatal problems from the automatic
    # rebuild-dataset/retrain chain start_results_job now runs after
    # extracting a round's official result (see that function's own
    # docstring) - populated only when one of those best-effort steps
    # failed; the job itself still reports status="done" since the thing
    # it was actually asked to do (fetch the official result) succeeded.
    warnings: List[str] = []


class AuthStatusResponse(BaseModel):
    """Whether THIS deployment has admin gating turned on at all - the
    frontend's Settings panel uses this to decide whether to show up in
    the first place. False for local docker-compose (no ADMIN_TOKEN set,
    same frictionless single-user behavior as before this existed); true
    once a deployment's environment sets ADMIN_TOKEN (see
    deploy/huggingface/SETUP.md for the Hugging Face Space case)."""

    admin_required: bool


class AuthCheckResponse(BaseModel):
    ok: bool


def _configured_admin_token() -> Optional[str]:
    """Read fresh from the environment on every call, not cached at
    import time - so a Space secret set (or changed) after the container
    starts still takes effect without a code change or restart. Empty/
    unset means gating is OFF entirely, see require_admin below."""
    token = os.environ.get("ADMIN_TOKEN", "").strip()
    return token or None


def require_admin(x_admin_token: str = Header(default="", alias="X-Admin-Token")) -> None:
    """FastAPI dependency gating Launch and the on-demand data-fetch
    triggers behind one shared secret (Sep 13 2026, point 4 of the
    "final version of the application" plan - see this module's
    docstring). A no-op whenever ADMIN_TOKEN isn't set at all: local
    docker-compose use (only Damiano can reach localhost anyway) stays
    exactly as frictionless as every earlier phase - only a deployment
    that actually sets the secret turns this into a real check. Compares
    with hmac.compare_digest rather than `==` so a wrong guess can't be
    narrowed down via response-time differences."""
    required = _configured_admin_token()
    if required is None:
        return
    if not x_admin_token or not hmac.compare_digest(x_admin_token, required):
        raise HTTPException(
            status_code=401,
            detail="Admin token missing or incorrect. Preview still works for everyone - "
            "Launch and on-demand data fetch need the admin token (see the app's Settings).",
        )


# Populated at startup (see `lifespan` below), read by every request handler.
# A module-level dict rather than a bare global so tests can monkeypatch it
# without an `global` statement, and so its absence (artifacts failed to
# load) is a clear, checkable falsy value rather than an uninitialized name.
_state: dict = {"artifacts": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sep 13 2026 addition (the other half of the "final version of the
    # application" storage requirement - Damiano: "storage should be into
    # hugging face without be in my local storage anymore"): restore
    # whatever this app most recently wrote at runtime (launches,
    # features.parquet, qualifying_targets.parquet) from the Hugging Face
    # dataset repo BEFORE loading artifacts below, so a freshly
    # (re)deployed Space picks up exactly where the last one left off. A
    # no-op locally and in every test (HF_DATASET_REPO/HF_TOKEN unset) -
    # see f1qp.serving.hf_sync's module docstring. Reads predict_mod/
    # history_mod's own attributes (not a bare imported name) so a test
    # that monkeypatches predict_mod.FEATURES_PATH /
    # history_mod.LAUNCHES_DIR / history_mod.TARGETS_PATH before starting
    # the app is honored here too, same as everywhere else those are used.
    #
    # Sep 16 2026 addition: also restore the live model / staged candidate
    # / MLflow store from Hugging Face - see hf_sync's module docstring's
    # "Sep 16 2026 correction". A retrain or promote done straight from a
    # running Render instance only ever lands on that instance's own
    # ephemeral disk otherwise - this is what makes it survive the NEXT
    # redeploy or restart without depending on Damiano's Mac.
    hf_sync.pull_all(
        predictions_dir=history_mod.LAUNCHES_DIR,
        features_path=predict_mod.FEATURES_PATH,
        targets_path=history_mod.TARGETS_PATH,
        models_dir=MODELS_DIR,
        mlruns_dir=_MLRUNS_DIR,
    )
    # Raises ArtifactsNotFoundError with an actionable message (see
    # f1qp.serving.predict) if models/lstm/ doesn't have a trained
    # production model yet - fails FAST at startup, not on the first
    # request, so a misconfigured deployment is obvious immediately.
    _state["artifacts"] = load_production_artifacts()
    yield
    _state["artifacts"] = None


app = FastAPI(
    title="F1 Qualifying Predictor API",
    description="LSTM qualifying-time predictions (Q1/Q2/Q3, coalesced to "
    "one final_quali_time per driver) for the 2026 F1 season.",
    version="0.1.0",
    lifespan=lifespan,
)


def _artifacts() -> ProductionArtifacts:
    artifacts = _state["artifacts"]
    if artifacts is None:
        # Should be unreachable outside tests that bypass the lifespan
        # context - kept as a clear 503 rather than an AttributeError if it
        # ever does happen (e.g. a request racing an app that failed startup).
        raise HTTPException(
            status_code=503,
            detail="Production model artifacts are not loaded. Check the "
            "API startup logs - models/lstm/ likely needs "
            "scripts/train_final_lstm.py run first.",
        )
    return artifacts


def _score_and_build_predictions(
    year: int, round_number: int, predictions: List[DriverPrediction]
) -> List[DriverPredictionResponse]:
    """Attach each driver's official-result comparison (if one exists yet)
    directly onto the prediction response - Damiano, Aug 30 2026: "Give
    the possibility to order also for official results... both in
    prediction page than history page." Reuses the exact join
    f1qp.serving.history already does for launched rounds
    (`score_predictions`), so `/predict` and `/predict/.../launch` both
    return the same official-position/delta/result data the History tab
    would show for the same round - shared by `predict()` and `launch()`
    below so they can never drift apart.

    NaN -> None conversion (`scored_clean`) mirrors
    `_rows_to_history_response`'s own `df.where(pd.notna(df), None)` a few
    lines down - a raw NaN would otherwise round-trip into the JSON
    response as a literal (invalid-JSON) `NaN` token.
    """
    scored_df = score_predictions(year, round_number, predictions)
    if scored_df.empty:
        scored_by_driver: Dict[str, dict] = {}
    else:
        scored_clean = scored_df.where(pd.notna(scored_df), None)
        scored_by_driver = {row["driver"]: row for row in scored_clean.to_dict(orient="records")}

    out = []
    for p in predictions:
        extra = scored_by_driver.get(p.driver, {})
        out.append(
            DriverPredictionResponse(
                **p.__dict__,
                final_quali_time=extra.get("final_quali_time"),
                has_target=bool(extra.get("has_target", False)),
                abs_error_seconds=extra.get("abs_error_seconds"),
                within_interval=extra.get("within_interval"),
                official_results_available=bool(extra.get("official_results_available", False)),
            )
        )
    return out


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifacts = _artifacts()
    return HealthResponse(status="ok", model_trained_at_utc=artifacts.metadata.get("trained_at_utc", ""))


@app.get("/auth/status", response_model=AuthStatusResponse)
def auth_status() -> AuthStatusResponse:
    """Public - lets the frontend decide whether to show the admin-token
    Settings UI at all (see AuthStatusResponse's docstring)."""
    return AuthStatusResponse(admin_required=_configured_admin_token() is not None)


@app.post("/auth/check", response_model=AuthCheckResponse)
def auth_check(_admin: None = Depends(require_admin)) -> AuthCheckResponse:
    """Lets the Settings panel validate a just-pasted token immediately,
    rather than the person only finding out it's wrong the next time they
    try to Launch. Reaching this line at all means require_admin already
    accepted the token (or gating is off) - there is nothing left to do
    but say so."""
    return AuthCheckResponse(ok=True)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Which production model is currently loaded, and the honest
    generalization/interval numbers that go with it - the "model
    versioning" success criterion's read side (the write side is
    `models/lstm/history/<UTC-timestamp>/`, populated by
    scripts/retrain_pipeline.py on every retrain)."""
    a = _artifacts()
    m = a.metadata
    return ModelInfoResponse(
        trained_at_utc=m.get("trained_at_utc", ""),
        n_train=m.get("n_train", 0),
        n_epochs=m.get("n_epochs", 0),
        reference_leave_one_round_out_pooled_mape=m.get(
            "reference_leave_one_round_out_pooled_mape", float("nan")
        ),
        reference_leave_one_round_out_pooled_r2=m.get(
            "reference_leave_one_round_out_pooled_r2", float("nan")
        ),
        deployment_quantile_seconds=a.deployment_quantile_seconds,
        deployment_quantile_exact=a.deployment_quantile_exact,
        interval_level_pct=a.coverage_target_pct,
        interval_label="typical range",
        note=m.get("note", ""),
        mlflow_run_id=m.get("mlflow_run_id"),
    )


@app.get("/model/runs", response_model=List[ModelRunResponse])
def model_runs() -> List[ModelRunResponse]:
    """Every logged retrain run (oldest first), for the Performance tab's
    "Retrain history" timeline - reads the same MLflow tracking store
    `scripts/retrain_pipeline.py` writes to (f1qp.modeling.tracking),
    rather than re-implementing the sqlite lookup a second time the way
    the now-retired Streamlit dashboard used to (see tracking.py's own
    docstring on that gap). Returns [] (never raises) if mlflow isn't
    installed, no run has ever been logged, or the tracking store can't
    be read - "no retrain history yet" is a normal, expected state early
    in the season."""
    df = search_runs(mlruns_dir=_MLRUNS_DIR)
    if df.empty:
        return []
    df = df.sort_values("start_time").reset_index(drop=True)
    out = []
    for _, row in df.iterrows():
        params = {
            col[len("params."):]: row[col]
            for col in df.columns
            if col.startswith("params.") and pd.notna(row[col])
        }
        metrics = {
            col[len("metrics."):]: float(row[col])
            for col in df.columns
            if col.startswith("metrics.") and pd.notna(row[col])
        }
        start_time = row.get("start_time")
        out.append(
            ModelRunResponse(
                run_id=str(row.get("run_id", "")),
                start_time=start_time.isoformat() if pd.notna(start_time) else None,
                params=params,
                metrics=metrics,
            )
        )
    return out


@app.get("/model/staged", response_model=StagedModelResponse)
def model_staged() -> StagedModelResponse:
    """The Performance tab's "a new model is ready to review" panel - see
    f1qp.modeling.promote's module docstring. Public (like every other GET
    here) - only the actual promote action below is admin-gated."""
    staged = latest_staged(MODELS_DIR)
    if staged is None:
        return StagedModelResponse(has_staged=False)
    previous = staged.get("previous_summary")
    current = staged.get("current_summary")
    rows = comparison_rows(previous, current) if current else []
    return StagedModelResponse(
        has_staged=True,
        trained_at_utc=staged.get("trained_at_utc"),
        holdout_included=staged.get("holdout_included"),
        mlflow_run_id=staged.get("mlflow_run_id"),
        comparison_text=format_retrain_comparison(previous, current) if current else None,
        comparison=[
            RetrainComparisonRow(key=r["key"], previous=r["previous"], current=r["current"]) for r in rows
        ],
    )


@app.post("/model/promote", response_model=PromoteResponse)
def model_promote(_admin: None = Depends(require_admin)) -> PromoteResponse:
    """The explicit human checkpoint Damiano asked for ("auto-stage,
    one-click to promote"): swaps in whatever the automatic post-results
    retrain chain most recently staged (see f1qp.modeling.promote), then
    reloads THIS process's cached `ProductionArtifacts` immediately - see
    this module's own long-standing docstring note on why that reload
    used to require a restart. Admin-gated, same as Launch and the
    on-demand data-fetch triggers (see require_admin) - promoting a new
    model is exactly the kind of action a guest shouldn't be able to
    trigger."""
    try:
        staged = promote_staged(MODELS_DIR)
    except NoStagedModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _state["artifacts"] = load_production_artifacts()
    # Sep 16 2026 addition: mirror the newly-promoted production artifacts
    # out to Hugging Face immediately, and drop the now-stale
    # staged/latest.json pointer there too, so a later pull_all() (the
    # next redeploy/restart) doesn't resurrect an already-promoted
    # candidate - see hf_sync's module docstring. Both best-effort/
    # fail-open, same as every other hf_sync call in this codebase: a
    # sync hiccup here must never turn a successful promote into a
    # visible error - the promoted model is already live in THIS
    # process either way.
    hf_sync.push_model(MODELS_DIR)
    hf_sync.delete_path("models/lstm/staged/latest.json")
    return PromoteResponse(
        promoted=True,
        staged_trained_at_utc=staged.get("trained_at_utc", ""),
        model_trained_at_utc=_state["artifacts"].metadata.get("trained_at_utc", ""),
    )


# Sep 16 2026 addition (Damiano: "give the opportunity to launch preview
# only on 2026 and not before"): 2023-2025 are TRAINING_SEASONS (see
# f1qp.config) - the model was fit on them, so "predicting" one of those
# rounds through the live app is meaningless (it's not a real
# out-of-sample prediction) and was never the point of Preview/Launch. A
# floor, not an exact match like _FETCH_YEAR below, so a future season
# (2027+) needs no code change here when it arrives.
_MIN_PREVIEW_YEAR = 2026


def _require_live_season(year: int) -> None:
    if year < _MIN_PREVIEW_YEAR:
        raise HTTPException(
            status_code=400,
            detail=f"Preview/Launch only supports {_MIN_PREVIEW_YEAR} and later - "
            f"{year} is a training season, not a live one to predict.",
        )


@app.post("/predict/{year}/{round_number}", response_model=PredictResponse)
def predict(year: int, round_number: int) -> PredictResponse:
    """Predict every driver's qualifying time for one weekend, reading its
    already-built practice-session features from
    `data/processed/features.parquet` (must exist for this (year,
    round_number) - run scripts/download_2026.py + scripts/build_features.py
    first). Works with a partial weekend too (e.g. FP1 only so far) - see
    f1qp.serving.predict's module docstring.

    Read-only - does NOT persist a launch (see `/predict/.../launch`
    below). Free for the dashboard to call as often as it wants (e.g. a
    quick look before deciding to launch for real) without touching the
    History tab's data.

    Sep 16 2026: restricted to `_MIN_PREVIEW_YEAR` and later - see
    `_require_live_season`'s own comment just above."""
    _require_live_season(year)
    artifacts = _artifacts()
    try:
        predictions = predict_event_from_disk(year, round_number, artifacts)
        excluded = test_drivers_for_event_from_disk(year, round_number)
    except EventNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ArtifactsNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictResponse(
        year=year,
        round_number=round_number,
        n_drivers=len(predictions),
        model_trained_at_utc=artifacts.metadata.get("trained_at_utc", ""),
        excluded_test_drivers=excluded,
        predictions=_score_and_build_predictions(year, round_number, predictions),
        event_name=get_event_name(year, round_number),
    )


@app.post("/predict/{year}/{round_number}/launch", response_model=LaunchResponse)
def launch(year: int, round_number: int, _admin: None = Depends(require_admin)) -> LaunchResponse:
    """The dashboard's "Launch" / "Re-run Prediction" button: runs the
    same prediction `POST /predict/{year}/{round_number}` does, but also
    PERSISTS it via `f1qp.serving.history.record_launch` - this is what
    makes the round show up in the Prediction tab's switcher and in the
    History tab from now on. Re-launching an already-launched round
    overwrites its prior launch record (see record_launch's docstring) -
    there is exactly one "current" launched prediction per round.

    Admin-gated (Sep 13 2026, see require_admin) - Preview
    (`POST /predict/{year}/{round_number}` without `/launch`) stays open
    to everyone; only persisting a launch is restricted.

    Sep 16 2026: restricted to `_MIN_PREVIEW_YEAR` and later, same as
    Preview - see `_require_live_season`'s own comment above `predict()`."""
    _require_live_season(year)
    artifacts = _artifacts()
    try:
        predictions = predict_event_from_disk(year, round_number, artifacts)
        excluded = test_drivers_for_event_from_disk(year, round_number)
    except EventNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ArtifactsNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    launched_at_utc = datetime.now(timezone.utc).isoformat()
    record_launch(
        year=year,
        round_number=round_number,
        predictions=predictions,
        excluded_test_drivers=excluded,
        model_trained_at_utc=artifacts.metadata.get("trained_at_utc", ""),
        launched_at_utc=launched_at_utc,
    )

    return LaunchResponse(
        year=year,
        round_number=round_number,
        n_drivers=len(predictions),
        model_trained_at_utc=artifacts.metadata.get("trained_at_utc", ""),
        excluded_test_drivers=excluded,
        predictions=_score_and_build_predictions(year, round_number, predictions),
        event_name=get_event_name(year, round_number),
        launched_at_utc=launched_at_utc,
    )


def _rows_to_history_response(df) -> HistoryResponse:
    if df.empty:
        return HistoryResponse(rows=[])
    clean = df.where(pd.notna(df), None)
    return HistoryResponse(rows=[HistoryRowResponse(**row) for row in clean.to_dict(orient="records")])


@app.get("/history", response_model=HistoryResponse)
def history(season: Optional[int] = None, round_number: Optional[int] = None) -> HistoryResponse:
    """Every launched round (optionally filtered by season and/or round
    number), each row scored against whatever official result currently
    exists - the History tab's season/round filter panel reads directly
    off this. A round still awaiting quali comes back with
    has_target=False rows rather than being omitted."""
    df = prediction_history(season=season, round_number=round_number)
    return _rows_to_history_response(df)


@app.get("/history/current", response_model=List[CurrentPredictionResponse])
def history_current() -> List[CurrentPredictionResponse]:
    """The Prediction tab's switcher data: up to the 2 most recently
    launched distinct rounds (newest first) - see
    f1qp.serving.history.latest_two_launches's docstring. Empty list
    before the very first launch of the season."""
    current = latest_two_launches()
    out = []
    for entry in current:
        rows = pd.DataFrame(entry["rows"])
        rows_clean = [] if rows.empty else rows.where(pd.notna(rows), None).to_dict(orient="records")
        out.append(
            CurrentPredictionResponse(
                year=entry["year"],
                round_number=entry["round_number"],
                launched_at_utc=entry["launched_at_utc"],
                model_trained_at_utc=entry["model_trained_at_utc"],
                excluded_test_drivers=entry["excluded_test_drivers"],
                scored=entry["scored"],
                rows=[
                    HistoryRowResponse(
                        **{
                            **row,
                            "year": entry["year"],
                            "round_number": entry["round_number"],
                            "launched_at_utc": entry["launched_at_utc"],
                        }
                    )
                    for row in rows_clean
                ],
            )
        )
    return out


# --------------------------------------------------------------------- #
# Sep 2026 addition: on-demand data fetch, so the app itself can prepare a
# new round instead of Damiano running scripts/download_2026.py etc. by
# hand every time. See f1qp.serving.data_fetch's module docstring and
# f1qp.config.session_readiness's docstring for the reasoning.
#
# download_2026.py / build_features.py / extract_qualifying_targets.py
# are all 2026-only tools by design (see their own docstrings) - the
# `year` path parameter is accepted for symmetry with /predict, but only
# 2026 is actually supported here; training-season data uses
# scripts/download_historical.py instead, run by hand once, not this
# on-demand flow.
# --------------------------------------------------------------------- #
_FETCH_YEAR = 2026


@app.get("/data/readiness/{year}/{round_number}", response_model=DataReadinessResponse)
def data_readiness(year: int, round_number: int) -> DataReadinessResponse:
    """Pre-flight advice for the "fetch this round's data" confirmation -
    see f1qp.config.session_readiness's docstring. Never errors on its
    own - a lookup problem comes back as ready=True, checked=False with
    an explanatory message, never a 4xx/5xx."""
    return DataReadinessResponse(**session_readiness(year, round_number))


@app.post("/data/fetch/{year}/{round_number}", response_model=FetchJobStartedResponse)
def data_fetch(
    year: int, round_number: int, _admin: None = Depends(require_admin)
) -> FetchJobStartedResponse:
    """Kicks off download_2026.py --round + build_features.py --round in
    the background so Preview/Launch can find this round's data without
    a terminal. Poll the returned job_id via GET /data/jobs/{job_id}.

    Admin-gated (Sep 13 2026, see require_admin) - a guest previewing a
    round with no data yet gets told it isn't ready, rather than being
    offered a button that would 401 if they clicked it."""
    if year != _FETCH_YEAR:
        raise HTTPException(
            status_code=400,
            detail=f"On-demand fetch is only supported for {_FETCH_YEAR} (the live season) - "
            f"scripts/download_2026.py is {_FETCH_YEAR}-only by design.",
        )
    job_id = start_fetch_job(year, round_number)
    return FetchJobStartedResponse(job_id=job_id)


@app.post("/data/fetch-results/{year}/{round_number}", response_model=FetchJobStartedResponse)
def data_fetch_results(
    year: int, round_number: int, _admin: None = Depends(require_admin)
) -> FetchJobStartedResponse:
    """Kicks off extract_qualifying_targets.py --round in the background -
    the History tab's "Check for official result" action, meant to be
    used once a round's Q session has actually happened. Poll the
    returned job_id via GET /data/jobs/{job_id}.

    Admin-gated (Sep 13 2026, see require_admin) - viewing History stays
    open to everyone, only triggering this fetch is restricted."""
    if year != _FETCH_YEAR:
        raise HTTPException(
            status_code=400,
            detail=f"On-demand fetch is only supported for {_FETCH_YEAR} (the live season).",
        )
    job_id = start_results_job(year, round_number)
    return FetchJobStartedResponse(job_id=job_id)


@app.get("/data/jobs/{job_id}", response_model=FetchJobStatusResponse)
def data_job_status(job_id: str) -> FetchJobStatusResponse:
    job = job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id {job_id}")
    return FetchJobStatusResponse(
        status=job["status"],
        stage=job["stage"],
        year=job["year"],
        round_number=job["round_number"],
        error=job.get("error"),
        warnings=job.get("warnings") or [],
    )


# --------------------------------------------------------------------- #
# Custom frontend: replaces the Streamlit dashboard (Aug 26 2026 pivot),
# then rebuilt again in React (Aug 30 2026, Damiano's own choice via
# AskUserQuestion) - see dashboard/app.py's own docstring and
# Task_List.txt for the full history. Served by THIS service at the same
# origin as the API itself - no separate container, no CORS, no "FastAPI
# base URL" field to configure. html=True makes "/" resolve to
# index.html.
#
# Mounted LAST, after every @app.get/@app.post route above - Starlette
# matches routes in registration order, so /health, /predict/..., etc.
# all take priority over this catch-all; only a path none of them own
# falls through to a static file.
#
# Points at frontend/dist - the React app's BUILT output (Vite), not its
# source (frontend/src/*.jsx) - built by the Dockerfile's frontend-build
# stage, or by running `npm run build` in frontend/ yourself for a local
# `uvicorn --reload` run from the repo root. F1QP_FRONTEND_DIR override
# exists for the same reason every other path in this codebase has one
# (Docker's WORKDIR vs. a host checkout aren't the same directory).
# --------------------------------------------------------------------- #
_frontend_dir = Path(os.environ.get("F1QP_FRONTEND_DIR", "frontend/dist"))
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
else:
    print(
        f"Frontend directory not found at {_frontend_dir.resolve()} - the API "
        f"endpoints still work, but the dashboard UI will not be served. Run "
        f"`npm run build` inside frontend/ (or use the Dockerfile, which does "
        f"this automatically), or set F1QP_FRONTEND_DIR if running from "
        f"somewhere other than the repo root.",
        flush=True,
    )
