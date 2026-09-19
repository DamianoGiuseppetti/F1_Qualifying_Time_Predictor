"""Sep 2026 addition: lets the app fetch a new round's data itself instead
of requiring scripts/download_2026.py / build_features.py /
extract_qualifying_targets.py to be run by hand first - Damiano: "I don't
want to run the py scripts outside everytime. The app should be able to
do everything."

Deliberately runs the EXISTING, already-verified scripts as subprocesses
rather than re-implementing their logic here - the exact same code path
Damiano would run by hand, just started by the API instead of a terminal.
Each job runs in a background thread so the triggering HTTP request
returns a job_id immediately (a real fetch can take anywhere from ~30
seconds to a couple of minutes - FastF1 network calls, not something an
API response should block on) - see f1qp.api.main's /data/... endpoints
for how the frontend polls this, and f1qp.config.session_readiness for
the pre-flight warning shown before a practice-data fetch is even
started.

Sep 16 2026 fix (Damiano: retraining after Round 14's official result
crashed the whole Render free instance - confirmed via Render's own
"Instance failed" event, timed exactly to the retraining stage): every
step used to be a `subprocess.run` of the matching script (see `Step`
below), on purpose - the exact same code path Damiano would run by hand,
just started by the API. That's still true for the two FastF1-network
steps (`downloading`/`extracting_results`), which genuinely benefit from
process isolation (a stuck network call can't wedge the API). It was
never true for `rebuilding_dataset`/`retraining` in `start_results_job`
though: those do no network I/O, and subprocess-ing them means a SECOND
full copy of torch/pandas/numpy gets imported into a SECOND process,
stacked on top of the API server's own already-loaded copies - on
Render's free 512MB instance that doubling is what actually OOM-killed
the container, not the retrain computation itself (retrain_pipeline.py's
own docstring: "under a minute total on a laptop CPU"). Those two steps
now run via an in-process `func` callable instead (see `Step.func` and
`_run_func_step`) - same interpreter, same already-loaded imports, no
second process, no doubling. `scripts.prepare_phase3_dataset.run()` and
`scripts.retrain_pipeline.main()` are imported lazily inside the two
`_run_*` wrappers below, not at this module's top, so a request that
never reaches these steps doesn't pay for importing torch either.

In-memory job store only (no database, matching every other part of this
project's "plain files, no infra beyond what's needed" approach - see
f1qp.serving.history's module docstring). A job's status is lost on an
API restart, which is fine: the underlying scripts are themselves
idempotent (an already-downloaded round is skipped, the feature/target
files are simply rebuilt), so a lost job just means clicking fetch again.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from f1qp.serving import hf_sync

# Sep 16 2026 fix (Damiano testing Round 14/Madrid: "downloading failed
# (exit 2): ... can't open file '/usr/local/lib/python3.11/scripts/
# download_2026.py'"): `Path(__file__).resolve().parents[3]` only lands on
# the actual repo root when f1qp is running from a live source checkout
# (src/f1qp/serving/data_fetch.py -> parents[3] = repo root) - true for
# local `uvicorn --reload` dev, but NOT true in any of this project's own
# Dockerfiles (root, deploy/render/, deploy/huggingface/), which all
# install via a normal `pip install .`. That copies this file into
# site-packages instead (.../site-packages/f1qp/serving/data_fetch.py), so
# parents[3] silently became /usr/local/lib/python3.11 there - exactly the
# bogus prefix in the error above. F1QP_REPO_ROOT lets every Dockerfile
# say explicitly where `scripts/` actually lives in that image (see each
# Dockerfile's own ENV block); the parents[3] guess stays as the fallback
# for local dev, where it's still correct and no env var is set.
REPO_ROOT = Path(os.environ.get("F1QP_REPO_ROOT") or Path(__file__).resolve().parents[3])

# So the lazy `from scripts.X import Y` imports inside `_run_prepare_phase3_dataset`/
# `_run_retrain_pipeline` below resolve - `scripts/` has no __init__.py (a
# plain namespace package is enough), it just needs REPO_ROOT on sys.path,
# same as running `python scripts/whatever.py` from the repo root already
# gives you for free via sys.path[0].
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Ceiling per SCRIPT, not per job - a hung FastF1 call (bad network, a
# stalled response) should fail the job eventually rather than leave it
# "running" forever for the frontend to poll against. Applies to
# subprocess steps only - the in-process `func` steps below have no
# equivalent ceiling (see `_run_func_step`).
_STEP_TIMEOUT_SECONDS = 900


def _run_subprocess_step(job_id: str, step: "Step") -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, *step.script_args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=_STEP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, f"{step.stage} timed out after {_STEP_TIMEOUT_SECONDS}s"
    with _jobs_lock:
        _jobs[job_id]["log"].append({
            "stage": step.stage,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-4000:],
        })
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        return False, f"{step.stage} failed (exit {proc.returncode}): " + " | ".join(tail)
    return True, ""


def _run_func_step(job_id: str, step: "Step") -> Tuple[bool, str]:
    """Sep 16 2026 addition: runs `step.func` directly in this worker
    thread instead of spawning a subprocess - see this module's docstring
    for why. No separate timeout ceiling (unlike `_run_subprocess_step`) -
    a hang here would hang this thread only, not the API's request-
    handling threads, and the two callers of this (`_run_prepare_phase3_dataset`,
    `_run_retrain_pipeline`) are both pure local compute, not network
    calls, so a genuine hang isn't the failure mode this needs to guard
    against the way it is for the FastF1 subprocess steps."""
    try:
        step.func()
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["log"].append({"stage": step.stage, "error": repr(exc)})
        return False, f"{step.stage} failed: {exc!r}"
    with _jobs_lock:
        _jobs[job_id]["log"].append({"stage": step.stage, "returncode": 0})
    return True, ""


def _run_step(job_id: str, step: "Step") -> Tuple[bool, str]:
    with _jobs_lock:
        _jobs[job_id]["stage"] = step.stage
    if step.func is not None:
        return _run_func_step(job_id, step)
    return _run_subprocess_step(job_id, step)


def _worker(job_id: str, steps: List["Step"]) -> None:
    for step in steps:
        ok, error = _run_step(job_id, step)
        if not ok:
            if step.required:
                with _jobs_lock:
                    _jobs[job_id]["status"] = "error"
                    _jobs[job_id]["error"] = error
                return
            # Sep 16 2026 addition (see start_results_job's docstring): a
            # best-effort step (the automatic retrain chain) failing must
            # never turn an otherwise-successful job into a scary "error"
            # for the frontend - the thing the user actually asked for
            # (the official result / the practice data) already landed.
            # Recorded as a warning instead, surfaced by job_status() so
            # it's still visible somewhere rather than silently lost.
            with _jobs_lock:
                _jobs[job_id].setdefault("warnings", []).append(f"{step.stage}: {error}")
            continue
        if step.on_success is not None:
            # Sep 13 2026 addition: mirror the file this step just
            # rebuilt out to the Hugging Face dataset repo (see
            # f1qp.serving.hf_sync) so it survives a Space redeploy.
            # Best-effort by design (hf_sync itself never raises) - still
            # wrapped here too so a completely unexpected failure in this
            # callback can never flip an otherwise successful job into
            # "error".
            try:
                step.on_success()
            except Exception:
                pass
    with _jobs_lock:
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["stage"] = "done"


class Step:
    """One step of a background job (see `_start`/`_worker`), run either
    as a subprocess (`script_args`) or in-process (`func`) - exactly one
    of the two must be given. `func` (Sep 16 2026 addition) is for steps
    that do no network I/O and shouldn't pay for a second
    torch/pandas/numpy import in a second process - see this module's
    docstring.

    `required=False` (Sep 16 2026 addition, for the automatic retrain
    chain in `start_results_job`) means a failure here is recorded as a
    warning rather than failing the whole job - the step this gates
    (rebuilding the modeling dataset / retraining) is a nice-to-have on
    top of whatever the job was actually asked to do, not the reason it
    was started."""

    def __init__(
        self,
        stage: str,
        script_args: Optional[List[str]] = None,
        func: Optional[Callable[[], None]] = None,
        on_success: Optional[Callable[[], None]] = None,
        required: bool = True,
    ) -> None:
        if (script_args is None) == (func is None):
            raise ValueError(f"Step {stage!r} needs exactly one of script_args or func")
        self.stage = stage
        self.script_args = script_args
        self.func = func
        self.on_success = on_success
        self.required = required


def _start(steps: List[Step], year: int, round_number: int) -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "stage": "queued",
            "year": year,
            "round_number": round_number,
            "error": None,
            "warnings": [],
            "log": [],
        }
    thread = threading.Thread(target=_worker, args=(job_id, steps), daemon=True)
    thread.start()
    return job_id


def start_fetch_job(year: int, round_number: int) -> str:
    """Download + build features for one round, in that order - the two
    steps Preview/Launch need before they can find this round's data.
    Reuses scripts/download_2026.py and scripts/build_features.py's
    --round flag (Aug 30 2026) exactly as Damiano would run them by hand -
    except the download step, see below.

    Sep 16 2026 fix (Damiano testing Round 14: this stayed on "Downloading
    practice laps" for 10+ minutes): download_2026.py's plain --round
    APPENDS to its R1-R12 default rather than replacing it, and none of
    this project's deploy images bake data/raw/ in (deliberately, to keep
    them small - see each Dockerfile's own comment), so every fresh
    container had nothing cached and re-downloaded the entire R1-R12
    season before ever reaching the round actually asked for. --only-round
    (see download_2026.py's own docstring) skips that baseline entirely -
    this is the one caller that needs it; build_features.py keeps plain
    --round since its own pass over R1-R12 is pure local file I/O (cheap)
    and its carry-forward logic already handles missing raw laps.

    Sep 13 2026 addition: once both steps succeed, pushes the rebuilt
    features.parquet out to the Hugging Face dataset repo (a no-op
    locally - see f1qp.serving.hf_sync) so a fetch done directly against
    the Space stays durable across a redeploy.
    """
    features_path = Path(
        os.environ.get("F1QP_FEATURES_PATH", REPO_ROOT / "data" / "processed" / "features.parquet")
    )
    steps = [
        Step("downloading", ["scripts/download_2026.py", "--only-round", str(round_number)]),
        Step(
            "building_features",
            ["scripts/build_features.py", "--round", str(round_number)],
            on_success=lambda: hf_sync.push_file(features_path, "processed/features.parquet"),
        ),
    ]
    return _start(steps, year, round_number)


def _run_prepare_phase3_dataset() -> None:
    """In-process replacement for `subprocess.run(["scripts/prepare_phase3_dataset.py",
    "--include-holdout"])` - see this module's docstring for why. Imported
    lazily (not at module top) so importing f1qp.serving.data_fetch itself
    never pulls in pandas just for this."""
    from scripts.prepare_phase3_dataset import run as run_prepare_phase3_dataset

    run_prepare_phase3_dataset(include_holdout=True)


def _run_retrain_pipeline() -> None:
    """In-process replacement for `subprocess.run(["scripts/retrain_pipeline.py"])` -
    see this module's docstring for why. Imported lazily (not at module
    top) so importing f1qp.serving.data_fetch itself never pulls in torch
    just for this."""
    from scripts.retrain_pipeline import main as run_retrain_pipeline

    run_retrain_pipeline()


def _push_retrain_artifacts() -> None:
    """`retraining` step's on_success (Sep 16 2026): mirrors what the
    in-process retrain just produced - the newly staged candidate AND the
    MLflow run it logged - out to the Hugging Face dataset repo, same
    "durable without Damiano's Mac" reasoning as every other hf_sync call
    in this codebase (see f1qp.serving.hf_sync's module docstring). Reads
    F1QP_MODELS_DIR/F1QP_MLRUNS_DIR fresh here rather than caching them at
    import time, matching hf_sync's own env-vars-read-fresh convention -
    a Space/Render secret set after the container starts still takes
    effect without a code change."""
    models_dir = Path(os.environ.get("F1QP_MODELS_DIR", REPO_ROOT / "models" / "lstm"))
    mlruns_dir = Path(os.environ.get("F1QP_MLRUNS_DIR", REPO_ROOT / "mlruns"))
    hf_sync.push_staged(models_dir)
    hf_sync.push_mlruns(mlruns_dir)


def start_results_job(year: int, round_number: int) -> str:
    """Pull the official qualifying result for one round, once its Q
    session has actually happened - scripts/extract_qualifying_targets.py
    --round. Kept separate from start_fetch_job on purpose: this can only
    ever succeed AFTER qualifying, whereas the practice-data fetch above
    happens BEFORE it - see f1qp.config.session_readiness's docstring on
    why this action doesn't get the same pre-flight timing warning.

    Sep 13 2026 addition: once the step succeeds, pushes the rebuilt
    qualifying_targets.parquet out to the Hugging Face dataset repo (a
    no-op locally - see f1qp.serving.hf_sync), same reasoning as
    start_fetch_job above.

    Sep 16 2026 addition (Damiano's report: "I don't see the button to
    retrain the model after the prediction... I think that not showing
    the official results stops the application from the followed path of
    retrain the model" - and his own "auto-stage, one-click to promote"
    design): this is exactly the gap that let Round 13's official result
    sit in qualifying_targets.parquet without ever reaching the model -
    nobody remembered to run scripts/prepare_phase3_dataset.py
    --include-holdout + scripts/retrain_pipeline.py by hand afterward. Now
    chained automatically onto this same job, so "check for official
    result" succeeding can never again silently stop short of producing a
    retrain candidate. Both chained steps are `required=False`: extracting
    the official result is this job's actual purpose and must still fail
    loudly on its own if FastF1 has nothing yet; a hiccup in rebuilding
    the dataset or retraining afterward must not turn an already-
    successful result fetch into a scary error banner - see `_worker`'s
    warnings handling. Either way, retrain_pipeline.py only ever produces
    a STAGED candidate (models/lstm/staged/<timestamp>/) - nothing here
    changes what the API is actually serving; see f1qp.modeling.promote's
    module docstring for the explicit "Promote to production" step that
    does.

    Sep 16 2026 fix: `rebuilding_dataset`/`retraining` used to be
    subprocess steps (like `extracting_results` still is) - changed to
    in-process `func` steps (`_run_prepare_phase3_dataset`,
    `_run_retrain_pipeline`) because subprocess-ing them was what crashed
    Render's free instance - see this module's own top-level docstring.
    `retraining`'s on_success (`_push_retrain_artifacts`) is new too: the
    staged candidate + MLflow run this produces now live only on
    whichever container ran this job, so they need to reach Hugging Face
    the same way a Launch or a fetched round already does, or they're
    gone on the next redeploy/restart, before anyone gets a chance to
    even see the "Promote to production" panel.
    """
    targets_path = Path(
        os.environ.get("F1QP_TARGETS_PATH", REPO_ROOT / "data" / "processed" / "qualifying_targets.parquet")
    )
    steps = [
        Step(
            "extracting_results",
            script_args=["scripts/extract_qualifying_targets.py", "--round", str(round_number)],
            on_success=lambda: hf_sync.push_file(targets_path, "processed/qualifying_targets.parquet"),
        ),
        Step(
            "rebuilding_dataset",
            func=_run_prepare_phase3_dataset,
            required=False,
        ),
        Step(
            "retraining",
            func=_run_retrain_pipeline,
            required=False,
            on_success=_push_retrain_artifacts,
        ),
    ]
    return _start(steps, year, round_number)


def job_status(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return None if job is None else dict(job)
