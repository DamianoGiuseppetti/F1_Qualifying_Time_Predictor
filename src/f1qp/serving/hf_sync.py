"""Sep 13 2026 addition - resolves the ONE thing SETUP.md flagged as still
open after the Hugging Face Spaces deployment was built: "nothing this
container writes at runtime survives a redeploy." Damiano's own framing:
"The storage should be into hugging face without be in my local storage
anymore" - so a real Launch, a `/data/fetch` job, and a "check for
official result" click must all stay durable WITHOUT depending on
Damiano's Mac (running `sync_and_push.sh`) to carry that history forward
on every deploy, the way SETUP.md's old "accept it" default assumed.

**What this does**: mirrors what this app writes at runtime -
`data/predictions/launches/*.json`, `data/processed/features.parquet`,
`data/processed/qualifying_targets.parquet` - to a private Hugging Face
Hub DATASET repo (free, git-based storage). A Dataset repo, not the
Space's own repo, on purpose: pushing to a Dataset never triggers a Space
rebuild the way pushing to the Space repo would - see sync_and_push.sh's
own docstring on how slow/heavy a rebuild is (installing torch).

**Sep 16 2026 correction**: `models/lstm` and `mlruns` used to be
excluded here entirely ("only ever change via a manual retrain +
sync_and_push.sh, Aug 24 2026 decision"). That assumption broke once
f1qp.serving.data_fetch.start_results_job started chaining an automatic
in-process retrain onto every "check for official result" call, and
f1qp.api.main's `POST /model/promote` started running from inside the
live app too - a staged candidate, a promoted model, or a logged MLflow
run that only ever exists on Render's own ephemeral disk is lost on the
next redeploy or restart, same durability gap this module was built to
close for launches/features/targets. `push_model`/`push_staged`/
`push_mlruns` below (and `pull_all`'s optional `models_dir`/`mlruns_dir`
params) now cover them too - called from `_push_retrain_artifacts` right
after an in-process retrain, and from `f1qp.api.main`'s `model_promote()`
right after a promote. Damiano's Mac + `sync_and_push.sh` is still what
bakes the FIRST copy of a model into a fresh image; this is what keeps a
model trained by the running app itself from being a dead end.

**Fail-open, same pattern as f1qp.api.main.require_admin**: every
function here is a no-op unless BOTH `HF_DATASET_REPO` (e.g.
"damiano9801/f1qp-space-data") and `HF_TOKEN` (a Hugging Face user access
token with Write access to that repo) are set. Local docker-compose sets
neither, so this is zero behavior change there - exactly the same
frictionless single-user workflow as before this module existed. Only a
deployment that sets both (the Space, via SETUP.md's setup step) gets
real persistence. Both are read fresh on every call, not cached at import
time, matching `_configured_admin_token`'s own reasoning: a Space secret
set (or changed) after the container starts still takes effect without a
code change or restart.

**Best-effort, never breaks the request that triggered it**: every
push/pull swallows its own exceptions (network hiccup, the dataset repo
not created yet, huggingface_hub not installed) and logs a warning
instead of raising - a durability sync failing must never turn a
successful Launch or a successful data fetch into a visible error; the
data is still safe on local disk either way, just not yet mirrored out.
`huggingface_hub` itself is imported lazily, inside each function, so a
deployment that never sets the two env vars above (local docker-compose,
every existing test) never needs the package installed at all.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _repo_id() -> Optional[str]:
    repo_id = os.environ.get("HF_DATASET_REPO", "").strip()
    return repo_id or None


def _token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    return token or None


def enabled() -> bool:
    """True only once both HF_DATASET_REPO and HF_TOKEN are set - see
    module docstring. Read fresh every call rather than cached."""
    return _repo_id() is not None and _token() is not None


def push_file(local_path: Path, path_in_repo: str) -> None:
    """Upload one file to the dataset repo, overwriting whatever was
    there for that path. A no-op when disabled (see `enabled`) or when
    `local_path` doesn't exist (nothing to push). Best-effort: see module
    docstring - any failure here is logged and swallowed, never raised.
    """
    if not enabled():
        return
    local_path = Path(local_path)
    if not local_path.exists():
        return
    try:
        from huggingface_hub import HfApi

        HfApi(token=_token()).upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=_repo_id(),
            repo_type="dataset",
            commit_message=f"sync {path_in_repo}",
        )
    except Exception as exc:  # pragma: no cover - network/HF-side failure
        logger.warning(
            "hf_sync: push of %s to %s failed (%s) - continuing with the local copy only, "
            "will retry on the next write.",
            path_in_repo,
            _repo_id(),
            exc,
        )


def _copy_tree(src: Path, dest: Path) -> None:
    """Merge `src` into `dest` (dest may already exist and have content -
    e.g. models/lstm baked in by sync_and_push.sh) - files present in both
    are overwritten with the (fresher, from Hugging Face) `src` copy,
    files only in `dest` are left alone."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)


def pull_all(
    predictions_dir: Path,
    features_path: Path,
    targets_path: Path,
    models_dir: Optional[Path] = None,
    mlruns_dir: Optional[Path] = None,
) -> None:
    """Called once at API startup (see f1qp.api.main's lifespan), BEFORE
    the production model/artifacts are read: restores whatever this app
    itself most recently wrote at runtime - every launch JSON, the
    current features.parquet / qualifying_targets.parquet - from the
    dataset repo, so a freshly (re)deployed Space picks up exactly where
    the last one left off instead of resetting to whatever
    sync_and_push.sh last baked into the image.

    `models_dir`/`mlruns_dir` (Sep 16 2026 addition, optional - see module
    docstring's correction) restore the live model, staged candidate, and
    MLflow tracking store the same way, ONLY if the caller passes them -
    main.py always does in practice, but keeping them optional means a
    caller (or test) that only cares about the original three files
    doesn't pay for the extra download.

    A no-op when disabled (see `enabled`). A repo or file that doesn't
    exist yet (the very first deploy, before anything has ever been
    pushed) is a normal, expected state, not an error - falls back to
    whatever the image already has baked in, same fallback every other
    best-effort path in this module uses.
    """
    if not enabled():
        return
    allow_patterns = ["predictions/launches/*.json", "processed/*.parquet"]
    if models_dir is not None:
        allow_patterns.append("models/lstm/**")
    if mlruns_dir is not None:
        allow_patterns.append("mlruns/**")
    try:
        from huggingface_hub import snapshot_download

        snapshot_dir = snapshot_download(
            repo_id=_repo_id(),
            repo_type="dataset",
            token=_token(),
            allow_patterns=allow_patterns,
        )
    except Exception as exc:  # pragma: no cover - network/HF-side failure, or nothing pushed yet
        logger.warning(
            "hf_sync: pull from %s failed or nothing has been pushed there yet (%s) - "
            "using what the image already has.",
            _repo_id(),
            exc,
        )
        return

    snapshot_root = Path(snapshot_dir)

    launches_src = snapshot_root / "predictions" / "launches"
    if launches_src.is_dir():
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        for f in launches_src.glob("*.json"):
            shutil.copy2(f, predictions_dir / f.name)

    for name, dest in (
        ("features.parquet", Path(features_path)),
        ("qualifying_targets.parquet", Path(targets_path)),
    ):
        src = snapshot_root / "processed" / name
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    if models_dir is not None:
        models_src = snapshot_root / "models" / "lstm"
        if models_src.is_dir():
            _copy_tree(models_src, Path(models_dir))

    if mlruns_dir is not None:
        mlruns_src = snapshot_root / "mlruns"
        if mlruns_src.is_dir():
            _copy_tree(mlruns_src, Path(mlruns_dir))


def push_model(models_dir: Path) -> None:
    """Mirror the 4 LIVE production artifacts (models_dir/*.pt,
    *.npz, *.json - see f1qp.modeling.promote.ARTIFACT_NAMES) out to the
    dataset repo. Sep 16 2026 addition - see module docstring's
    correction. Called from f1qp.api.main's `POST /model/promote`, right
    after a successful `promote_staged()`."""
    models_dir = Path(models_dir)
    from f1qp.modeling.promote import ARTIFACT_NAMES

    for name in ARTIFACT_NAMES:
        push_file(models_dir / name, f"models/lstm/{name}")


def push_staged(models_dir: Path) -> None:
    """Mirror the latest STAGED (not yet promoted) retrain candidate out
    to the dataset repo, so a container restart between "retrain" and
    "promote" doesn't lose the candidate the Performance tab is showing -
    see module docstring's correction. Called as the `retraining` step's
    on_success in f1qp.serving.data_fetch.start_results_job. A no-op if
    there's nothing staged."""
    models_dir = Path(models_dir)
    from f1qp.modeling.promote import ARTIFACT_NAMES, latest_staged

    staged = latest_staged(models_dir)
    if staged is None:
        return
    stamp = staged["timestamp"]
    staged_dir = staged["dir"]
    for name in [*ARTIFACT_NAMES, "staged_meta.json"]:
        push_file(staged_dir / name, f"models/lstm/staged/{stamp}/{name}")
    # The pointer file itself, last - so a pull mid-push never lands on a
    # pointer naming a candidate whose files haven't all arrived yet.
    push_file(models_dir / "staged" / "latest.json", "models/lstm/staged/latest.json")


def push_mlruns(mlruns_dir: Path) -> None:
    """Mirror the local MLflow tracking store (mlflow.db + artifacts/) out
    to the dataset repo - same reasoning as `push_model`: a retrain done
    straight from the running app writes its MLflow run to that
    container's own ephemeral disk, previously never durable anywhere
    else. Walks the whole tree rather than naming files explicitly since
    MLflow itself decides the artifacts/ layout, not something this
    module should have to track."""
    mlruns_dir = Path(mlruns_dir)
    if not mlruns_dir.is_dir():
        return
    for path in mlruns_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(mlruns_dir)
            push_file(path, f"mlruns/{rel.as_posix()}")


def delete_path(path_in_repo: str) -> None:
    """Delete one file from the dataset repo - used right after a promote
    to remove the now-stale `models/lstm/staged/latest.json` pointer, so
    a later `pull_all` doesn't resurrect an already-promoted candidate's
    pointer (the candidate's own 4 files get correctly overwritten by
    that same promote's `push_model` call regardless - this only cleans
    up the pointer that says "still pending"). A no-op when disabled;
    "nothing to delete" (already gone, or never pushed) is swallowed same
    as every other best-effort path here, not treated as a failure."""
    if not enabled():
        return
    try:
        from huggingface_hub import HfApi

        HfApi(token=_token()).delete_file(
            path_in_repo=path_in_repo,
            repo_id=_repo_id(),
            repo_type="dataset",
            commit_message=f"remove {path_in_repo}",
        )
    except Exception as exc:  # pragma: no cover - network/HF-side failure, or already gone
        logger.warning(
            "hf_sync: delete of %s in %s failed or it was already gone (%s) - continuing.",
            path_in_repo,
            _repo_id(),
            exc,
        )
