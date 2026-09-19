"""Sep 16 2026 addition: the explicit human checkpoint half of Damiano's
"auto-stage, one-click to promote" design for post-weekend retraining.

The gap this closes: Round 13's official result landed in
qualifying_targets.parquet just fine, but nobody remembered to run
scripts/prepare_phase3_dataset.py --include-holdout + scripts/
retrain_pipeline.py by hand afterward, so the model never actually saw
that data. f1qp.serving.data_fetch.start_results_job now chains straight
into both scripts automatically once a round's official result is
fetched (see that module's docstring) - but retraining automatically does
NOT mean serving the result automatically. Damiano's own design note:
"The output of that automatic run is staged, not live... the app keeps
serving the previous production model until you act... One explicit
action ('Promote to production') swaps the live model. If you don't act,
last week's model keeps serving." This module is that swap.

**Where the staged candidate lives**: scripts/retrain_pipeline.py now
writes its output to `models/lstm/staged/<UTC-timestamp>/` (the 4
artifact files + a `staged_meta.json` with the before/after comparison
and the MLflow run id) instead of directly into `models/lstm/` - see that
script's own module docstring for the reasoning. A `models/lstm/staged/
latest.json` pointer file names the most recent one; `latest_staged`
resolves it, `promote_staged` consumes it.

**What promoting actually does**: archives whatever the 4 live
`models/lstm/*` artifacts currently are into `models/lstm/history/
<UTC-timestamp>/` (unchanged from what retrain_pipeline.py itself used to
do before this feature, just moved here - production's own history stays
inspectable across a promotion, not just across a raw retrain), copies
the staged candidate's 4 files into `models/lstm/` in their place, then
deletes the `staged/latest.json` pointer - once promoted, a candidate IS
production, not still "pending". The caller (f1qp.api.main's
`POST /model/promote`) is responsible for reloading the running process's
cached `ProductionArtifacts` afterward (`load_production_artifacts()`) so
the new weights actually take effect without a restart - seeing this
through was main.py's own long-standing TODO before this feature existed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

ARTIFACT_NAMES = [
    "lstm_final_production.pt",
    "final_preprocessing.npz",
    "final_model_metadata.json",
    "conformal_intervals.json",
]


class NoStagedModelError(Exception):
    """Raised by `promote_staged` when there's nothing staged to promote -
    either no retrain has ever run, or the last one already got promoted."""


def latest_staged(models_dir: Path) -> Optional[dict]:
    """The most recent staged retrain candidate, or None if there isn't
    one - a normal, common state (most of the time, between race
    weekends), not an error. Returns the merged contents of
    `staged_meta.json` plus `timestamp` and `dir` (the staged artifacts'
    own directory, as a `Path`)."""
    pointer_path = models_dir / "staged" / "latest.json"
    if not pointer_path.exists():
        return None
    try:
        with open(pointer_path) as f:
            pointer = json.load(f)
        stamp = pointer["timestamp"]
    except Exception:
        return None

    staged_dir = models_dir / "staged" / stamp
    meta_path = staged_dir / "staged_meta.json"
    if not staged_dir.is_dir() or not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return {"timestamp": stamp, "dir": staged_dir, **meta}


def promote_staged(models_dir: Path) -> dict:
    """Swap the current production artifacts for the latest staged
    candidate. Raises `NoStagedModelError` (never a bare
    FileNotFoundError) if there's nothing staged. Returns the same dict
    `latest_staged` would have returned for the candidate just promoted -
    the caller's response can report exactly what went live."""
    staged = latest_staged(models_dir)
    if staged is None:
        raise NoStagedModelError(
            "No staged retrain candidate found - the automatic post-results "
            "chain (or a manual scripts/retrain_pipeline.py run) needs to "
            "produce one before there's anything to promote."
        )
    staged_dir = staged["dir"]

    existing = [models_dir / name for name in ARTIFACT_NAMES if (models_dir / name).exists()]
    if existing:
        from datetime import datetime, timezone

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = models_dir / "history" / stamp
        dest.mkdir(parents=True, exist_ok=True)
        for path in existing:
            shutil.copy2(path, dest / path.name)

    models_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        src = staged_dir / name
        if src.exists():
            shutil.copy2(src, models_dir / name)

    pointer_path = models_dir / "staged" / "latest.json"
    if pointer_path.exists():
        pointer_path.unlink()

    return staged
