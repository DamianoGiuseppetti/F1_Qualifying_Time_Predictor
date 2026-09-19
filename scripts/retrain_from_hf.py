"""Sep 18 2026 addition: the "don't keep any data on my Mac" retrain pipeline.

Damiano's own framing: Render's free instance is too slow for a real
retrain (one leave-one-round-out fold took 6+ minutes there - see
f1qp.serving.data_fetch's module docstring), so retraining happens
locally. But local data quietly went stale the moment the app started
fetching rounds directly against Render + Hugging Face instead of your
Mac (see scripts/pull_from_hf.py's docstring for that whole story), and
Damiano doesn't want to keep the data parked locally between retrains
anyway. This script is the single command that replaces the multi-step
manual sequence: pull, rebuild, retrain, push, clean up after itself.

**What it does, in order:**
1. Pull the current features.parquet / qualifying_targets.parquet from
   Hugging Face straight into data/processed/ (same two files
   scripts/pull_from_hf.py pulls - this script inlines that instead of
   shelling out to it, one process, no subprocess overhead).
2. Rebuild the modeling dataset with Round 12 folded in
   (scripts.prepare_phase3_dataset.run(include_holdout=True) -
   in-process, same function f1qp.serving.data_fetch calls).
3. Retrain (scripts.retrain_pipeline.main(), in-process) - writes a
   staged candidate under models/lstm/staged/<timestamp>/, logs an
   MLflow run to local mlruns/.
4. Push the staged candidate + the MLflow run back to Hugging Face
   directly (f1qp.serving.hf_sync.push_staged / push_mlruns) - NOT via
   sync_and_push.sh, so this never rebuilds the Docker image just to
   hand a model off.
5. VERIFY each pushed path actually landed on Hugging Face (a fresh
   HfApi.list_repo_files() call, not just trusting that push_file didn't
   raise - hf_sync is deliberately fail-open/best-effort, see its module
   docstring, which is right for a background sync but wrong for "is it
   now safe to delete the only local copy"). Only if verification passes
   does step 6 run.
6. Delete the local files this run produced or pulled: features.parquet,
   qualifying_targets.parquet, phase3_dataset.parquet,
   phase3_feature_columns.json, any pulled launch JSONs, and the staged
   candidate directory under models/lstm/staged/. Leaves models/lstm's
   LIVE artifacts and the local mlruns/ tracking DB alone - those are
   the current production model and its history, not inputs this run
   pulled from Hugging Face, and other local tooling (evaluate_holdout.py,
   ad-hoc analysis) may still expect them to exist.

**If verification fails**: nothing local gets deleted, and the script
exits with a clear message saying so. The staged candidate stays on
disk, safe, exactly where retrain_pipeline.py left it - the failure mode
here is "you keep local data for one extra retrain cycle", never "you
lose a retrain you just paid several minutes for".

**Requires** HF_DATASET_REPO and HF_TOKEN (same as Render). Refuses
loudly at the very start if either is missing, rather than quietly
retraining on whatever happens to already be on disk.

Run from the repo root:

    HF_DATASET_REPO=... HF_TOKEN=... python scripts/retrain_from_hf.py

**After it finishes**: the new staged candidate is on Hugging Face, not
yet on Render (Render only pulls from Hugging Face at startup). Restart
the Render service (dashboard -> Manual Deploy -> Restart service, ~10s,
NOT a redeploy) so the running container picks it up, then use the
app's own "Promote to production" to go live - same explicit checkpoint
as every other retrain path in this project (see
f1qp.modeling.promote's module docstring).
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from f1qp.serving import hf_sync  # noqa: E402
from f1qp.modeling.promote import ARTIFACT_NAMES, latest_staged  # noqa: E402


DATA_DIR = REPO_ROOT / "data" / "processed"
LAUNCHES_DIR = REPO_ROOT / "data" / "predictions" / "launches"
MODELS_DIR = REPO_ROOT / "models" / "lstm"
MLRUNS_DIR = REPO_ROOT / "mlruns"

FEATURES_PATH = DATA_DIR / "features.parquet"
TARGETS_PATH = DATA_DIR / "qualifying_targets.parquet"
PHASE3_DATASET_PATH = DATA_DIR / "phase3_dataset.parquet"
PHASE3_COLUMNS_PATH = DATA_DIR / "phase3_feature_columns.json"


def _step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def _pull_from_hf() -> None:
    _step("1/6  Pulling current data from Hugging Face")
    before = TARGETS_PATH.stat().st_size if TARGETS_PATH.exists() else None
    hf_sync.pull_all(
        predictions_dir=LAUNCHES_DIR,
        features_path=FEATURES_PATH,
        targets_path=TARGETS_PATH,
    )
    if not TARGETS_PATH.exists():
        print("qualifying_targets.parquet was not pulled - check HF_DATASET_REPO/HF_TOKEN above.", flush=True)
        raise SystemExit(1)
    after = TARGETS_PATH.stat().st_size
    if before == after:
        print(
            "Warning: qualifying_targets.parquet is the same size as before the pull. "
            "Could be genuinely unchanged, or the pull silently failed - hf_sync only "
            "warns on failure, it doesn't raise. Check the warning output above.",
            flush=True,
        )

    import pandas as pd

    t = pd.read_parquet(TARGETS_PATH)
    rounds_2026 = sorted(t.loc[t["Year"] == 2026, "RoundNumber"].unique().tolist())
    print(f"2026 rounds now present in qualifying_targets.parquet: {rounds_2026}", flush=True)


def _rebuild_dataset() -> None:
    _step("2/6  Rebuilding the modeling dataset (Round 12 included)")
    from scripts.prepare_phase3_dataset import run as prepare_dataset

    prepare_dataset(include_holdout=True)


def _retrain() -> None:
    _step("3/6  Retraining")
    from scripts.retrain_pipeline import main as run_retrain

    run_retrain()


def _staged_paths_in_repo() -> List[str]:
    staged = latest_staged(MODELS_DIR)
    if staged is None:
        print("No staged candidate found after retrain_pipeline.py ran - something went wrong upstream.", flush=True)
        raise SystemExit(1)
    stamp = staged["timestamp"]
    paths = [f"models/lstm/staged/{stamp}/{name}" for name in (*ARTIFACT_NAMES, "staged_meta.json")]
    paths.append("models/lstm/staged/latest.json")
    return paths


def _push_to_hf() -> List[str]:
    _step("4/6  Pushing the staged candidate + MLflow run to Hugging Face")
    expected = _staged_paths_in_repo()
    hf_sync.push_staged(MODELS_DIR)
    hf_sync.push_mlruns(MLRUNS_DIR)
    return expected


def _verify_pushed(expected_staged_paths: List[str]) -> bool:
    _step("5/6  Verifying the push actually landed")
    try:
        from huggingface_hub import HfApi

        # Read directly from the environment rather than reaching into
        # hf_sync's private _token()/_repo_id() helpers - hf_sync.enabled()
        # already confirmed above that both are set.
        repo_files = set(
            HfApi(token=os.environ["HF_TOKEN"]).list_repo_files(
                repo_id=os.environ["HF_DATASET_REPO"], repo_type="dataset"
            )
        )
    except Exception as exc:
        print(f"Could not verify (listing the repo failed: {exc!r}) - treating as NOT verified.", flush=True)
        return False

    missing = [p for p in expected_staged_paths if p not in repo_files]
    mlflow_db_present = any(p.startswith("mlruns/") and p.endswith("mlflow.db") for p in repo_files)

    if missing:
        print(f"Missing on Hugging Face after push: {missing}", flush=True)
        return False
    if not mlflow_db_present:
        print("mlruns/mlflow.db not found on Hugging Face after push.", flush=True)
        return False
    print("All staged-candidate files and the MLflow store are confirmed present on Hugging Face.", flush=True)
    return True


def _cleanup() -> None:
    _step("6/6  Deleting the local transient copies")
    staged = latest_staged(MODELS_DIR)

    removed = []
    for path in (FEATURES_PATH, TARGETS_PATH, PHASE3_DATASET_PATH, PHASE3_COLUMNS_PATH):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    if LAUNCHES_DIR.is_dir():
        for f in LAUNCHES_DIR.glob("*.json"):
            f.unlink()
            removed.append(str(f))
    if staged is not None and staged["dir"].is_dir():
        shutil.rmtree(staged["dir"])
        removed.append(str(staged["dir"]))
    pointer = MODELS_DIR / "staged" / "latest.json"
    if pointer.exists():
        pointer.unlink()
        removed.append(str(pointer))

    print(f"Removed {len(removed)} local path(s):", flush=True)
    for r in removed:
        print(f"  - {r}", flush=True)
    print(
        "\nLocal models/lstm's live (currently-promoted) artifacts and the local mlruns/ "
        "tracking DB were left alone on purpose - see this script's module docstring.",
        flush=True,
    )


def main() -> None:
    if not hf_sync.enabled():
        print(
            "HF_DATASET_REPO and/or HF_TOKEN are not set - refusing to run (would otherwise "
            "silently retrain on whatever happens to already be on disk). Set both, same "
            "values Render uses, and re-run:\n\n"
            "    HF_DATASET_REPO=... HF_TOKEN=... python scripts/retrain_from_hf.py\n",
            flush=True,
        )
        raise SystemExit(1)

    started = datetime.now(timezone.utc).isoformat()
    print(f"retrain_from_hf.py starting {started}", flush=True)

    _pull_from_hf()
    _rebuild_dataset()
    _retrain()
    expected = _push_to_hf()
    ok = _verify_pushed(expected)

    if not ok:
        print(
            "\nPush could not be verified - leaving all local files in place so nothing is "
            "lost. Re-run this script (it will just retrain again), or check "
            "HF_DATASET_REPO/HF_TOKEN and push manually before deleting anything yourself.",
            flush=True,
        )
        raise SystemExit(1)

    _cleanup()
    print(
        "\nDone. The new staged candidate is on Hugging Face, not yet on Render - "
        "restart the Render service (Manual Deploy > Restart service, ~10s, not a "
        "redeploy) so it picks the candidate up, then Promote to production in the app.",
        flush=True,
    )


if __name__ == "__main__":
    main()
