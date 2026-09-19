"""Sep 16 2026 addition: the missing direction in the Hugging Face sync.

f1qp.serving.hf_sync covers app <-> Hugging Face (the running API pushes
what it writes at runtime, and pulls it all back at startup - see that
module's docstring). It never covered Hugging Face -> Damiano's Mac,
because nothing needed it to - local development used local files, and
the deployed app used its own.

That gap became a real bug the day round-14 data started being fetched
entirely through the deployed app (Damiano: "I need to use the retrain
in the application because i need to free my storage"): the app's own
features.parquet / qualifying_targets.parquet moved ahead of whatever
was last sitting in Damiano's local data/processed/, and a LOCAL retrain
(scripts/prepare_phase3_dataset.py + scripts/retrain_pipeline.py) silently
trained on the stale, smaller local dataset - no error, just a retrain
that quietly missed the newest round(s). Confirmed for real: a local
retrain "with round 14" on Sep 16 2026 reproduced the exact same numbers
as the previous production model, because the local qualifying_targets.parquet
/ features.parquet only went up to Round 13.

This script closes that gap the same way f1qp.serving.hf_sync.pull_all
does for the API at startup: pulls whatever is currently on the Hugging
Face dataset repo down into this local checkout's data/processed/ (and,
optionally, models/lstm/ + mlruns/) BEFORE a local retrain, so the local
retrain actually sees what the app has already fetched.

Run before ANY local retrain, from the repo root:

    HF_DATASET_REPO=... HF_TOKEN=... python scripts/pull_from_hf.py
    python scripts/prepare_phase3_dataset.py --include-holdout
    python scripts/retrain_pipeline.py

--with-model / --with-mlruns additionally restore the live production
model and MLflow tracking store from Hugging Face - not needed just to
pick up new round data, but useful if the local checkout's models/lstm
or mlruns/ have also drifted from whatever the app is actually serving
(e.g. after a promote done straight from the deployed app).

Same fail-open reasoning as hf_sync itself: requires both HF_DATASET_REPO
and HF_TOKEN to be set (as env vars, same names Render uses), and does
nothing destructive - it only ever ADDS/overwrites the specific files
Hugging Face has, never deletes anything local pull_all doesn't know
about.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from f1qp.serving import hf_sync  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Also pull models/lstm/ (live production artifacts + any staged candidate) from Hugging Face.",
    )
    parser.add_argument(
        "--with-mlruns",
        action="store_true",
        help="Also pull mlruns/ (the MLflow tracking store) from Hugging Face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not hf_sync.enabled():
        print(
            "HF_DATASET_REPO and/or HF_TOKEN are not set - nothing to pull. "
            "Set both (same values Render uses) and re-run.",
            flush=True,
        )
        raise SystemExit(1)

    predictions_dir = REPO_ROOT / "data" / "predictions" / "launches"
    features_path = REPO_ROOT / "data" / "processed" / "features.parquet"
    targets_path = REPO_ROOT / "data" / "processed" / "qualifying_targets.parquet"
    models_dir = (REPO_ROOT / "models" / "lstm") if args.with_model else None
    mlruns_dir = (REPO_ROOT / "mlruns") if args.with_mlruns else None

    print(f"Pulling from {os.environ.get('HF_DATASET_REPO')} ...", flush=True)
    before = targets_path.stat().st_size if targets_path.exists() else None

    hf_sync.pull_all(
        predictions_dir=predictions_dir,
        features_path=features_path,
        targets_path=targets_path,
        models_dir=models_dir,
        mlruns_dir=mlruns_dir,
    )

    after = targets_path.stat().st_size if targets_path.exists() else None
    if before == after:
        print(
            "Warning: qualifying_targets.parquet is the same size after the pull as before - "
            "either nothing changed, or the pull silently failed (hf_sync logs a warning on "
            "failure rather than raising - check the output above for one).",
            flush=True,
        )
    print(
        f"Done. features.parquet -> {features_path}\n"
        f"      qualifying_targets.parquet -> {targets_path}"
        + ("\n      models/lstm/ -> " + str(models_dir) if models_dir else "")
        + ("\n      mlruns/ -> " + str(mlruns_dir) if mlruns_dir else ""),
        flush=True,
    )


if __name__ == "__main__":
    main()
