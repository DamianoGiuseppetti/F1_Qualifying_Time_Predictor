"""Phase 3, steps 1-3: assemble the modeling dataset from features.parquet
and qualifying_targets.parquet, build the weekend-level train/val/holdout
split, and reduce the feature set.

The .py, repo-ready version of
notebooks/01_phase3_data_assembly_split_feature_reduction.ipynb - same
logic, moved into f1qp.modeling.dataset so it's importable, testable, and
reusable from scripts/train_baseline.py instead of copy-pasted.

Run from anywhere - paths are resolved relative to this file, not the
current working directory:

    python scripts/prepare_phase3_dataset.py

Phase 5 addition (Aug 25 2026) - `--include-holdout`:

    python scripts/prepare_phase3_dataset.py --include-holdout

By default (no flag) Round 12 (Zandvoort/Olanda) is held out of train/val
entirely - this is the mode scripts/evaluate_holdout.py's offline test
needs, since it evaluates the CURRENT production model against a round it
never trained on. Pass --include-holdout only AFTER that offline test has
run, to fold Round 12 into the normal train/val split ahead of the final
pre-Round-13 retrain (scripts/retrain_pipeline.py) - see
f1qp.modeling.dataset.build_weekend_split's docstring and
scripts/evaluate_holdout.py's module docstring for the full sequencing.
Running --include-holdout BEFORE the offline test silently makes that test
meaningless (it would be scoring the model against data it already
trained on); evaluate_holdout.py checks for and refuses this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from f1qp.modeling.dataset import (
    CORRELATION_THRESHOLD,
    DROPPED_FEATURES,
    HOLDOUT_ROUND,
    HOLDOUT_YEAR,
    add_practice_reference_and_gaps,
    assemble_dataset,
    build_weekend_split,
    compute_correlation_matrix,
    find_high_correlation_pairs,
    get_feature_columns,
    resolve_feature_columns,
    resolve_target_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help=(
            "Fold Round 12 (Zandvoort/Olanda) into the normal train/val "
            "split instead of holding it out. Only pass this AFTER "
            "scripts/evaluate_holdout.py's offline test has run - see this "
            "script's module docstring."
        ),
    )
    return parser.parse_args()


def run(include_holdout: bool) -> None:
    """The actual work, split out from `main()` (Sep 16 2026, paired with
    the in-process retrain fix in f1qp.serving.data_fetch) so
    f1qp.serving.data_fetch.start_results_job can call this directly in
    the API process instead of `subprocess.run`-ing this file as a
    script. Running it in-process means it shares the interpreter's
    already-loaded pandas/numpy instead of re-importing them in a brand
    new process on top of the running API - the exact thing that was
    doubling memory use and crashing Render's 512MB free instance mid-
    retrain (see start_results_job's own docstring for the full story).
    `main()` below is now a thin CLI wrapper around this, unchanged for
    anyone still running `python scripts/prepare_phase3_dataset.py` by
    hand."""
    holdout_arg = None if include_holdout else (HOLDOUT_YEAR, HOLDOUT_ROUND)
    if include_holdout:
        print(f"Holdout mode: DISABLED (--include-holdout) - Round {HOLDOUT_ROUND} goes into train/val")
    else:
        print(f"Holdout mode: Round {HOLDOUT_ROUND} ({HOLDOUT_YEAR}) held out as the offline-test set")

    features_path = DATA_DIR / "features.parquet"
    targets_path = DATA_DIR / "qualifying_targets.parquet"

    features_df = pd.read_parquet(features_path)
    targets_df = pd.read_parquet(targets_path)
    print(f"Loaded features.parquet {features_df.shape}, "
          f"qualifying_targets.parquet {targets_df.shape}")

    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    print(f"Resolved feature columns: {fcols}")
    print(f"Resolved target columns:  {tcols}")

    # Step 1 - data assembly
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)
    print(f"\nAssembled: {merged.shape}")
    print(f"Rows with no Q1 target at all (no match / DNS / DSQ): {(~merged['has_Q1']).sum()}")

    # Step 2 - train/val/holdout split
    merged = build_weekend_split(merged, fcols, holdout=holdout_arg)
    print("\nRows per split, per era:")
    print(merged.groupby(["split", fcols.era]).size().unstack(fill_value=0))

    # Step 3 - feature reduction
    feature_cols = get_feature_columns(merged, dropped=DROPPED_FEATURES)
    print(f"\nWorking feature set ({len(feature_cols)}), "
          f"after dropping {DROPPED_FEATURES}:")
    print(feature_cols)

    train_mask = merged["split"] == "train"
    corr_matrix = compute_correlation_matrix(merged, feature_cols, train_mask)
    pairs_df = find_high_correlation_pairs(corr_matrix, threshold=CORRELATION_THRESHOLD)
    print(f"\nFeature pairs with |r| >= {CORRELATION_THRESHOLD}: {len(pairs_df)}")
    if len(pairs_df):
        print(pairs_df.to_string(index=False))

    merged = add_practice_reference_and_gaps(merged, fcols)

    output_path = DATA_DIR / "phase3_dataset.parquet"
    merged.to_parquet(output_path, index=False)

    feature_meta_path = DATA_DIR / "phase3_feature_columns.json"
    with open(feature_meta_path, "w") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "dropped_now": DROPPED_FEATURES,
                "holdout_included": include_holdout,
                "high_correlation_pairs": pairs_df.to_dict(orient="records"),
                "column_map": {
                    "year": fcols.year,
                    "round_number": fcols.round_number,
                    "driver": fcols.driver,
                    "session": fcols.session,
                    "is_sprint": fcols.is_sprint,
                    "era": fcols.era,
                },
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {output_path}")
    print(f"Saved: {feature_meta_path}")


def main() -> None:
    args = parse_args()
    run(args.include_holdout)


if __name__ == "__main__":
    main()
