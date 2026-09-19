"""Phase 3 diagnostic: leave-one-round-out cross-validation within 2026.

Why this exists: the main 80/20 validation split in train_baseline.py puts
only ~2 of the 11 2026 training weekends into era 1's validation slice -
too small a sample to trust a claim that the model "works on 2026" (one
unlucky or lucky weekend, e.g. the verified Las Vegas rain weekend, could
swing that number either way). This holds out each 2026 round in turn,
trains on everything else (all of 2023-2025 plus the other 2026 rounds -
the same setup as the real Round 13 deployment: train on all prior data,
predict one unseen round), and reports both the per-round breakdown and
the pooled result across all 11 folds.

A round with a much worse MAPE than the rest is worth a look on its own -
check whether it lines up with a known disruption (rain, red flags,
mechanical issues) the way Sao Paulo 2024 and Las Vegas 2025 did in the
gap-target investigation - rather than averaging it away.

Run from anywhere - paths are resolved relative to this file:

    python scripts/loro_2026_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from f1qp.modeling.baseline import leave_one_round_out_cv
from f1qp.modeling.dataset import pivot_to_weekend_features, resolve_feature_columns

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"


def main() -> None:
    dataset_path = DATA_DIR / "phase3_dataset.parquet"
    feature_meta_path = DATA_DIR / "phase3_feature_columns.json"

    merged = pd.read_parquet(dataset_path)
    with open(feature_meta_path) as f:
        feature_meta = json.load(f)
    feature_cols = feature_meta["feature_cols"]

    fcols = resolve_feature_columns(merged)
    wide_df = pivot_to_weekend_features(merged, feature_cols, fcols)

    # era values are whatever f1qp.features.build encodes - 1 is 2026 per
    # docs/feature_engineering.md ("era - 0 for 2023-2025, 1 for 2026").
    result = leave_one_round_out_cv(wide_df, feature_cols, fcols, formulation="gap", era_value=1)

    print(f"Leave-one-round-out CV within era={result['era_value']} "
          f"({result['formulation']} formulation), "
          f"{len(result['per_round'])} rounds:\n")

    rows = []
    for round_number, metrics in sorted(result["per_round"].items()):
        rows.append((round_number, metrics["mape"], metrics["r2"], metrics["n_test"]))
        print(f"  Round {round_number:>3}: MAPE={metrics['mape']:6.3f}%  "
              f"R2={metrics['r2']:6.3f}  n_test={metrics['n_test']}")

    mapes = [r[1] for r in rows]
    print(f"\nAcross rounds: min MAPE={min(mapes):.3f}%  max MAPE={max(mapes):.3f}%  "
          f"spread={max(mapes) - min(mapes):.3f} points")

    pooled = result["pooled"]
    print(f"\nPooled across all {pooled['n_test']} 2026 test rows: "
          f"MAPE={pooled['mape']:.3f}%  R2={pooled['r2']:.3f}")
    print("(This is the number to trust for \"does it work on 2026\" - "
          "every 2026 round was held out and tested exactly once, always "
          "trained on all prior data, mirroring the real Round 13 setup.)")

    worst_round = max(rows, key=lambda r: r[1])
    if worst_round[1] > 2 * pooled["mape"]:
        print(f"\nRound {worst_round[0]} is more than 2x the pooled MAPE "
              f"({worst_round[1]:.3f}% vs {pooled['mape']:.3f}%) - worth checking "
              f"whether it lines up with a known disruption (rain, red flags, "
              f"mechanical issues) rather than averaging it away.")


if __name__ == "__main__":
    main()
