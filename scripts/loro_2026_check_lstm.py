"""Phase 3, LSTM diagnostic: leave-one-round-out CV within 2026, for the LSTM.

Why this exists: the LSTM's first real run (scripts/train_lstm.py) didn't
beat the XGBoost baseline on the standard 80/20 split (1.445% vs 1.090%
val MAPE). But that same kind of split already produced ONE misleadingly
good number before - XGBoost's own era-1 (2026) result on the 80/20 split
looked great (0.579% MAPE) purely because only ~2 of the 11 2026 weekends
landed in that validation slice, and the honest leave-one-round-out number
turned out to be 1.153%, much closer to era 0's. The LSTM's "doesn't beat
baseline" verdict deserves the same scrutiny before it gets treated as
settled - especially since each LSTM training run took well under a
second on real data, so running it 11 times instead of once is cheap
insurance, not a real cost.

See f1qp.modeling.lstm_model.leave_one_round_out_cv_lstm's own docstring
for exactly how each fold is built (own internal train/val split for early
stopping, own freshly-fit imputer/scaler - never reusing statistics fit
on a fold that includes the round being tested).

Run from anywhere - paths are resolved relative to this file:

    python scripts/loro_2026_check_lstm.py

Prints as it goes (flush=True) - one line per fold as that fold finishes
training and gets evaluated, not batched until the end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from f1qp.modeling.dataset import pivot_to_weekend_features, resolve_feature_columns
from f1qp.modeling.lstm_model import leave_one_round_out_cv_lstm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"

# Reference numbers from the already-confirmed real runs (Aug 23 2026) -
# printed for comparison only, never loaded live.
XGB_LORO_POOLED_MAPE = 1.153  # XGBoost leave-one-round-out, pooled
LSTM_8020_VAL_MAPE = 1.445  # LSTM's first run, standard 80/20 split


def main() -> None:
    dataset_path = DATA_DIR / "phase3_dataset.parquet"
    feature_meta_path = DATA_DIR / "phase3_feature_columns.json"

    print(f"Loading dataset from {dataset_path}", flush=True)
    merged = pd.read_parquet(dataset_path)
    with open(feature_meta_path) as f:
        feature_meta = json.load(f)
    feature_cols = feature_meta["feature_cols"]

    fcols = resolve_feature_columns(merged)
    print("Pivoting to weekend-level...", flush=True)
    wide_df = pivot_to_weekend_features(merged, feature_cols, fcols)

    n_before = len(wide_df)
    wide_df = wide_df[wide_df["split"] != "holdout"].reset_index(drop=True)
    print(f"Excluded {n_before - len(wide_df)} holdout row(s) (Zandvoort)", flush=True)

    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    print(f"Usable driver-weekends: {len(wide_df)}", flush=True)

    print("\nRunning leave-one-round-out CV for the LSTM within era=1 (2026) - "
          "11 folds, each training its own model from scratch:\n", flush=True)
    result = leave_one_round_out_cv_lstm(wide_df, feature_cols, fcols, era_value=1)

    print(f"\n{len(result['per_round'])} rounds tested:")
    rows = sorted(result["per_round"].items())
    mapes = [m["mape"] for _, m in rows]
    print(f"Across rounds: min MAPE={min(mapes):.3f}%  max MAPE={max(mapes):.3f}%  "
          f"spread={max(mapes) - min(mapes):.3f} points")

    pooled = result["pooled"]
    print(f"\nPooled across all {pooled['n_test']} 2026 test rows: "
          f"MAPE={pooled['mape']:.3f}%  R2={pooled['r2']:.3f}")
    print("(This is the number to trust for \"does the LSTM work on 2026\" - "
          "every round held out and tested exactly once, mirroring the real "
          "Round 13 setup, not one random 80/20 slice.)")

    print(f"\nFor comparison - LSTM's first run, standard 80/20 split: "
          f"{LSTM_8020_VAL_MAPE:.3f}% val MAPE")
    print(f"For comparison - XGBoost leave-one-round-out pooled: "
          f"{XGB_LORO_POOLED_MAPE:.3f}% MAPE")
    if pooled["mape"] < XGB_LORO_POOLED_MAPE:
        print("-> On this more robust comparison, the LSTM DOES beat the XGBoost "
              "baseline.")
    else:
        print("-> The LSTM still does NOT beat the XGBoost baseline on this more "
              "robust comparison - the earlier 80/20 result wasn't just split-luck "
              "against the LSTM specifically.")

    worst_round = max(rows, key=lambda kv: kv[1]["mape"])
    if worst_round[1]["mape"] > 2 * pooled["mape"]:
        print(f"\nRound {worst_round[0]} is more than 2x the pooled MAPE "
              f"({worst_round[1]['mape']:.3f}% vs {pooled['mape']:.3f}%) - worth "
              f"checking whether it lines up with a known disruption, same as the "
              f"XGBoost LORO's Round 10 (Spa) finding.")


if __name__ == "__main__":
    main()
