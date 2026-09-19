"""Phase 3, LSTM step: train and validate the sequence model.

Loads the same prepared dataset and the same weekend-level train/val split
as scripts/train_baseline.py - deliberately the identical split, not a
fresh random one, so "does the LSTM beat the baseline" (Task_List.txt's own
requirement) is an apples-to-apples comparison on the same held-out
weekends, not two different validation sets that happen to both be called
"val". Zandvoort (holdout) is excluded the same way, for the same reason:
Phase 5's offline test, never touched for model selection.

Target: `gap_final` (practice-to-qualifying gap), reconstructed to absolute
time before scoring - see f1qp.modeling.lstm_model's module docstring for
why, and why this replaces Task_List.txt's original three-masked-head plan.

Feature scaling is fit ONLY on the train split's real (non-padded)
timesteps (see f1qp.modeling.sequences.FeatureScaler) and applied
identically to val - never refit on val, which would leak val's own
distribution into what the model was "trained" to expect as normal.

Run from anywhere - paths are resolved relative to this file:

    python scripts/train_lstm.py

Expect this to run in well under 10 minutes on a laptop CPU - see the
printed wall-clock time at the end, and scripts/train_lstm.py's own console
output for the per-epoch breakdown if it runs slower than that on your
machine (worth reporting back if so, since the model is tiny - it would
point at a data-loading or environment issue, not the model itself).

Every stage below prints as it happens (flush=True throughout, including
inside f1qp.modeling.lstm_model.train_lstm's per-epoch loop) rather than
buffering output until the whole script finishes - so a print appearing on
screen means that stage genuinely started/finished, not that the script is
about to dump everything at once at the end. If it goes quiet for more
than a few seconds anywhere between "Loading dataset" and the first
printed epoch line, that gap - not the epoch table itself - is where
something is stuck.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from f1qp.modeling.dataset import pivot_to_weekend_features, resolve_feature_columns
from f1qp.modeling.lstm_model import train_lstm
from f1qp.modeling.sequences import FeatureImputer, FeatureScaler, build_lstm_sequences

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "lstm"

# Reference numbers from the already-confirmed real runs (Aug 23 2026) -
# printed for comparison only, never loaded live. See Task_List.txt /
# phase3-planning.md for the full context behind each.
BASELINE_VAL_MAPE = 1.090  # XGBoost gap formulation, 80/20 split
LORO_POOLED_MAPE = 1.153  # leave-one-round-out within 2026, pooled


def main() -> None:
    dataset_path = DATA_DIR / "phase3_dataset.parquet"
    feature_meta_path = DATA_DIR / "phase3_feature_columns.json"

    print(f"Loading dataset from {dataset_path}", flush=True)
    merged = pd.read_parquet(dataset_path)
    with open(feature_meta_path) as f:
        feature_meta = json.load(f)
    feature_cols = feature_meta["feature_cols"]
    print(f"Loaded {len(merged)} practice-session rows, {len(feature_cols)} features", flush=True)

    fcols = resolve_feature_columns(merged)
    print("Pivoting to weekend-level (one row per driver-weekend)...", flush=True)
    wide_df = pivot_to_weekend_features(merged, feature_cols, fcols)
    print(f"Pivoted: {wide_df.shape}", flush=True)

    n_before = len(wide_df)
    wide_df = wide_df[wide_df["split"] != "holdout"].reset_index(drop=True)
    print(f"Excluded {n_before - len(wide_df)} holdout row(s) (Zandvoort)", flush=True)

    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    print(f"Usable driver-weekends (has a final_quali_time): {len(wide_df)}", flush=True)

    print("Building LSTM input sequences (padding mask + static context)...", flush=True)
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)

    train_mask_rows = batch.split == "train"
    val_mask_rows = batch.split == "val"
    print(f"Train sequences: {train_mask_rows.sum()}  Val sequences: {val_mask_rows.sum()}", flush=True)

    print("Fitting feature imputer on train split...", flush=True)
    imputer = FeatureImputer.fit(batch.X[train_mask_rows], batch.mask[train_mask_rows])
    n_imputed = imputer.n_imputed_per_feature
    if n_imputed.sum() > 0:
        print("Real sessions in the TRAIN split with an individually-missing feature "
              "value (median-imputed, fit on train only - see f1qp.modeling.sequences "
              "docstring for why this differs from a missing session):", flush=True)
        for feat, count in sorted(zip(feature_cols, n_imputed), key=lambda t: -t[1]):
            if count > 0:
                print(f"  {feat}: {int(count)} occurrence(s)", flush=True)
    X_imputed = imputer.transform(batch.X, batch.mask)

    print("Fitting feature scaler on train split...", flush=True)
    scaler = FeatureScaler.fit(X_imputed[train_mask_rows], batch.mask[train_mask_rows])
    X_scaled = scaler.transform(X_imputed, batch.mask)

    print("Starting training (see f1qp.modeling.lstm_model.train_lstm's own live "
          "per-epoch output below)...\n", flush=True)
    result = train_lstm(
        train_X=X_scaled[train_mask_rows],
        train_lengths=batch.lengths[train_mask_rows],
        train_static=batch.static[train_mask_rows],
        train_y_gap=batch.y_gap[train_mask_rows],
        val_X=X_scaled[val_mask_rows],
        val_lengths=batch.lengths[val_mask_rows],
        val_static=batch.static[val_mask_rows],
        val_y_abs=batch.y_abs[val_mask_rows],
        val_practice_reference=batch.practice_reference[val_mask_rows],
        val_era=batch.era[val_mask_rows],
        n_features=len(feature_cols),
    )
    # train_lstm already printed one line per epoch live, plus its own
    # "Done." summary line - nothing to reprint here, just the extra
    # detail (era breakdown, comparison to baseline) it doesn't have.

    best_metrics = result.history[result.best_epoch - 1]
    print("\nEra-stratified at best epoch:")
    for era_value in sorted(best_metrics.val_mape_by_era):
        print(f"  era={era_value}: MAPE={best_metrics.val_mape_by_era[era_value]:.3f}%  "
              f"R2={best_metrics.val_r2_by_era[era_value]:.3f}")

    print(f"\nFor comparison - XGBoost baseline (80/20 split): {BASELINE_VAL_MAPE:.3f}% val MAPE")
    print(f"For comparison - leave-one-round-out pooled (2026 only): {LORO_POOLED_MAPE:.3f}% MAPE")
    if result.best_val_mape < BASELINE_VAL_MAPE:
        print("-> LSTM beats the XGBoost baseline on this split.")
    else:
        print("-> LSTM does NOT beat the XGBoost baseline on this split - "
              "the baseline stays the production fallback per the project brief.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), MODELS_DIR / "lstm_gap_final_quali_time.pt")
    np.savez(
        MODELS_DIR / "feature_scaler.npz",
        mean=scaler.mean, std=scaler.std, feature_cols=np.array(feature_cols),
    )
    history_df = pd.DataFrame([
        {
            "epoch": m.epoch, "train_loss": m.train_loss,
            "val_mape": m.val_mape, "val_r2": m.val_r2,
            "elapsed_seconds": m.elapsed_seconds,
            **{f"val_mape_era{k}": v for k, v in m.val_mape_by_era.items()},
            **{f"val_r2_era{k}": v for k, v in m.val_r2_by_era.items()},
        }
        for m in result.history
    ])
    history_df.to_csv(MODELS_DIR / "learning_curve.csv", index=False)

    print(f"\nSaved model, scaler, and learning curve to {MODELS_DIR}")


if __name__ == "__main__":
    main()
