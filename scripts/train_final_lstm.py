"""Phase 3, LSTM step: train the FINAL production model on ALL available data.

This is Task_List.txt's "Train on 2023-2025 + 2026 R1-R11 (excluding the
Zandvoort holdout)" item. Everything up to this point (the 80/20 run, the
leave-one-round-out check) existed to answer two questions: does the
architecture generalize, and roughly how many epochs does it need. Both
are now answered - see FINAL_TRAIN_EPOCHS below - so this script trains
the one model that will actually be used for Zandvoort's Phase 5 offline
test and Round 13 (Monza) predictions, on every non-holdout driver-weekend
at once rather than the ~80% a train/val split would leave it.

FINAL_TRAIN_EPOCHS is a FIXED constant, not something this script decides
at runtime - deliberately. Early stopping needs a held-out validation set
to know when to stop; carving one out here would shrink the final model's
training data purely to re-answer a question the leave-one-round-out
check already answered more robustly (11 folds instead of 1). The value
below is grounded in two independent real-data observations from Aug 24
2026 (see Task_List.txt / phase3-planning.md for the full LORO output):

  - The LSTM's first real run (standard 80/20 split, ~1,284 training
    sequences) had its best validation MAPE at epoch 8.
  - Across leave_one_round_out_cv_lstm's 11 real folds (each trained on
    ~1,580 sequences - close to this script's ~1,602), the per-fold best
    epoch was mostly 4, with several folds at 7 and one at 8. Max was 8.

Both signals agree at 8. Not the most common single value (4), but the
upper end actually observed to still help rather than hurt in every real
fold plus the standalone run - a defensible, evidence-grounded choice
over either guessing higher (no fold ever needed more) or matching only
the mode (would ignore the folds and the standalone run that clearly
benefited from more).

No val split also means no fresh held-out metric for THIS specific model
- that's expected, not a gap to fix. The number to trust for "what should
I expect from this model on an unseen 2026 round" remains the
leave-one-round-out pooled result (1.053% MAPE, R² 0.989) - printed below
for reference, never recomputed here.

Feature imputation and scaling are fit on ALL the training data used here
(not a train-only subset) - there's no leakage concern in doing so,
because nothing in this script evaluates against a held-out set at all;
the genuinely held-out set (Zandvoort) is untouched by this script
entirely and reserved for Phase 5.

Run from anywhere - paths are resolved relative to this file:

    python scripts/train_final_lstm.py

Prints live per-epoch progress (flush=True), same convention as
scripts/train_lstm.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from f1qp.modeling.dataset import pivot_to_weekend_features, resolve_feature_columns
from f1qp.modeling.lstm_model import train_final_model
from f1qp.modeling.sequences import FeatureImputer, FeatureScaler, build_lstm_sequences

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "lstm"

FINAL_TRAIN_EPOCHS = 8  # see module docstring for the real-data reasoning

# Reference numbers from already-confirmed real runs (Aug 23-24 2026) -
# printed for context only, never loaded live.
XGB_LORO_POOLED_MAPE = 1.153
LSTM_LORO_POOLED_MAPE = 1.053
LSTM_LORO_POOLED_R2 = 0.989


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
    print("Pivoting to weekend-level...", flush=True)
    wide_df = pivot_to_weekend_features(merged, feature_cols, fcols)

    n_before = len(wide_df)
    wide_df = wide_df[wide_df["split"] != "holdout"].reset_index(drop=True)
    print(f"Excluded {n_before - len(wide_df)} holdout row(s) (Zandvoort) - "
          f"reserved for Phase 5, never touched by this script", flush=True)

    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    print(f"Training on ALL {len(wide_df)} usable driver-weekends "
          f"(no train/val split - see module docstring for why)", flush=True)

    print("Building LSTM input sequences...", flush=True)
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)

    print("Fitting feature imputer on all training data...", flush=True)
    imputer = FeatureImputer.fit(batch.X, batch.mask)
    n_imputed = imputer.n_imputed_per_feature
    if n_imputed.sum() > 0:
        print("Sessions with an individually-missing feature value "
              "(median-imputed):", flush=True)
        for feat, count in sorted(zip(feature_cols, n_imputed), key=lambda t: -t[1]):
            if count > 0:
                print(f"  {feat}: {int(count)} occurrence(s)", flush=True)
    X_imputed = imputer.transform(batch.X, batch.mask)

    print("Fitting feature scaler on all training data...", flush=True)
    scaler = FeatureScaler.fit(X_imputed, batch.mask)
    X_scaled = scaler.transform(X_imputed, batch.mask)

    print(f"\nStarting final training - {FINAL_TRAIN_EPOCHS} fixed epochs, "
          f"no validation split (see module docstring)...\n", flush=True)
    result = train_final_model(
        train_X=X_scaled,
        train_lengths=batch.lengths,
        train_static=batch.static,
        train_y_gap=batch.y_gap,
        n_features=len(feature_cols),
        n_epochs=FINAL_TRAIN_EPOCHS,
    )

    print(f"\nFor reference - leave-one-round-out pooled result (the number to "
          f"trust for expected real-world performance): "
          f"LSTM {LSTM_LORO_POOLED_MAPE:.3f}% MAPE / R² {LSTM_LORO_POOLED_R2:.3f}, "
          f"vs XGBoost {XGB_LORO_POOLED_MAPE:.3f}% MAPE", flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lstm_final_production.pt"
    preprocessing_path = MODELS_DIR / "final_preprocessing.npz"
    metadata_path = MODELS_DIR / "final_model_metadata.json"

    torch.save(result.model.state_dict(), model_path)
    np.savez(
        preprocessing_path,
        imputer_median=imputer.median,
        scaler_mean=scaler.mean,
        scaler_std=scaler.std,
        feature_cols=np.array(feature_cols),
        static_cols=np.array(batch.static_cols),
    )
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": len(wide_df),
        "n_epochs": FINAL_TRAIN_EPOCHS,
        "final_train_loss": result.train_loss_history[-1],
        "total_seconds": result.total_seconds,
        "target": "gap_final (reconstructed to final_quali_time via + practice_reference)",
        "reference_leave_one_round_out_pooled_mape": LSTM_LORO_POOLED_MAPE,
        "reference_leave_one_round_out_pooled_r2": LSTM_LORO_POOLED_R2,
        "reference_xgboost_loro_pooled_mape": XGB_LORO_POOLED_MAPE,
        "note": "No fresh held-out metric exists for this specific model by "
                "design (trained on all non-holdout data, no val split). "
                "Trust the reference LORO numbers above for what to expect "
                "on an unseen round; Zandvoort (Phase 5) is the next real "
                "test of this exact artifact.",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved final model to {model_path}", flush=True)
    print(f"Saved preprocessing stats (imputer + scaler) to {preprocessing_path}", flush=True)
    print(f"Saved run metadata to {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
