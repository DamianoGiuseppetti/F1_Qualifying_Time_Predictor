"""Phase 3, LSTM step: learning curves + SHAP error analysis.

Task_List.txt: "Learning curves + SHAP error analysis; sanity-check
against the small-sample risk flagged during planning (~1,500 independent
driver-weekend sequences feeding a 64-hidden-unit LSTM)." Two independent
checks, both diagnostic - neither touches any saved model or calibration
artifact:

1. LEARNING CURVE DIAGNOSIS: reads the already-saved
   models/lstm/learning_curve.csv (from the standalone 80/20 run,
   scripts/train_lstm.py) and quantifies the overfitting pattern already
   observed by eye in that run's real output (Aug 24 2026: train_loss kept
   falling to epoch 33 while val MAPE's best was epoch 8) - see
   f1qp.modeling.interpretability.diagnose_learning_curve for exactly what
   "overfitting signature" means here. Also saves a plot (train_loss +
   val_mape vs epoch, best epoch marked) for a quick visual check.

2. SHAP ERROR ANALYSIS: re-fits the standalone 80/20 LSTM (same seed,
   split, and defaults as scripts/train_lstm.py - self-contained, same
   pattern as scripts/calibrate_conformal.py) and explains its
   predictions with shap.KernelExplainer, then compares feature
   attributions between the model's WORST and BEST predictions on its own
   val set - do the same features that dominate XGBoost's importance
   ranking (gap_to_session_best, track_temp_mean, rainfall_share - see
   Task_List.txt's baseline entry) also show up disproportionately in the
   LSTM's highest-error cases?

   Scope is narrowed to non-sprint (3-real-session) val rows only, and
   explained on the GAP-scale prediction rather than reconstructed
   absolute time - see f1qp.modeling.interpretability's module docstring
   for the full reasoning behind both choices (they're about SHAP
   mechanics, not this script, so they live there).

Both checks are independent - learning curve diagnosis only reads a CSV
that's already on disk; the SHAP step does its own fresh (cheap, ~1s)
model fit and doesn't depend on the learning curve step having run.

Run from anywhere - paths are resolved relative to this file:

    python scripts/analyze_lstm.py

Requires the `shap` package (added to requirements.txt this step -
`pip install -r requirements.txt` first if not already present).

The SHAP step's runtime is the real unknown here - this hasn't been run on
real data yet. KernelExplainer cost scales with background size x
samples-per-instance x instances explained; the constants below (30
background rows, 500 samples/instance, 40 explained instances total) are
chosen to stay in the "a few minutes at most" range on a laptop CPU for
this tiny (~25k-parameter) model, but if it takes dramatically longer than
that in practice, that's itself useful information worth reporting back -
same convention as every other script in this project (see
scripts/train_lstm.py's own docstring). Prints as it goes (flush=True);
shap.KernelExplainer also shows its own built-in progress bar by default.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless - this script only ever saves files, never shows a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from f1qp.modeling.dataset import pivot_to_weekend_features, resolve_feature_columns
from f1qp.modeling.interpretability import (
    aggregate_shap_by_feature,
    diagnose_learning_curve,
    flatten_features_for_shap,
    select_error_buckets,
)
from f1qp.modeling.lstm_model import train_lstm
from f1qp.modeling.sequences import FeatureImputer, FeatureScaler, build_lstm_sequences

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "lstm"

RANDOM_SEED = 42
SHAP_BACKGROUND_SIZE = 30
SHAP_NSAMPLES = 500
SHAP_BUCKET_SIZE = 20  # each of low-error / high-error


def _analyze_learning_curve() -> None:
    history_path = MODELS_DIR / "learning_curve.csv"
    print("\n--- Learning curve diagnosis ---", flush=True)
    print(f"Loading {history_path}", flush=True)
    if not history_path.exists():
        print(
            f"  SKIPPED: {history_path} not found - run scripts/train_lstm.py "
            f"first (this reads that script's saved output).",
            flush=True,
        )
        return

    history_df = pd.read_csv(history_path)
    diagnosis = diagnose_learning_curve(history_df)
    print(
        f"  best_epoch={diagnosis.best_epoch}  final_epoch={diagnosis.final_epoch}  "
        f"epochs_trained_past_best={diagnosis.epochs_trained_past_best}",
        flush=True,
    )
    print(
        f"  train_loss: {diagnosis.train_loss_at_best_epoch:.4f} (at best epoch) -> "
        f"{diagnosis.train_loss_at_final_epoch:.4f} (at final epoch)  "
        f"kept_falling_after_best={diagnosis.train_loss_kept_falling_after_best}",
        flush=True,
    )
    print(
        f"  val_mape: {diagnosis.best_val_mape:.3f}% (best) -> "
        f"{diagnosis.val_mape_worst_after_best:.3f}% (worst after best)  "
        f"drift={diagnosis.val_mape_drift_after_best:+.3f} points",
        flush=True,
    )
    if diagnosis.overfitting_signature:
        print(
            "  -> CONFIRMS the small-sample overfitting risk flagged during "
            "planning: train loss kept improving past the best epoch while "
            "val MAPE got WORSE, not just plateaued. Early stopping (already "
            "in place) correctly rolled back to the best epoch's weights - "
            "this is the safeguard working as designed, not a new problem.",
            flush=True,
        )
    else:
        print(
            "  -> Does NOT show the classic overfitting signature (train "
            "loss falling while val MAPE gets worse) - either training "
            "stopped right around the useful point, or val MAPE plateaued "
            "rather than reversing.",
            flush=True,
        )

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history_df["epoch"], history_df["train_loss"], color="tab:blue", label="train_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train_loss (Huber)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(history_df["epoch"], history_df["val_mape"], color="tab:red", label="val_MAPE")
    ax2.set_ylabel("val MAPE (%)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.axvline(diagnosis.best_epoch, color="gray", linestyle="--", linewidth=1)
    ax1.set_title(f"LSTM learning curve (best epoch={diagnosis.best_epoch})")
    fig.tight_layout()
    plot_path = MODELS_DIR / "learning_curve.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  Saved plot to {plot_path}", flush=True)

    diagnosis_path = MODELS_DIR / "learning_curve_diagnosis.json"
    with open(diagnosis_path, "w") as f:
        json.dump(diagnosis.to_dict(), f, indent=2)
    print(f"  Saved diagnosis to {diagnosis_path}", flush=True)


def _run_shap_error_analysis(wide_df, feature_cols, fcols) -> None:
    print("\n--- SHAP error analysis ---", flush=True)
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)
    train_idx = batch.split == "train"
    val_idx = batch.split == "val"

    imputer = FeatureImputer.fit(batch.X[train_idx], batch.mask[train_idx])
    X_imputed = imputer.transform(batch.X, batch.mask)
    scaler = FeatureScaler.fit(X_imputed[train_idx], batch.mask[train_idx])
    X_scaled = scaler.transform(X_imputed, batch.mask)

    print(
        "Re-fitting the standalone 80/20 LSTM (same seed/split/defaults as "
        "scripts/train_lstm.py) to get a model to explain...",
        flush=True,
    )
    result = train_lstm(
        train_X=X_scaled[train_idx],
        train_lengths=batch.lengths[train_idx],
        train_static=batch.static[train_idx],
        train_y_gap=batch.y_gap[train_idx],
        val_X=X_scaled[val_idx],
        val_lengths=batch.lengths[val_idx],
        val_static=batch.static[val_idx],
        val_y_abs=batch.y_abs[val_idx],
        val_practice_reference=batch.practice_reference[val_idx],
        val_era=batch.era[val_idx],
        n_features=len(feature_cols),
        verbose=False,
    )
    model = result.model
    model.eval()

    # Non-sprint (3-real-session) rows only - see
    # f1qp.modeling.interpretability's module docstring for why.
    non_sprint = batch.lengths == 3
    print(
        f"Restricting SHAP analysis to non-sprint (3-real-session) rows: "
        f"{int((non_sprint & train_idx).sum())} of {int(train_idx.sum())} train rows, "
        f"{int((non_sprint & val_idx).sum())} of {int(val_idx.sum())} val rows usable.",
        flush=True,
    )

    train_pool_idx = np.where(train_idx & non_sprint)[0]
    val_pool_idx = np.where(val_idx & non_sprint)[0]
    if len(val_pool_idx) < 4:
        print(
            f"  SKIPPED: only {len(val_pool_idx)} non-sprint val rows "
            f"available - too few for a meaningful low/high error comparison.",
            flush=True,
        )
        return

    n_features = len(feature_cols)

    def predict_fn(flat_batch: np.ndarray) -> np.ndarray:
        n = flat_batch.shape[0]
        X_flat = flat_batch[:, : 3 * n_features].astype(np.float32)
        X = X_flat.reshape(n, 3, n_features)
        era_col = flat_batch[:, 3 * n_features].astype(np.float32)
        static = np.stack([np.zeros(n, dtype=np.float32), era_col], axis=1)
        lengths = np.full(n, 3, dtype=np.int64)
        with torch.no_grad():
            pred = model(
                torch.as_tensor(X, dtype=torch.float32),
                torch.as_tensor(lengths, dtype=torch.int64),
                torch.as_tensor(static, dtype=torch.float32),
            ).numpy()
        return pred

    rng = np.random.default_rng(RANDOM_SEED)
    background_size = min(SHAP_BACKGROUND_SIZE, len(train_pool_idx))
    background_idx = rng.choice(train_pool_idx, size=background_size, replace=False)
    background_flat = flatten_features_for_shap(
        X_scaled[background_idx], batch.era[background_idx]
    )

    # Val predictions/residuals on ALL non-sprint val rows, to pick the
    # low/high error buckets - reconstructed to absolute time here, since
    # error must be measured in real seconds (the SHAP explanation itself
    # below does NOT need this reconstruction - see module docstring).
    val_flat = flatten_features_for_shap(X_scaled[val_pool_idx], batch.era[val_pool_idx])
    val_pred_gap = predict_fn(val_flat)
    val_pred_abs = val_pred_gap + batch.practice_reference[val_pool_idx]
    val_abs_residuals = np.abs(val_pred_abs - batch.y_abs[val_pool_idx])

    low_rel_idx, high_rel_idx = select_error_buckets(val_abs_residuals, SHAP_BUCKET_SIZE)
    print(
        f"Explaining {len(low_rel_idx)} lowest-error + {len(high_rel_idx)} "
        f"highest-error non-sprint val rows "
        f"(mean |residual|: low={val_abs_residuals[low_rel_idx].mean():.3f}s, "
        f"high={val_abs_residuals[high_rel_idx].mean():.3f}s) with "
        f"shap.KernelExplainer (background={background_size}, "
        f"nsamples={SHAP_NSAMPLES})...",
        flush=True,
    )

    explain_flat = val_flat[np.concatenate([low_rel_idx, high_rel_idx])]

    start = time.monotonic()
    explainer = shap.KernelExplainer(predict_fn, background_flat)
    shap_values = explainer.shap_values(explain_flat, nsamples=SHAP_NSAMPLES)
    if isinstance(shap_values, list):  # defensive - some shap versions wrap in a list
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    elapsed = time.monotonic() - start
    print(f"  Done in {elapsed:.1f}s.", flush=True)

    overall = aggregate_shap_by_feature(shap_values, feature_cols)
    print("\nTop 8 features by mean |SHAP| (all 40 explained rows pooled):", flush=True)
    for _, row in overall.head(8).iterrows():
        print(f"  {row['feature']:<28} mean|SHAP|={row['mean_abs_shap']:.4f}", flush=True)

    n_low = len(low_rel_idx)
    low_shap = aggregate_shap_by_feature(shap_values[:n_low], feature_cols)
    high_shap = aggregate_shap_by_feature(shap_values[n_low:], feature_cols)
    comparison = low_shap.merge(
        high_shap, on="feature", suffixes=("_low_error", "_high_error")
    )
    comparison["abs_shap_diff_high_minus_low"] = (
        comparison["mean_abs_shap_high_error"] - comparison["mean_abs_shap_low_error"]
    )
    comparison = comparison.sort_values(
        "abs_shap_diff_high_minus_low", ascending=False
    ).reset_index(drop=True)

    print(
        "\nTop 5 features whose attribution grows MOST in high-error rows "
        "vs low-error rows (candidates for 'what's driving the mistakes'):",
        flush=True,
    )
    for _, row in comparison.head(5).iterrows():
        print(
            f"  {row['feature']:<28} low={row['mean_abs_shap_low_error']:.4f}  "
            f"high={row['mean_abs_shap_high_error']:.4f}  "
            f"diff={row['abs_shap_diff_high_minus_low']:+.4f}",
            flush=True,
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(MODELS_DIR / "shap_importance_by_feature.csv", index=False)
    comparison.to_csv(MODELS_DIR / "shap_error_analysis.csv", index=False)
    print(
        f"\nSaved shap_importance_by_feature.csv and shap_error_analysis.csv "
        f"to {MODELS_DIR}",
        flush=True,
    )


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

    _analyze_learning_curve()
    _run_shap_error_analysis(wide_df, feature_cols, fcols)


if __name__ == "__main__":
    main()
