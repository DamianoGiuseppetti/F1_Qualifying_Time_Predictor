"""Phase 4/5 groundwork: retrain the production LSTM + recalibrate
confidence intervals after a new round's real results have been added to
the dataset.

Not wired to any automatic trigger yet (Damiano's decision, Aug 24 2026:
"build the logic now, wire automation later") - this is the script a
human runs by hand today. Phase 4's API/Docker service is free to call
the same underlying f1qp.modeling functions from wherever it eventually
decides to hook in (a background job, a scheduled task, an endpoint that
fires once a round's results are confirmed) - nothing here assumes a
particular trigger mechanism.

**What this script does NOT do**: fetch or process new race data itself.
That's Phase 1/2's job (FastF1 pulls, feature engineering,
scripts/prepare_phase3_dataset.py). This script assumes
data/processed/phase3_dataset.parquet already reflects however many
completed rounds exist when it's run. Its job starts from there.

**What it does, every time it runs** (supersedes running
scripts/loro_2026_check_lstm.py, scripts/train_final_lstm.py, and
scripts/calibrate_conformal.py separately by hand - all three are folded
into one pipeline here, in the right order, sharing one dataset load):

1. A FRESH `leave_one_round_out_cv_lstm` check on era 1 (now covering
   however many 2026 rounds exist) - both to report the current honest
   generalization estimate, and to derive `train_final_model`'s fixed
   epoch count via `f1qp.modeling.retrain.select_final_epoch_count`
   (the max best_epoch observed across folds) rather than reusing a
   stale constant from an earlier, smaller round count.
2. Retrain the production model on EVERY non-holdout row now available
   (grows by however many driver-weekends the new round added).
3. Recalibrate split-conformal intervals (all 4 levels: 50/68/80/90%)
   against the now-larger era-1 LORO residual pool, plus a fresh era-0
   reference re-fit (same recipe as scripts/calibrate_conformal.py).

**Era policy** (Aug 24 2026 decision, revisit later): always blend era 0
and era 1 data - no era-1-only training attempted yet, era 0 still
provides far more volume than era 1 does today. See
f1qp.modeling.retrain's module docstring for the full reasoning; nothing
in this script changes that policy on its own.

**Staged, not live (Sep 16 2026 addition - Damiano's "auto-stage,
one-click to promote" design)**: this script no longer touches the live
`models/lstm/*` artifacts at all. Every run writes its 4 output files
(model weights, preprocessing stats, metadata, conformal intervals) plus
a `staged_meta.json` (the before/after comparison + MLflow run id) into a
fresh `models/lstm/staged/<UTC-timestamp>/` directory, and updates
`models/lstm/staged/latest.json` to point at it - the actual production
artifacts, and whatever this API process currently has loaded in memory,
are completely untouched until someone explicitly promotes this
candidate (see f1qp.modeling.promote's module docstring, and
f1qp.api.main's `POST /model/promote` / the Performance tab's "Promote to
production" button). This is what lets
f1qp.serving.data_fetch.start_results_job chain straight into this script
automatically after every "Check for official result" fetch without ever
silently swapping in an unreviewed model. A before/after comparison
(f1qp.modeling.retrain.format_retrain_comparison) is printed every run -
this is the actual evidence for whether retraining is helping as more
real 2026 data accumulates, which is the whole point of running this
repeatedly rather than once. The very first run has nothing to compare
against yet - that's reported plainly, not as an error.

**MLflow (Phase 5 addition, Aug 25 2026)**: every run also logs its
params/metrics and registers the retrained model to a local MLflow
tracking store (./mlruns, see f1qp.modeling.tracking) - the "Model
versioning in MLflow (v1.0)" success criterion. This is IN ADDITION to the
history/ archive above, not a replacement for it - best-effort, never
blocks a retrain if mlflow isn't installed. Read back via
`mlflow.search_runs(...)` or the Streamlit dashboard's "Model performance
history" tab (dashboard/app.py).

**Holdout handling (Phase 5 addition)**: reads
`phase3_feature_columns.json`'s `holdout_included` flag (written by
scripts/prepare_phase3_dataset.py) purely to report which mode built the
loaded dataset - see scripts/evaluate_holdout.py's module docstring for
why Round 12 must stay excluded until that offline test has run, and only
then get folded in (`--include-holdout`) before the final pre-Round-13
run of this pipeline.

Run from anywhere - paths are resolved relative to this file:

    python scripts/retrain_pipeline.py

Expect roughly the sum of the three scripts this replaces (~9s LORO +
<1s final retrain + ~10s conformal, scaling up slowly as more rounds are
added to the LORO fold count) - under a minute total on a laptop CPU for
the data volumes seen so far. Prints live (flush=True) throughout, same
convention as every other script in this project.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from f1qp.modeling.conformal import (
    k_fold_conformal_check,
    leave_one_round_out_conformal_check,
)
from f1qp.modeling.dataset import pivot_to_weekend_features, resolve_feature_columns
from f1qp.modeling.lstm_model import (
    leave_one_round_out_cv_lstm,
    train_final_model,
    train_lstm,
)
from f1qp.modeling.retrain import format_retrain_comparison, select_final_epoch_count
from f1qp.modeling.sequences import FeatureImputer, FeatureScaler, build_lstm_sequences
from f1qp.modeling.tracking import REGISTERED_MODEL_NAME, log_retrain_run

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "lstm"
# Sep 16 2026: the models/lstm/history/<timestamp>/ archive of SUPERSEDED
# production artifacts is now written by f1qp.modeling.promote at
# promotion time, not by this script - see this module's own "Staged, not
# live" docstring section. STAGED_DIR (defined further down, next to
# _write_staged_pointer) is where THIS script's own output goes.

K_FOLD_K = 5
ALPHA_LEVELS = [0.50, 0.32, 0.20, 0.10]  # 50/68/80/90% - same as scripts/calibrate_conformal.py


def _json_safe(obj):
    """Same numpy-scalar/array/dict-key sanitizer as
    scripts/calibrate_conformal.py - see that script's comment for why
    json.dump's `default=` hook alone can't fix numpy dict keys."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    return obj


def _load_previous_summary():
    metadata_path = MODELS_DIR / "final_model_metadata.json"
    conformal_path = MODELS_DIR / "conformal_intervals.json"
    if not metadata_path.exists() or not conformal_path.exists():
        return None
    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(conformal_path) as f:
        conformal = json.load(f)
    interval_50 = (
        conformal.get("alphas", {})
        .get(str(0.50), {})
        .get("era_1", {})
        .get("deployment_quantile_seconds")
    )
    return {
        "n_train": metadata.get("n_train"),
        "pooled_mape": metadata.get("reference_leave_one_round_out_pooled_mape"),
        "pooled_r2": metadata.get("reference_leave_one_round_out_pooled_r2"),
        "epoch_count": metadata.get("n_epochs"),
        "interval_50pct": interval_50,
    }


STAGED_DIR = MODELS_DIR / "staged"


def _write_staged_pointer(stamp: str) -> None:
    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    with open(STAGED_DIR / "latest.json", "w") as f:
        json.dump({"timestamp": stamp}, f, indent=2)


def _era0_val_abs_residuals(wide_df, feature_cols, fcols) -> np.ndarray:
    """Same recipe as scripts/calibrate_conformal.py's own helper of the
    same name - duplicated rather than imported so this pipeline script
    has no cross-script dependency; both are short enough that the
    duplication is cheaper than the coupling would be."""
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)
    train_idx = batch.split == "train"
    val_idx = batch.split == "val"

    imputer = FeatureImputer.fit(batch.X[train_idx], batch.mask[train_idx])
    X_imputed = imputer.transform(batch.X, batch.mask)
    scaler = FeatureScaler.fit(X_imputed[train_idx], batch.mask[train_idx])
    X_scaled = scaler.transform(X_imputed, batch.mask)

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
    with torch.no_grad():
        val_pred_gap = model(
            torch.as_tensor(X_scaled[val_idx], dtype=torch.float32),
            torch.as_tensor(batch.lengths[val_idx], dtype=torch.int64),
            torch.as_tensor(batch.static[val_idx], dtype=torch.float32),
        ).numpy()
    val_pred_abs = val_pred_gap + batch.practice_reference[val_idx]
    val_y_abs = batch.y_abs[val_idx]
    val_era = batch.era[val_idx]
    era0_mask = val_era == 0
    return np.abs(val_pred_abs[era0_mask] - val_y_abs[era0_mask])


def main() -> None:
    previous_summary = _load_previous_summary()
    if previous_summary is None:
        print(
            "No previous production artifacts found - this looks like the "
            "first-ever run of this pipeline.",
            flush=True,
        )

    dataset_path = DATA_DIR / "phase3_dataset.parquet"
    feature_meta_path = DATA_DIR / "phase3_feature_columns.json"

    print(f"Loading dataset from {dataset_path}", flush=True)
    merged = pd.read_parquet(dataset_path)
    with open(feature_meta_path) as f:
        feature_meta = json.load(f)
    feature_cols = feature_meta["feature_cols"]

    # Phase 5 addition (Aug 25 2026): phase3_feature_columns.json now records
    # whether prepare_phase3_dataset.py was run with --include-holdout. Round
    # 12 (Zandvoort) either has no "holdout" rows to exclude below (if it
    # was folded into train/val - the mode used for the final pre-Round-13
    # retrain, after scripts/evaluate_holdout.py's offline test already ran)
    # or is still excluded (if it wasn't - e.g. a stale dataset). Recorded
    # into this run's own metadata/MLflow tags below so it's never ambiguous
    # after the fact which mode produced a given production model.
    holdout_included = bool(feature_meta.get("holdout_included", False))
    print(
        f"Dataset built with --include-holdout={holdout_included} "
        + (
            "- Round 12 IS part of this run's training pool."
            if holdout_included
            else "- Round 12 is EXCLUDED from this run's training pool (offline-test mode)."
        ),
        flush=True,
    )

    fcols = resolve_feature_columns(merged)
    print("Pivoting to weekend-level...", flush=True)
    wide_df = pivot_to_weekend_features(merged, feature_cols, fcols)

    n_before = len(wide_df)
    wide_df = wide_df[wide_df["split"] != "holdout"].reset_index(drop=True)
    print(f"Excluded {n_before - len(wide_df)} holdout row(s) (Zandvoort)", flush=True)

    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    print(f"Training pool: {len(wide_df)} usable driver-weekends", flush=True)

    print("\n--- Step 1/3: fresh leave-one-round-out check (era 1) ---", flush=True)
    loro_result = leave_one_round_out_cv_lstm(wide_df, feature_cols, fcols, era_value=1)
    per_round_best_epochs = [m["best_epoch"] for m in loro_result["per_round"].values()]
    final_epoch_count = select_final_epoch_count(per_round_best_epochs)
    print(
        f"\nDerived final epoch count from {len(per_round_best_epochs)} folds: "
        f"{final_epoch_count} (max of {sorted(per_round_best_epochs)})",
        flush=True,
    )

    print("\n--- Step 2/3: retrain production model on ALL non-holdout data ---", flush=True)
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)
    imputer = FeatureImputer.fit(batch.X, batch.mask)
    X_imputed = imputer.transform(batch.X, batch.mask)
    scaler = FeatureScaler.fit(X_imputed, batch.mask)
    X_scaled = scaler.transform(X_imputed, batch.mask)

    final_result = train_final_model(
        train_X=X_scaled,
        train_lengths=batch.lengths,
        train_static=batch.static,
        train_y_gap=batch.y_gap,
        n_features=len(feature_cols),
        n_epochs=final_epoch_count,
    )

    print("\n--- Step 3/3: recalibrate split-conformal intervals ---", flush=True)
    residuals_by_round = loro_result["residuals_by_round"]
    era0_abs_residuals = _era0_val_abs_residuals(wide_df, feature_cols, fcols)

    conformal_results = {"computed_at_utc": datetime.now(timezone.utc).isoformat(), "alphas": {}}
    interval_50 = None
    for alpha in ALPHA_LEVELS:
        coverage_pct = int(round((1 - alpha) * 100))
        era1 = leave_one_round_out_conformal_check(residuals_by_round, alpha=alpha)
        era0 = k_fold_conformal_check(era0_abs_residuals, alpha=alpha, k=K_FOLD_K)
        if coverage_pct == 50:
            interval_50 = era1.final_quantile.quantile
        conformal_results["alphas"][str(alpha)] = {
            "coverage_target_pct": coverage_pct,
            "era_1": {
                "pooled_coverage": era1.pooled_coverage,
                "deployment_quantile_seconds": era1.final_quantile.quantile,
                "deployment_quantile_exact": era1.final_quantile.exact,
                "n_calibration": era1.n_test_total,
                "per_round": era1.per_round,
            },
            "era_0": {
                "pooled_coverage": era0.pooled_coverage,
                "reference_quantile_seconds": era0.final_quantile.quantile,
                "reference_quantile_exact": era0.final_quantile.exact,
                "n_calibration": len(era0_abs_residuals),
                "per_fold": era0.per_fold,
            },
        }
        print(
            f"  {coverage_pct}%: era1 +/-{era1.final_quantile.quantile:.3f}s  "
            f"(empirical coverage {era1.pooled_coverage * 100:.1f}%)",
            flush=True,
        )

    current_summary = {
        "n_train": len(wide_df),
        "pooled_mape": loro_result["pooled"]["mape"],
        "pooled_r2": loro_result["pooled"]["r2"],
        "epoch_count": final_epoch_count,
        "interval_50pct": interval_50,
    }
    print("\n" + format_retrain_comparison(previous_summary, current_summary), flush=True)

    # Sep 16 2026: this run's output is a STAGED candidate, not a live
    # swap - see module docstring's "Staged, not live" section. Nothing
    # under the live MODELS_DIR (lstm_final_production.pt etc.) is touched
    # by this script anymore; everything below writes into its own
    # timestamped directory under STAGED_DIR instead.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_dir = STAGED_DIR / stamp
    stage_dir.mkdir(parents=True, exist_ok=True)

    torch.save(final_result.model.state_dict(), stage_dir / "lstm_final_production.pt")
    np.savez(
        stage_dir / "final_preprocessing.npz",
        imputer_median=imputer.median,
        scaler_mean=scaler.mean,
        scaler_std=scaler.std,
        feature_cols=np.array(feature_cols),
        static_cols=np.array(batch.static_cols),
    )
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": len(wide_df),
        "n_epochs": final_epoch_count,
        "final_train_loss": final_result.train_loss_history[-1],
        "total_seconds": final_result.total_seconds,
        "target": "gap_final (reconstructed to final_quali_time via + practice_reference)",
        "reference_leave_one_round_out_pooled_mape": loro_result["pooled"]["mape"],
        "reference_leave_one_round_out_pooled_r2": loro_result["pooled"]["r2"],
        # Phase 5 addition: lets scripts/evaluate_holdout.py refuse to run
        # (loudly, not silently) if this production model already trained on
        # Round 12 - see that script's module docstring.
        "trained_with_holdout": holdout_included,
        "note": (
            "No fresh held-out metric exists for this specific model by design "
            "(trained on all non-holdout data, no val split). Trust the "
            "leave-one-round-out pooled numbers above - freshly recomputed by "
            "THIS pipeline run against the current dataset, not carried over "
            "from an earlier run."
        ),
    }
    metadata_path = stage_dir / "final_model_metadata.json"
    conformal_path = stage_dir / "conformal_intervals.json"
    with open(metadata_path, "w") as f:
        json.dump(_json_safe(metadata), f, indent=2)
    with open(conformal_path, "w") as f:
        json.dump(_json_safe(conformal_results), f, indent=2)

    # Phase 5 addition (Aug 25 2026): log this run to a local MLflow tracking
    # store + register the retrained model - the "Model versioning in MLflow
    # (v1.0)" success criterion. Best-effort: never blocks a retrain if
    # mlflow isn't installed or logging fails (see f1qp.modeling.tracking's
    # module docstring) - the models/lstm/history/<timestamp>/ archive that
    # f1qp.modeling.promote writes at PROMOTION time already covers
    # versioning even if this step is skipped.
    interval_metrics = {
        f"interval_{int(round((1 - a) * 100))}pct": conformal_results["alphas"][str(a)]["era_1"][
            "deployment_quantile_seconds"
        ]
        for a in ALPHA_LEVELS
    }
    run_id = log_retrain_run(
        params={
            "n_train": len(wide_df),
            "epoch_count": final_epoch_count,
            "holdout_included": holdout_included,
            "k_fold_k": K_FOLD_K,
        },
        metrics={
            "pooled_mape": loro_result["pooled"]["mape"],
            "pooled_r2": loro_result["pooled"]["r2"],
            **interval_metrics,
        },
        artifact_paths=[metadata_path, conformal_path],
        model=final_result.model,
    )
    if run_id:
        print(f"Logged MLflow run {run_id} (registered model '{REGISTERED_MODEL_NAME}')", flush=True)
        # Patch the staged metadata with the run id now that it's known
        # (log_retrain_run needed the file to already exist on disk to
        # attach it as an artifact) - this is what lets `/model/info`
        # show which MLflow run is actually live once this candidate is
        # promoted (f1qp.modeling.promote copies this file verbatim).
        metadata["mlflow_run_id"] = run_id
        with open(metadata_path, "w") as f:
            json.dump(_json_safe(metadata), f, indent=2)

    staged_meta = {
        "trained_at_utc": metadata["trained_at_utc"],
        "holdout_included": holdout_included,
        "mlflow_run_id": run_id,
        "previous_summary": previous_summary,
        "current_summary": current_summary,
    }
    with open(stage_dir / "staged_meta.json", "w") as f:
        json.dump(_json_safe(staged_meta), f, indent=2)
    _write_staged_pointer(stamp)

    print(
        f"\nSTAGED (not live) at {stage_dir} - production keeps serving the "
        f"previous model until explicitly promoted (POST /model/promote, or "
        f"the Performance tab's \"Promote to production\" button).",
        flush=True,
    )


if __name__ == "__main__":
    main()
