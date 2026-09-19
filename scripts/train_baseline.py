"""Phase 3, step 4: XGBoost baseline before the LSTM.

Loads the dataset built by scripts/prepare_phase3_dataset.py, pivots it to
one row per driver-weekend, trains both target formulations
(absolute / gap-then-reconstruct) against `final_quali_time` - the time
that actually decided each driver's grid position (Q3 if reached,
otherwise Q2, otherwise Q1; see f1qp.modeling.dataset docstring) - reports
validation MAPE and R2 (aggregate and era-stratified, always on
reconstructed absolute time), picks one winning formulation, and saves the
winning model plus its gain-based and permutation feature importances.

Run from anywhere - paths are resolved relative to this file:

    python scripts/train_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from f1qp.modeling.baseline import run_baseline_comparison
from f1qp.modeling.dataset import pivot_to_weekend_features, resolve_feature_columns

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "baseline"


def main() -> None:
    dataset_path = DATA_DIR / "phase3_dataset.parquet"
    feature_meta_path = DATA_DIR / "phase3_feature_columns.json"

    merged = pd.read_parquet(dataset_path)
    with open(feature_meta_path) as f:
        feature_meta = json.load(f)
    feature_cols = feature_meta["feature_cols"]

    fcols = resolve_feature_columns(merged)
    wide_df = pivot_to_weekend_features(merged, feature_cols, fcols)
    print(f"Pivoted to weekend-level: {wide_df.shape}")
    print(f"Driver-weekends with a final_quali_time: {wide_df['has_target'].sum()} / {len(wide_df)}")

    # Never touch the holdout weekend (Zandvoort) here - Phase 5's job.
    n_before = len(wide_df)
    wide_df = wide_df[wide_df["split"] != "holdout"].reset_index(drop=True)
    print(f"Excluded {n_before - len(wide_df)} holdout row(s) (Zandvoort)")

    comparison = run_baseline_comparison(wide_df, feature_cols, fcols)

    print("\nValidation MAPE by formulation (reconstructed absolute time):")
    for formulation, value in comparison["aggregate_val_mape"].items():
        print(f"  {formulation}: {value:.3f}%")

    winner = comparison["winning_formulation"]
    res = comparison["results"][winner]
    print(f"\nWinning formulation: {winner}")
    print(f"  val MAPE={res.val_mape:.3f}%  R2={res.val_r2:.3f}  "
          f"n_train={res.n_train}  n_val={res.n_val}")
    for era_value in sorted(res.val_mape_by_era):
        print(f"      era={era_value}: MAPE={res.val_mape_by_era[era_value]:.3f}%  "
              f"R2={res.val_r2_by_era[era_value]:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    res.model.save_model(str(MODELS_DIR / f"xgb_{winner}_final_quali_time.json"))
    res.gain_importance.to_csv(MODELS_DIR / "gain_importance.csv")
    res.permutation_importance.to_csv(MODELS_DIR / "permutation_importance.csv")

    print(f"\nSaved model and feature importances to {MODELS_DIR}")
    print("\nTop-10 permutation importance (candidate final feature set for the LSTM):")
    print(res.permutation_importance.head(10).to_string())


if __name__ == "__main__":
    main()
