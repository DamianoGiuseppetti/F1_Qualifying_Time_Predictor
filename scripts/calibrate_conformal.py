"""Phase 3, LSTM step: confidence intervals via split-conformal calibration.

See f1qp.modeling.conformal's module docstring for the full method and the
reasoning behind every design choice summarized here:

  - Era 1 (2026): calibration residuals come from `leave_one_round_out_cv_lstm`
    (re-run here rather than loaded from a prior run's output, so this
    script is self-contained) - out-of-fold by construction, and the era
    that actually matters, since the production model only ever predicts
    2026 rounds.
  - Era 0 (2023-2025): calibration residuals come from re-fitting the exact
    same standalone 80/20 LSTM as scripts/train_lstm.py (same seed, split,
    and defaults - re-run rather than loaded from the saved checkpoint so
    this script never depends on a possibly-stale models/lstm/ file),
    restricted to that run's era-0 val rows. Diagnostic/documentation only
    - satisfies Task_List's "era-stratified" requirement, never used for a
    real Round 13 decision.
  - Coverage is checked WITHOUT circularity: era 1 via leave-one-round-out
    (calibrate on the other 10 rounds, check the 11th), era 0 via a 5-fold
    split of its val residuals - see conformal.py for why checking coverage
    on the same pool used to fit the quantile would be close to
    tautological.

**Coverage/width tradeoff sweep (Aug 24 2026, added after the first real
run):** the first real run computed only 80%/90% intervals and both came
out over 1.8s at 90% - a real, mathematically consistent result (the
pooled LORO MAPE is ~1.05%, i.e. ~0.95s mean absolute error on a ~90s lap;
a 90th-percentile interval on a right-skewed error distribution with real
disruption-weekend outliers in the tail (Spa 2026, etc.) sits well above
that mean, not at it), but wider than wanted. Getting a materially
narrower number is a question of WHICH coverage level to report, not a
calibration bug to fix - so this script now sweeps FOUR coverage targets
(50%/68%/80%/90%, i.e. alpha in [0.50, 0.32, 0.20, 0.10] - 68% chosen as
the familiar "one-sigma-equivalent" reference point) and prints a compact
tradeoff table at the end, so the actual width-vs-confidence curve for
this model's real residuals is visible at a glance rather than guessed at.
Picking, say, the 50% or 68% row over the 90% one is a legitimate choice
IF what's being communicated changes to match ("typical case" instead of
"we're 90% sure") - reporting a smaller number under the 90% label without
actually recomputing it at a lower alpha would silently break the coverage
guarantee, which is exactly what this step's own coverage check exists to
catch.

Run from anywhere - paths are resolved relative to this file:

    python scripts/calibrate_conformal.py

Prints as it goes (flush=True), same convention as every other script in
this project. Expect well under a minute total - this reruns the already-
confirmed-fast LORO check (8.9s on real data) plus one more ~1s standalone
LSTM fit, then reuses both fits' residuals across all four alpha levels
(cheap - only the quantile/coverage arithmetic repeats, not the training).
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
from f1qp.modeling.lstm_model import leave_one_round_out_cv_lstm, train_lstm
from f1qp.modeling.sequences import FeatureImputer, FeatureScaler, build_lstm_sequences

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "lstm"

ALPHA_LEVELS = [0.50, 0.32, 0.20, 0.10]  # 50% / 68% / 80% / 90% coverage targets
K_FOLD_K = 5  # era-0 coverage check fold count - no Task_List-specified value, standard default


def _json_safe(obj):
    """Recursively convert numpy scalar/array types (and dict keys, which
    `json.dump`'s own `default=` hook can't fix - it's only ever called on
    non-serializable VALUES, never keys) into plain Python types. Needed
    because round numbers flow through this pipeline as numpy int64 (from
    a pandas column) and end up as dict keys in `per_round`/`per_fold`.
    """
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


def _era0_val_abs_residuals(wide_df, feature_cols, fcols) -> np.ndarray:
    """Re-fit the standalone 80/20 LSTM (same seed/split/defaults as
    scripts/train_lstm.py) and return its val-split residuals' absolute
    value, restricted to era 0 (2023-2025). See module docstring for why
    era 1 does NOT use this source.
    """
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)
    train_idx = batch.split == "train"
    val_idx = batch.split == "val"

    imputer = FeatureImputer.fit(batch.X[train_idx], batch.mask[train_idx])
    X_imputed = imputer.transform(batch.X, batch.mask)
    scaler = FeatureScaler.fit(X_imputed[train_idx], batch.mask[train_idx])
    X_scaled = scaler.transform(X_imputed, batch.mask)

    print(
        "Re-fitting the standalone 80/20 LSTM (same seed/split/defaults as "
        "scripts/train_lstm.py) to get held-out val residuals for era 0...",
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
    residuals = val_pred_abs[era0_mask] - val_y_abs[era0_mask]
    print(
        f"  best_epoch={result.best_epoch}  n_val_total={int(val_idx.sum())}  "
        f"n_val_era0={int(era0_mask.sum())}",
        flush=True,
    )
    return np.abs(residuals)


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

    print(
        "\n--- Era 1 (2026): leave-one-round-out out-of-fold residuals ---",
        flush=True,
    )
    loro_result = leave_one_round_out_cv_lstm(wide_df, feature_cols, fcols, era_value=1)
    residuals_by_round = loro_result["residuals_by_round"]

    print(
        "\n--- Era 0 (2023-2025): held-out val-split residuals (diagnostic only) ---",
        flush=True,
    )
    era0_abs_residuals = _era0_val_abs_residuals(wide_df, feature_cols, fcols)

    results = {"computed_at_utc": datetime.now(timezone.utc).isoformat(), "alphas": {}}
    tradeoff_rows = []

    for alpha in ALPHA_LEVELS:
        coverage_pct = int(round((1 - alpha) * 100))
        print(f"\n=== {coverage_pct}% interval (alpha={alpha}) ===", flush=True)

        era1 = leave_one_round_out_conformal_check(residuals_by_round, alpha=alpha)
        print(
            f"Era 1 (2026): leave-one-round-out coverage check across "
            f"{len(era1.per_round)} rounds -> pooled empirical coverage "
            f"{era1.pooled_coverage * 100:.1f}% (target {coverage_pct}%)",
            flush=True,
        )
        for round_number, metrics in sorted(era1.per_round.items()):
            warn = (
                "" if metrics["exact"]
                else "  (WARNING: too few calibration residuals for an exact guarantee)"
            )
            print(
                f"  round={round_number:>3}  q=+/-{metrics['quantile']:.3f}s  "
                f"n_test={metrics['n_test']:2d}  coverage={metrics['coverage'] * 100:5.1f}%{warn}",
                flush=True,
            )
        warn = "" if era1.final_quantile.exact else "  (WARNING: not exact)"
        print(
            f"  -> Deployment quantile for Round 13 (fit on all "
            f"{era1.n_test_total} era-1 residuals): "
            f"+/-{era1.final_quantile.quantile:.3f}s{warn}",
            flush=True,
        )

        era0 = k_fold_conformal_check(era0_abs_residuals, alpha=alpha, k=K_FOLD_K)
        print(
            f"Era 0 (2023-2025): {era0.k}-fold coverage check -> pooled "
            f"empirical coverage {era0.pooled_coverage * 100:.1f}% "
            f"(target {coverage_pct}%)",
            flush=True,
        )
        warn = "" if era0.final_quantile.exact else "  (WARNING: not exact)"
        print(
            f"  -> Reference quantile (fit on all {len(era0_abs_residuals)} "
            f"era-0 val residuals): +/-{era0.final_quantile.quantile:.3f}s{warn}",
            flush=True,
        )

        results["alphas"][str(alpha)] = {
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
        tradeoff_rows.append({
            "coverage_pct": coverage_pct,
            "era1_quantile": era1.final_quantile.quantile,
            "era1_pooled_coverage": era1.pooled_coverage,
            "era0_quantile": era0.final_quantile.quantile,
            "era0_pooled_coverage": era0.pooled_coverage,
        })

    print("\n=== Coverage / width tradeoff (era 1 = the number that matters for "
          "Round 13) ===", flush=True)
    print(f"{'target':>8}  {'era1 +/-s':>10}  {'era1 empirical':>15}  "
          f"{'era0 +/-s':>10}  {'era0 empirical':>15}", flush=True)
    for row in sorted(tradeoff_rows, key=lambda r: r["coverage_pct"]):
        print(
            f"{row['coverage_pct']:>7}%  {row['era1_quantile']:>10.3f}  "
            f"{row['era1_pooled_coverage'] * 100:>14.1f}%  "
            f"{row['era0_quantile']:>10.3f}  {row['era0_pooled_coverage'] * 100:>14.1f}%",
            flush=True,
        )
    print(
        "\nPick the row whose coverage level matches what you actually want to "
        "communicate - a narrower row is a legitimate choice IF the claim "
        "attached to it changes too (e.g. \"typical case\" for the 50%/68% row "
        "instead of \"90% confidence\"). Relabeling a narrower row's number as "
        "if it were the 90% figure would silently break the coverage guarantee.",
        flush=True,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "conformal_intervals.json"
    with open(out_path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    print(f"\nSaved conformal calibration results to {out_path}", flush=True)
    print(
        "\nUse era 1's deployment_quantile_seconds as the +/- half-width "
        "around the final production model's point prediction for Round 13 "
        "(Monza) - era 0's numbers are reference/documentation only, since "
        "the model is never actually deployed against a 2023-2025 weekend.",
        flush=True,
    )


if __name__ == "__main__":
    main()
