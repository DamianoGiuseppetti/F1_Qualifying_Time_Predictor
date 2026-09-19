"""Phase 5: pure scoring logic for scripts/evaluate_holdout.py's Olanda
(Zandvoort, Round 12) offline test.

Kept separate from the script's I/O (loading production artifacts, reading
phase3_dataset.parquet, calling the FastAPI/serving prediction path) so the
actual scoring logic - joining predictions onto real results, computing
MAPE/R^2, checking whether the real time fell inside the shipped conformal
interval - is unit-testable on synthetic data, same pattern as
f1qp.modeling.retrain/conformal/interpretability.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from f1qp.modeling.baseline import mape, r_squared


def build_scored_dataframe(predictions_df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
    """Join per-driver predictions onto the real qualifying result and
    compute per-driver error + interval coverage.

    `predictions_df` needs a `Driver` column plus
    `predicted_quali_time_seconds`/`interval_low_seconds`/
    `interval_high_seconds` (the shape `f1qp.serving.predict.DriverPrediction`
    produces once its `driver` field is renamed to `Driver` - see
    scripts/evaluate_holdout.py). `targets_df` needs `Driver`/
    `final_quali_time`/`has_target` (phase3_dataset.parquet's holdout rows,
    deduplicated to one row per driver - a driver's `final_quali_time` is
    identical across every practice-session row for that weekend, see
    f1qp.modeling.dataset.assemble_dataset).

    Drivers with no real target (DNS/DSQ/no match, `has_target=False`) are
    excluded from the returned frame entirely - there is nothing to score
    them against. A driver in `targets_df` but missing from
    `predictions_df` (shouldn't happen - `predict_event` predicts every
    driver with practice data) is likewise excluded rather than crashing.
    """
    merged = predictions_df.merge(targets_df, on="Driver", how="left")
    merged["has_target"] = merged["has_target"].fillna(False).astype(bool)
    scored = merged[merged["has_target"]].copy()
    if scored.empty:
        return scored

    scored["abs_error_seconds"] = (
        scored["predicted_quali_time_seconds"] - scored["final_quali_time"]
    ).abs()
    scored["within_interval"] = scored["final_quali_time"].between(
        scored["interval_low_seconds"], scored["interval_high_seconds"]
    )
    return scored


def summarize(scored: pd.DataFrame, n_missing_target: int, interval_level_pct: float) -> Dict:
    """Aggregate `build_scored_dataframe`'s output into the summary dict
    scripts/evaluate_holdout.py saves to holdout_evaluation.json.

    Raises `ValueError` (not a silent NaN-filled summary) if `scored` is
    empty - a genuinely uninformative offline test (every driver missing a
    target) should be loud, not produce a summary that looks normal.
    """
    if scored.empty:
        raise ValueError(
            "No driver has a real qualifying target to score against - "
            "nothing to summarize. Check that extract_qualifying_targets.py "
            "has been run for this round."
        )
    return {
        "n_drivers_scored": int(len(scored)),
        "n_drivers_missing_target": int(n_missing_target),
        "mape_pct": mape(scored["final_quali_time"], scored["predicted_quali_time_seconds"]),
        "r2": r_squared(scored["final_quali_time"], scored["predicted_quali_time_seconds"]),
        "interval_level_pct": float(interval_level_pct),
        "interval_empirical_coverage_pct": float(scored["within_interval"].mean() * 100),
    }


def metadata_trained_with_holdout(metadata: Dict) -> bool:
    """Whether `final_model_metadata.json` shows the CURRENT production
    model already trained on the holdout round - if so, evaluating it
    against that round measures in-sample fit, not a real offline test.
    Missing key (older metadata written before this Phase 5 addition)
    defaults to False - the historical/expected state, not a false alarm.
    """
    return bool(metadata.get("trained_with_holdout", False))
