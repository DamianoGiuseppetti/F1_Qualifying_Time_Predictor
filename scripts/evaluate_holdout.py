"""Phase 5: Olanda (Zandvoort, Round 12) offline test - the project brief's
"Test A: Olanda Round 12 (offline validation)".

This is the ONE deliberately held-out 2026 weekend
(f1qp.modeling.dataset.HOLDOUT_YEAR/HOLDOUT_ROUND = 2026/12, same round as
f1qp.config.OFFLINE_TEST_ROUND). It checks, per driver: the CURRENT
production model's predicted final_quali_time (via
f1qp.serving.predict - unchanged from Phase 4, no special "offline test"
code path) against the REAL Q1/Q2/Q3 result now available for Round 12,
plus whether the real time actually falls inside the shipped 50% "typical
range" interval - the first real-world check of that interval's honesty
outside the leave-one-round-out cross-validation it was calibrated from.

**Sequencing - read before running** (see also
scripts/prepare_phase3_dataset.py's own docstring):

1. scripts/download_2026.py - confirm Round 12 raw laps are on disk
   (already done per HANDOVER.md, Aug 25 2026).
2. scripts/extract_weather.py, extract_qualifying_targets.py,
   extract_telemetry.py, build_features.py - these already include Round
   12 unconditionally (f1qp.config.OFFLINE_TEST_ROUND is baked into each
   script's `_events()`), no flag needed.
3. scripts/prepare_phase3_dataset.py WITHOUT --include-holdout (the
   default) - keeps Round 12 labeled "holdout", so whatever production
   model exists right now is one that has NEVER trained on Round 12.
4. THIS script.
5. Only AFTER reviewing this script's result: re-run
   prepare_phase3_dataset.py --include-holdout, then
   scripts/retrain_pipeline.py, to fold Round 12 into the production model
   ahead of Round 13 (Task_List.txt's "Retrain on R1-R11 including
   Olanda"). Running that retrain BEFORE this script would make this
   evaluation meaningless (scoring the model against data it already
   trained on) - this script reads
   final_model_metadata.json's `trained_with_holdout` flag and refuses to
   run if it's already True, to make that mistake loud instead of silent.

Saves models/lstm/holdout_evaluation.json (summary) and
models/lstm/holdout_evaluation.csv (full per-driver table, worst-error
first) - inputs for the model card and Progress_Reports' Phase 5 write-up.

Run from anywhere - paths are resolved relative to this file:

    python scripts/evaluate_holdout.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from f1qp.config import OFFLINE_TEST_ROUND
from f1qp.modeling.dataset import HOLDOUT_ROUND, HOLDOUT_YEAR
from f1qp.modeling.holdout_eval import build_scored_dataframe, metadata_trained_with_holdout, summarize
from f1qp.serving.predict import load_production_artifacts, predict_event_from_disk
from f1qp.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "lstm"

# f1qp.config and f1qp.modeling.dataset each independently name the offline
# test round (they're read by different parts of the pipeline and have no
# import relationship to each other) - this guards against the two ever
# silently drifting apart.
assert OFFLINE_TEST_ROUND == HOLDOUT_ROUND, (
    f"f1qp.config.OFFLINE_TEST_ROUND ({OFFLINE_TEST_ROUND}) and "
    f"f1qp.modeling.dataset.HOLDOUT_ROUND ({HOLDOUT_ROUND}) have drifted "
    f"apart - these must name the same round."
)


class HoldoutAlreadyTrainedError(RuntimeError):
    """Raised when final_model_metadata.json shows the production model
    already trained on the holdout round - see module docstring, step 5."""


def _load_metadata() -> dict:
    metadata_path = MODELS_DIR / "final_model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found - train the production model first "
            f"(scripts/train_final_lstm.py or scripts/retrain_pipeline.py)."
        )
    with open(metadata_path) as f:
        return json.load(f)


def _require_not_already_trained_on_holdout(metadata: dict) -> None:
    if metadata_trained_with_holdout(metadata):
        raise HoldoutAlreadyTrainedError(
            f"final_model_metadata.json says trained_with_holdout=True - the "
            f"current production model already trained on {HOLDOUT_YEAR} "
            f"R{HOLDOUT_ROUND}. Restore a pre-holdout model from "
            f"models/lstm/history/ before running this offline test, or "
            f"accept this run only checks in-sample fit, not real "
            f"generalization - see this script's module docstring."
        )


def _load_real_targets() -> pd.DataFrame:
    dataset_path = DATA_DIR / "phase3_dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} not found - run scripts/prepare_phase3_dataset.py first."
        )
    merged = pd.read_parquet(dataset_path)
    holdout = merged[(merged["Year"] == HOLDOUT_YEAR) & (merged["RoundNumber"] == HOLDOUT_ROUND)]
    if holdout.empty:
        raise FileNotFoundError(
            f"No {HOLDOUT_YEAR} R{HOLDOUT_ROUND} rows in {dataset_path} - run "
            f"extract_qualifying_targets.py + build_features.py + "
            f"prepare_phase3_dataset.py for this round first."
        )
    if (holdout["split"] != "holdout").any():
        logger.warning(
            "%s R%s is not labeled 'holdout' in phase3_dataset.parquet - "
            "prepare_phase3_dataset.py looks like it was already re-run with "
            "--include-holdout. Continuing, but this evaluation is no longer "
            "testing against data the model never trained on.",
            HOLDOUT_YEAR, HOLDOUT_ROUND,
        )
    return (
        holdout[["Driver", "final_quali_time", "has_target"]]
        .drop_duplicates(subset=["Driver"])
        .reset_index(drop=True)
    )


def main() -> None:
    metadata = _load_metadata()
    _require_not_already_trained_on_holdout(metadata)

    targets_df = _load_real_targets()

    artifacts = load_production_artifacts()
    predictions = predict_event_from_disk(HOLDOUT_YEAR, HOLDOUT_ROUND, artifacts)
    predictions_df = pd.DataFrame([p.__dict__ for p in predictions]).rename(columns={"driver": "Driver"})

    n_missing_target = int(
        len(set(predictions_df["Driver"]) - set(targets_df.loc[targets_df["has_target"], "Driver"]))
    )
    scored = build_scored_dataframe(predictions_df, targets_df)
    summary = summarize(
        scored,
        n_missing_target=n_missing_target,
        interval_level_pct=float(predictions_df["interval_level_pct"].iloc[0]),
    )
    summary.update(
        {
            "year": HOLDOUT_YEAR,
            "round_number": HOLDOUT_ROUND,
            "event": "Zandvoort (Dutch GP, \"Olanda\") - offline validation",
            "production_model_trained_at_utc": metadata.get("trained_at_utc"),
            "production_model_n_train": metadata.get("n_train"),
            "reference_leave_one_round_out_pooled_mape": metadata.get(
                "reference_leave_one_round_out_pooled_mape"
            ),
        }
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = MODELS_DIR / "holdout_evaluation.json"
    out_csv = MODELS_DIR / "holdout_evaluation.csv"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    scored.sort_values("abs_error_seconds", ascending=False).to_csv(out_csv, index=False)

    print("\n=== Olanda (Zandvoort, Round 12) offline test ===", flush=True)
    print(
        f"Scored {summary['n_drivers_scored']} drivers "
        f"({summary['n_drivers_missing_target']} without a real target - DNS/DSQ/no match)",
        flush=True,
    )
    print(f"MAPE: {summary['mape_pct']:.3f}%   R^2: {summary['r2']:.3f}", flush=True)
    print(
        f"{summary['interval_level_pct']:.0f}% typical-range interval: empirical coverage "
        f"{summary['interval_empirical_coverage_pct']:.1f}% "
        f"({int(scored['within_interval'].sum())}/{len(scored)} drivers)",
        flush=True,
    )
    print(
        f"\nFor reference, the reference leave-one-round-out pooled MAPE across "
        f"the 11 training rounds was "
        f"{summary['reference_leave_one_round_out_pooled_mape']}% - this single "
        f"held-out round's MAPE above is one real data point, not a second "
        f"cross-validation.",
        flush=True,
    )
    print("\nWorst 5 predictions:", flush=True)
    print(
        scored.sort_values("abs_error_seconds", ascending=False)
        .head(5)[["Driver", "final_quali_time", "predicted_quali_time_seconds", "abs_error_seconds"]]
        .to_string(index=False),
        flush=True,
    )
    print(f"\nSaved {out_json}\nSaved {out_csv}", flush=True)


if __name__ == "__main__":
    main()
