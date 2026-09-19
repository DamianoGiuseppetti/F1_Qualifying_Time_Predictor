"""Tests for f1qp.modeling.holdout_eval - pure scoring logic for the Olanda
(Zandvoort) offline test. Hand-computed synthetic data throughout, no
model/artifact dependency (same pattern as tests/test_retrain.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1qp.modeling.holdout_eval import build_scored_dataframe, metadata_trained_with_holdout, summarize


def _predictions_df():
    return pd.DataFrame(
        {
            "Driver": ["VER", "HAM", "LEC", "NOR"],
            "predicted_quali_time_seconds": [88.0, 89.0, 90.0, 87.5],
            "interval_low_seconds": [87.2, 88.2, 89.2, 86.7],
            "interval_high_seconds": [88.8, 89.8, 90.8, 88.3],
        }
    )


def _targets_df():
    return pd.DataFrame(
        {
            "Driver": ["VER", "HAM", "LEC", "NOR"],
            # VER: real time inside its interval, small error.
            # HAM: real time OUTSIDE its interval (a genuine miss).
            # LEC: DNS - has_target False, must be excluded from scoring.
            # NOR: not present in real targets at all (edge case).
            "final_quali_time": [88.3, 91.5, np.nan, 87.6],
            "has_target": [True, True, False, True],
        }
    )


def test_build_scored_dataframe_excludes_missing_targets():
    scored = build_scored_dataframe(_predictions_df(), _targets_df())
    assert set(scored["Driver"]) == {"VER", "HAM", "NOR"}  # LEC excluded (has_target=False)


def test_build_scored_dataframe_computes_abs_error_and_interval_coverage():
    scored = build_scored_dataframe(_predictions_df(), _targets_df()).set_index("Driver")

    assert scored.loc["VER", "abs_error_seconds"] == pytest.approx(0.3)
    assert bool(scored.loc["VER", "within_interval"]) is True  # 88.3 is inside [87.2, 88.8]

    assert scored.loc["HAM", "abs_error_seconds"] == pytest.approx(2.5)
    assert bool(scored.loc["HAM", "within_interval"]) is False  # 91.5 is outside [88.2, 89.8]

    assert scored.loc["NOR", "abs_error_seconds"] == pytest.approx(0.1)


def test_build_scored_dataframe_returns_empty_when_no_driver_has_a_target():
    all_missing = _targets_df().assign(has_target=False)
    scored = build_scored_dataframe(_predictions_df(), all_missing)
    assert scored.empty


def test_summarize_computes_pooled_mape_r2_and_coverage():
    scored = build_scored_dataframe(_predictions_df(), _targets_df())
    summary = summarize(scored, n_missing_target=1, interval_level_pct=50.0)

    assert summary["n_drivers_scored"] == 3
    assert summary["n_drivers_missing_target"] == 1
    assert summary["interval_level_pct"] == 50.0
    # 2 of 3 scored drivers (VER, NOR) fall inside their interval.
    assert summary["interval_empirical_coverage_pct"] == pytest.approx(200 / 3)
    # Hand-computed MAPE: mean(|err|/true) * 100 over VER/HAM/NOR.
    errs = np.array([0.3 / 88.3, 2.5 / 91.5, 0.1 / 87.6])
    assert summary["mape_pct"] == pytest.approx(errs.mean() * 100, rel=1e-6)


def test_summarize_raises_on_empty_scored_dataframe():
    empty = build_scored_dataframe(_predictions_df(), _targets_df().assign(has_target=False))
    with pytest.raises(ValueError, match="No driver has a real qualifying target"):
        summarize(empty, n_missing_target=4, interval_level_pct=50.0)


def test_metadata_trained_with_holdout_reads_the_flag():
    assert metadata_trained_with_holdout({"trained_with_holdout": True}) is True
    assert metadata_trained_with_holdout({"trained_with_holdout": False}) is False


def test_metadata_trained_with_holdout_defaults_false_when_key_missing():
    # Older metadata, written before this Phase 5 addition, has no such key
    # at all - must default to False (the historical/expected state), not
    # raise or default to a false alarm.
    assert metadata_trained_with_holdout({}) is False
