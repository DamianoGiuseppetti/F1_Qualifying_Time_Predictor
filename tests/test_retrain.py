"""Tests for f1qp.modeling.retrain - pure logic only, no torch/model
dependency (same pattern as f1qp.modeling.conformal/interpretability:
keep reusable logic testable without the heavy real-data pipeline)."""

from __future__ import annotations

import pytest

from f1qp.modeling.retrain import format_retrain_comparison, select_final_epoch_count


def test_select_final_epoch_count_picks_the_maximum():
    # Mirrors the real LORO per-fold best_epoch distribution that produced
    # FINAL_TRAIN_EPOCHS=8 the first time (mostly 4, several 7, one 8).
    assert select_final_epoch_count([4, 7, 4, 8, 4, 7, 4, 7, 4, 4, 7]) == 8


def test_select_final_epoch_count_single_fold():
    assert select_final_epoch_count([5]) == 5


def test_select_final_epoch_count_raises_on_empty_list():
    with pytest.raises(ValueError, match="at least 1"):
        select_final_epoch_count([])


def test_format_retrain_comparison_first_run_has_no_previous():
    current = {
        "n_train": 1602, "pooled_mape": 1.053, "pooled_r2": 0.989,
        "epoch_count": 8, "interval_50pct": 0.752,
    }
    result = format_retrain_comparison(None, current)
    assert "No previous production run" in result


def test_format_retrain_comparison_shows_both_values():
    previous = {
        "n_train": 1602, "pooled_mape": 1.053, "pooled_r2": 0.989,
        "epoch_count": 8, "interval_50pct": 0.752,
    }
    current = {
        "n_train": 1624, "pooled_mape": 0.981, "pooled_r2": 0.991,
        "epoch_count": 7, "interval_50pct": 0.688,
    }
    result = format_retrain_comparison(previous, current)
    assert "1602" in result and "1624" in result
    assert "1.053" in result and "0.981" in result
    assert "8" in result and "7" in result
    assert "0.752" in result and "0.688" in result


def test_format_retrain_comparison_handles_missing_fields_gracefully():
    previous = {"n_train": 1602, "pooled_mape": None, "pooled_r2": 0.989, "epoch_count": 8, "interval_50pct": 0.752}
    current = {"n_train": 1624, "pooled_mape": 0.981, "pooled_r2": 0.991, "epoch_count": 7, "interval_50pct": 0.688}
    result = format_retrain_comparison(previous, current)
    assert "?" in result  # the missing previous pooled_mape shows as "?", not a crash
