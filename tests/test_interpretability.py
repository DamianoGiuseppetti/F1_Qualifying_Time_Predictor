"""Tests for f1qp.modeling.interpretability - all synthetic, hand-computable
data. No `shap` or `torch` dependency needed to run these (the module
itself deliberately imports neither) - the actual SHAP explainer call
lives entirely in scripts/analyze_lstm.py, which is what real data will
exercise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1qp.modeling.interpretability import (
    aggregate_shap_by_feature,
    diagnose_learning_curve,
    flatten_features_for_shap,
    select_error_buckets,
    unflatten_features_from_shap,
)


def test_diagnose_learning_curve_hand_computed_overfitting_case():
    # train_loss monotonically decreasing throughout; val_mape best at
    # epoch 3, then gets WORSE (not just plateaus) - the classic
    # small-sample overfitting signature.
    history_df = pd.DataFrame({
        "epoch": [1, 2, 3, 4, 5, 6],
        "train_loss": [2.0, 1.5, 1.2, 1.0, 0.9, 0.85],
        "val_mape": [3.0, 2.0, 1.5, 1.7, 1.6, 1.8],
    })
    diagnosis = diagnose_learning_curve(history_df)

    assert diagnosis.best_epoch == 3
    assert diagnosis.final_epoch == 6
    assert diagnosis.best_val_mape == pytest.approx(1.5)
    assert diagnosis.train_loss_at_best_epoch == pytest.approx(1.2)
    assert diagnosis.train_loss_at_final_epoch == pytest.approx(0.85)
    assert diagnosis.train_loss_kept_falling_after_best is True
    assert diagnosis.val_mape_worst_after_best == pytest.approx(1.8)
    assert diagnosis.val_mape_drift_after_best == pytest.approx(0.3)
    assert diagnosis.epochs_trained_past_best == 3
    assert diagnosis.overfitting_signature is True


def test_diagnose_learning_curve_no_overfitting_when_best_is_last_epoch():
    # val_mape monotonically improving - best epoch IS the final epoch, so
    # there's nothing "after best" to have gotten worse.
    history_df = pd.DataFrame({
        "epoch": [1, 2, 3],
        "train_loss": [2.0, 1.5, 1.2],
        "val_mape": [3.0, 2.0, 1.0],
    })
    diagnosis = diagnose_learning_curve(history_df)

    assert diagnosis.best_epoch == 3
    assert diagnosis.epochs_trained_past_best == 0
    assert diagnosis.train_loss_kept_falling_after_best is False
    assert diagnosis.val_mape_worst_after_best == pytest.approx(1.0)
    assert diagnosis.val_mape_drift_after_best == pytest.approx(0.0)
    assert diagnosis.overfitting_signature is False


def test_diagnose_learning_curve_raises_on_empty_history():
    with pytest.raises(ValueError, match="empty"):
        diagnose_learning_curve(pd.DataFrame(columns=["epoch", "train_loss", "val_mape"]))


def test_flatten_and_unflatten_round_trip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3, 4)).astype(np.float32)
    era = np.array([0, 1, 0, 1, 1], dtype=np.float32)

    flat = flatten_features_for_shap(X, era)
    assert flat.shape == (5, 3 * 4 + 1)

    X_back, era_back = unflatten_features_from_shap(flat, n_features=4)
    np.testing.assert_allclose(X_back, X)
    np.testing.assert_allclose(era_back, era)


def test_flatten_raises_on_wrong_session_count():
    X = np.zeros((5, 2, 4), dtype=np.float32)  # only 2 sessions, not MAX_SESSIONS=3
    era = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError, match="3 session slots"):
        flatten_features_for_shap(X, era)


def test_unflatten_raises_on_wrong_column_count():
    flat = np.zeros((5, 10), dtype=np.float32)  # should be 3*4+1=13 for n_features=4
    with pytest.raises(ValueError, match="Expected 13 columns"):
        unflatten_features_from_shap(flat, n_features=4)


def test_aggregate_shap_by_feature_hand_computed():
    # 2 features (feat_a, feat_b), MAX_SESSIONS=3, era last column ->
    # flat layout: [a_slot0, b_slot0, a_slot1, b_slot1, a_slot2, b_slot2, era]
    # 2 instances.
    shap_values = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5],
        [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -0.5],
    ])
    result = aggregate_shap_by_feature(shap_values, ["feat_a", "feat_b"])

    # feat_a columns are indices 0, 2, 4: instance0 -> [1,3,5], instance1 -> [-1,1,1]
    feat_a = result[result["feature"] == "feat_a"].iloc[0]
    all_a_values = np.array([1.0, 3.0, 5.0, -1.0, 1.0, 1.0])
    assert feat_a["mean_abs_shap"] == pytest.approx(np.mean(np.abs(all_a_values)))
    assert feat_a["mean_signed_shap"] == pytest.approx(np.mean(all_a_values))

    # feat_b columns are indices 1, 3, 5: instance0 -> [2,4,6], instance1 -> [0,0,0]
    feat_b = result[result["feature"] == "feat_b"].iloc[0]
    all_b_values = np.array([2.0, 4.0, 6.0, 0.0, 0.0, 0.0])
    assert feat_b["mean_abs_shap"] == pytest.approx(np.mean(np.abs(all_b_values)))
    assert feat_b["mean_signed_shap"] == pytest.approx(np.mean(all_b_values))

    # era column is index 6: [0.5, -0.5]
    era_row = result[result["feature"] == "era"].iloc[0]
    assert era_row["mean_abs_shap"] == pytest.approx(0.5)
    assert era_row["mean_signed_shap"] == pytest.approx(0.0)

    # feat_a and feat_b are tied at mean_abs_shap=2.0 here (by construction);
    # era's 0.5 is clearly smaller - just check it sorts to the bottom,
    # rather than asserting an exact tie-break order that isn't guaranteed.
    assert result.iloc[-1]["feature"] == "era"
    assert set(result["feature"][:2]) == {"feat_a", "feat_b"}


def test_aggregate_shap_by_feature_raises_on_wrong_column_count():
    shap_values = np.zeros((3, 5))  # should be 3*2+1=7 for 2 features
    with pytest.raises(ValueError, match="Expected 7 columns"):
        aggregate_shap_by_feature(shap_values, ["feat_a", "feat_b"])


def test_select_error_buckets_hand_computed():
    abs_residuals = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    low_idx, high_idx = select_error_buckets(abs_residuals, bucket_size=2)

    # Two smallest values are 1.0 (idx 1) and 2.0 (idx 3).
    assert set(low_idx.tolist()) == {1, 3}
    # Two largest values are 5.0 (idx 0) and 4.0 (idx 4).
    assert set(high_idx.tolist()) == {0, 4}
    assert abs_residuals[low_idx].max() < abs_residuals[high_idx].min()


def test_select_error_buckets_shrinks_when_pool_too_small():
    abs_residuals = np.array([3.0, 1.0, 2.0])  # n=3, requested bucket_size=5
    low_idx, high_idx = select_error_buckets(abs_residuals, bucket_size=5)

    assert len(low_idx) == 1  # n // 2 == 1
    assert len(high_idx) == 1
    assert abs_residuals[low_idx][0] == pytest.approx(1.0)
    assert abs_residuals[high_idx][0] == pytest.approx(3.0)


def test_select_error_buckets_raises_on_fewer_than_two_residuals():
    with pytest.raises(ValueError, match="at least 2"):
        select_error_buckets(np.array([1.0]), bucket_size=2)
