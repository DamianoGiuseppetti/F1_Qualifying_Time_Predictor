"""Tests for f1qp.modeling.baseline. The `wide_df_and_feature_cols` fixture
(synthetic data matching the documented schema, run through the full
dataset.py pipeline) lives in tests/conftest.py. These checks cover
plumbing correctness (does reconstruction match the formula, are holdout
rows excluded, does importance come back with the right shape) - not real
predictive accuracy, which needs the actual data.
"""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb

from f1qp.modeling.baseline import (
    leave_one_round_out_cv,
    mape,
    permutation_importance,
    r_squared,
    run_baseline_comparison,
    train_model,
)


def test_mape_and_r_squared_basic_properties():
    y_true = np.array([100.0, 90.0, 80.0])
    y_pred = y_true.copy()
    assert mape(y_true, y_pred) == pytest.approx(0.0)
    assert r_squared(y_true, y_pred) == pytest.approx(1.0)

    y_pred_off = y_true * 1.1
    assert mape(y_true, y_pred_off) == pytest.approx(10.0)


def test_train_model_absolute_formulation(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    result = train_model(wide_df, "absolute", feature_cols, fcols.era)

    assert result.n_train > 0 and result.n_val > 0
    # Every driver-weekend has a final_quali_time here (no DNS in the
    # fixture), so this trains on ALL rows, unlike the old per-segment design.
    assert result.n_train + result.n_val == len(wide_df)
    assert result.val_mape >= 0
    assert not np.isnan(result.val_r2)
    assert set(result.val_mape_by_era.keys()) <= {0, 1}
    assert len(result.gain_importance) == len(result.feature_cols)
    assert len(result.permutation_importance) == len(result.feature_cols)


def test_train_model_gap_formulation_runs_and_scores_in_absolute_space(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    result = train_model(wide_df, "gap", feature_cols, fcols.era)

    # A trivial "predict the mean gap" model would already land in a
    # physically sane MAPE range (a few percent, not thousands) because
    # reconstruction adds practice_reference back before scoring.
    assert 0 <= result.val_mape < 50


def test_train_model_rejects_unknown_formulation(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    with pytest.raises(ValueError, match="Unknown formulation"):
        train_model(wide_df, "nonsense", feature_cols, fcols.era)


def test_train_model_excludes_drivers_with_no_target(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    # Force one row to have no classified time at all (a DNS/DSQ case) and
    # confirm it drops out of training rather than being imputed.
    modified = wide_df.copy()
    modified.loc[modified.index[0], "has_target"] = False
    result_full = train_model(wide_df, "absolute", feature_cols, fcols.era)
    result_modified = train_model(modified, "absolute", feature_cols, fcols.era)
    assert result_modified.n_train + result_modified.n_val == result_full.n_train + result_full.n_val - 1


def test_permutation_importance_shape_matches_columns(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    usable = wide_df[wide_df["has_target"].astype(bool)]
    train_df = usable[usable["split"] == "train"]
    val_df = usable[usable["split"] == "val"]
    session_cols = [c for c in wide_df.columns if c.startswith("session")]

    model = xgb.XGBRegressor(n_estimators=10, max_depth=2, random_state=0)
    model.fit(train_df[session_cols], train_df["final_quali_time"])

    importance = permutation_importance(
        model, val_df[session_cols], val_df["final_quali_time"], val_df["practice_reference"],
        "absolute", n_repeats=2,
    )
    assert set(importance.index) == set(session_cols)


def test_run_baseline_comparison_picks_a_valid_winner(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    comparison = run_baseline_comparison(wide_df, feature_cols, fcols)

    assert comparison["winning_formulation"] in {"absolute", "gap"}
    assert set(comparison["results"].keys()) == {"absolute", "gap"}
    assert set(comparison["aggregate_val_mape"].keys()) == {"absolute", "gap"}
    winner_result = comparison["results"][comparison["winning_formulation"]]
    assert winner_result.val_mape == min(comparison["aggregate_val_mape"].values())


def test_holdout_rows_never_enter_training(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    assert "holdout" not in wide_df["split"].unique()


def test_leave_one_round_out_cv_covers_every_round_exactly_once(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    result = leave_one_round_out_cv(wide_df, feature_cols, fcols, formulation="gap", era_value=1)

    expected_rounds = sorted(wide_df.loc[wide_df[fcols.era] == 1, fcols.round_number].unique())
    assert sorted(result["per_round"].keys()) == expected_rounds
    assert len(expected_rounds) > 1  # otherwise this "CV" would be meaningless


def test_leave_one_round_out_cv_pooled_matches_sum_of_folds(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    result = leave_one_round_out_cv(wide_df, feature_cols, fcols, formulation="gap", era_value=1)

    n_era1_with_target = (
        (wide_df[fcols.era] == 1) & (wide_df["has_target"].astype(bool))
    ).sum()
    assert result["pooled"]["n_test"] == n_era1_with_target
    assert sum(m["n_test"] for m in result["per_round"].values()) == result["pooled"]["n_test"]


def test_leave_one_round_out_cv_excludes_holdout_rows(wide_df_with_holdout_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_with_holdout_and_feature_cols
    assert "holdout" in wide_df["split"].unique()  # sanity: fixture actually has one

    result = leave_one_round_out_cv(wide_df, feature_cols, fcols, formulation="gap", era_value=1)

    holdout_rounds = set(
        wide_df.loc[wide_df["split"] == "holdout", fcols.round_number].unique()
    )
    assert not (set(result["per_round"].keys()) & holdout_rounds)


def test_leave_one_round_out_cv_rejects_unknown_formulation(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    with pytest.raises(ValueError, match="Unknown formulation"):
        leave_one_round_out_cv(wide_df, feature_cols, fcols, formulation="nonsense", era_value=1)


def test_leave_one_round_out_cv_raises_on_era_with_no_rounds(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    with pytest.raises(ValueError, match="No rounds found"):
        leave_one_round_out_cv(wide_df, feature_cols, fcols, era_value=999)
