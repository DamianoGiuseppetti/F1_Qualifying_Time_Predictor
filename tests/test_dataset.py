"""Tests for f1qp.modeling.dataset. Fixtures (synthetic features_df /
targets_df matching the documented schema) live in tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1qp.modeling.dataset import (
    ALL_FEATURE_COLS,
    DROPPED_FEATURES,
    add_practice_reference_and_gaps,
    assemble_dataset,
    build_weekend_split,
    coalesce_final_quali_time,
    compute_correlation_matrix,
    find_high_correlation_pairs,
    get_feature_columns,
    pivot_to_weekend_features,
    resolve_column,
    resolve_feature_columns,
    resolve_target_columns,
)


def test_resolve_column_finds_first_match():
    df = pd.DataFrame({"RoundNumber": [1]})
    assert resolve_column(df, ["Round", "RoundNumber"], "round") == "RoundNumber"


def test_resolve_column_raises_with_available_columns_listed():
    df = pd.DataFrame({"Foo": [1]})
    with pytest.raises(KeyError, match="Foo"):
        resolve_column(df, ["Year", "year"], "Year")


def test_resolve_feature_and_target_columns(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    assert fcols.year == "Year"
    assert fcols.session == "SessionCode"
    assert tcols.q1 == "Q1"


def test_assemble_dataset_join_and_masks(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)

    # Every practice-session row for a driver-weekend carries the same target.
    n_sessions_2023_r1_ver = ((features_df["Year"] == 2023) & (features_df["RoundNumber"] == 1)
                               & (features_df["Driver"] == "VER")).sum()
    assert n_sessions_2023_r1_ver == 3  # FP1/FP2/FP3
    matching = merged[(merged["Year"] == 2023) & (merged["RoundNumber"] == 1) & (merged["Driver"] == "VER")]
    assert len(matching) == 3
    assert matching["Q1"].nunique() == 1  # same target repeated across sessions

    # Masks: driver index 0 (VER) reaches Q1/Q2/Q3, index 2 (LEC) only Q1.
    lec_mask = (merged["Year"] == 2023) & (merged["RoundNumber"] == 1) & (merged["Driver"] == "LEC")
    lec_row = merged[lec_mask].iloc[0]
    assert lec_row["has_Q1"] and not lec_row["has_Q2"] and not lec_row["has_Q3"]
    ver_mask = (merged["Year"] == 2023) & (merged["RoundNumber"] == 1) & (merged["Driver"] == "VER")
    ver_row = merged[ver_mask].iloc[0]
    assert ver_row["has_Q1"] and ver_row["has_Q2"] and ver_row["has_Q3"]


def test_assemble_dataset_raises_on_duplicate_target_rows(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    dup_targets = pd.concat([targets_df, targets_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(Exception):  # pandas raises MergeError, a ValueError subclass
        assemble_dataset(features_df, dup_targets, fcols, tcols)


def test_build_weekend_split_holds_out_zandvoort_and_covers_everything(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)
    split_df = build_weekend_split(merged, fcols, val_fraction=0.3, seed=0)

    holdout_rows = split_df[(split_df["Year"] == 2026) & (split_df["RoundNumber"] == 12)]
    assert (holdout_rows["split"] == "holdout").all()

    # No weekend is split across train and val (row-level leakage check).
    per_weekend_splits = split_df.groupby(["Year", "RoundNumber"])["split"].nunique()
    assert (per_weekend_splits == 1).all()

    assert split_df["split"].isna().sum() == 0
    assert set(split_df["split"].unique()) <= {"train", "val", "holdout"}


def test_build_weekend_split_with_holdout_none_folds_zandvoort_into_trainval(features_df, targets_df):
    """Phase 5 addition: holdout=None (wired via
    scripts/prepare_phase3_dataset.py's --include-holdout flag) should stop
    excluding Round 12 entirely, not just relabel it - used for the final
    pre-Round-13 retrain after scripts/evaluate_holdout.py's offline test."""
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)
    split_df = build_weekend_split(merged, fcols, holdout=None, val_fraction=0.3, seed=0)

    assert "holdout" not in set(split_df["split"].unique())
    zandvoort_rows = split_df[(split_df["Year"] == 2026) & (split_df["RoundNumber"] == 12)]
    assert len(zandvoort_rows) > 0
    assert set(zandvoort_rows["split"].unique()) <= {"train", "val"}

    # Still no weekend split across train and val (row-level leakage check) -
    # true regardless of whether a holdout round exists.
    per_weekend_splits = split_df.groupby(["Year", "RoundNumber"])["split"].nunique()
    assert (per_weekend_splits == 1).all()
    assert split_df["split"].isna().sum() == 0


def test_build_weekend_split_default_still_holds_out_zandvoort(features_df, targets_df):
    """Regression guard: adding the holdout=None mode must not change the
    default (holdout unset) behaviour - covered already by
    test_build_weekend_split_holds_out_zandvoort_and_covers_everything above,
    duplicated narrowly here to pin the exact default-arg call signature."""
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)
    split_df = build_weekend_split(merged, fcols, val_fraction=0.3, seed=0)
    zandvoort_rows = split_df[(split_df["Year"] == 2026) & (split_df["RoundNumber"] == 12)]
    assert (zandvoort_rows["split"] == "holdout").all()


def test_get_feature_columns_drops_fuel_corrected_pace(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)
    feature_cols = get_feature_columns(merged)
    assert "fuel_corrected_pace" not in feature_cols
    assert "best_lap_time" in feature_cols
    assert len(feature_cols) == len(ALL_FEATURE_COLS) - len(DROPPED_FEATURES)


def test_get_feature_columns_raises_on_missing_expected_column(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols).drop(columns=["best_lap_time"])
    with pytest.raises(KeyError, match="best_lap_time"):
        get_feature_columns(merged)


def test_correlation_matrix_and_pairs_are_symmetric_and_thresholded(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)
    merged = build_weekend_split(merged, fcols)
    feature_cols = get_feature_columns(merged)

    train_mask = merged["split"] == "train"
    corr = compute_correlation_matrix(merged, feature_cols, train_mask)
    assert corr.shape == (len(feature_cols), len(feature_cols))
    assert np.allclose(np.diag(corr.values), 1.0)

    pairs = find_high_correlation_pairs(corr, threshold=1.01)  # nothing can exceed 1.0
    assert len(pairs) == 0


def test_practice_reference_is_field_best_and_gap_matches_definition(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)
    merged = add_practice_reference_and_gaps(merged, fcols)

    weekend = merged[(merged["Year"] == 2023) & (merged["RoundNumber"] == 1)]
    expected_reference = weekend["best_lap_time"].min()
    assert np.allclose(weekend["practice_reference"].unique(), expected_reference)

    ver_row = weekend[weekend["Driver"] == "VER"].iloc[0]
    assert np.isclose(ver_row["gap_Q1"], ver_row["Q1"] - expected_reference)

    # Masked segments get NaN gaps, not a computed (meaningless) value.
    lec_row = weekend[weekend["Driver"] == "LEC"].iloc[0]
    assert pd.isna(lec_row["gap_Q2"])
    assert pd.isna(lec_row["gap_Q3"])


def test_pivot_to_weekend_features_shapes_and_pads_sprint_slot(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)
    merged = build_weekend_split(merged, fcols)
    merged = add_practice_reference_and_gaps(merged, fcols)
    feature_cols = get_feature_columns(merged)

    wide = pivot_to_weekend_features(merged, feature_cols, fcols)

    # One row per (Year, RoundNumber, Driver) - no invented combinations.
    n_driver_weekends = merged[["Year", "RoundNumber", "Driver"]].drop_duplicates().shape[0]
    assert len(wide) == n_driver_weekends

    assert "session0_best_lap_time" in wide.columns
    assert "session1_best_lap_time" in wide.columns
    assert "session2_best_lap_time" in wide.columns

    # Sprint weekend (2024, round 2) only has FP1 (slot 0) + SQ (slot 1) -
    # slot 2 must be NaN, not zero-filled.
    sprint_row = wide[(wide["Year"] == 2024) & (wide["RoundNumber"] == 2)].iloc[0]
    assert pd.isna(sprint_row["session2_best_lap_time"])
    assert not pd.isna(sprint_row["session0_best_lap_time"])
    assert not pd.isna(sprint_row["session1_best_lap_time"])

    # Normal weekend has all three slots filled.
    normal_row = wide[(wide["Year"] == 2024) & (wide["RoundNumber"] == 1)].iloc[0]
    assert not pd.isna(normal_row["session2_best_lap_time"])


def test_final_quali_time_coalesces_to_the_segment_that_decided_grid_position(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)

    weekend = merged[(merged["Year"] == 2023) & (merged["RoundNumber"] == 1)]

    ver_row = weekend[weekend["Driver"] == "VER"].iloc[0]  # reaches Q1/Q2/Q3
    assert ver_row["reached_segment"] == "Q3"
    assert ver_row["final_quali_time"] == ver_row["Q3"]

    ham_row = weekend[weekend["Driver"] == "HAM"].iloc[0]  # reaches Q1/Q2 only
    assert ham_row["reached_segment"] == "Q2"
    assert ham_row["final_quali_time"] == ham_row["Q2"]

    lec_row = weekend[weekend["Driver"] == "LEC"].iloc[0]  # reaches Q1 only
    assert lec_row["reached_segment"] == "Q1"
    assert lec_row["final_quali_time"] == lec_row["Q1"]

    assert weekend["has_target"].all()  # everyone here has at least a Q1 time


def test_gap_final_uses_final_quali_time_not_q1(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)
    merged = add_practice_reference_and_gaps(merged, fcols)

    weekend = merged[(merged["Year"] == 2023) & (merged["RoundNumber"] == 1)]
    reference = weekend["practice_reference"].iloc[0]

    ver_row = weekend[weekend["Driver"] == "VER"].iloc[0]
    assert np.isclose(ver_row["gap_final"], ver_row["Q3"] - reference)
    assert not np.isclose(ver_row["gap_final"], ver_row["Q1"] - reference)


def test_pivot_raises_on_unrecognised_session_code(features_df, targets_df):
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)
    merged = build_weekend_split(merged, fcols)
    merged = add_practice_reference_and_gaps(merged, fcols)
    feature_cols = get_feature_columns(merged)

    bad = merged.copy()
    bad.loc[bad.index[0], "SessionCode"] = "FP99"
    with pytest.raises(KeyError, match="FP99"):
        pivot_to_weekend_features(bad, feature_cols, fcols)


def test_coalesce_final_quali_time_prefers_q3_then_q2_then_q1(targets_df):
    """Uses the shared `targets_df` fixture (tests/conftest.py): per
    WEEKENDS/make_targets_df, VER always reaches Q3, HAM reaches Q2 only
    (eliminated in Q2), LEC reaches Q1 only (eliminated in Q1) - the same
    fixture assemble_dataset's own tests rely on for this exact rule, so
    f1qp.serving.history.score_launch (which calls this function) shares
    it instead of re-deriving it (see this function's own docstring)."""
    tcols = resolve_target_columns(targets_df)
    result = coalesce_final_quali_time(targets_df, tcols)

    one_weekend = result[(result["Year"] == 2023) & (result["RoundNumber"] == 1)].set_index("Driver")
    original = targets_df[(targets_df["Year"] == 2023) & (targets_df["RoundNumber"] == 1)].set_index("Driver")

    assert one_weekend.loc["VER", "final_quali_time"] == pytest.approx(original.loc["VER", "Q3"])
    assert one_weekend.loc["HAM", "final_quali_time"] == pytest.approx(original.loc["HAM", "Q2"])
    assert one_weekend.loc["LEC", "final_quali_time"] == pytest.approx(original.loc["LEC", "Q1"])
    assert one_weekend["has_target"].all()


def test_coalesce_final_quali_time_has_target_false_for_no_result():
    tcols_source = pd.DataFrame({
        "Year": [2026], "RoundNumber": [13], "Driver": ["DNS"],
        "Q1": [np.nan], "Q2": [np.nan], "Q3": [np.nan],
    })
    tcols = resolve_target_columns(tcols_source)
    result = coalesce_final_quali_time(tcols_source, tcols)

    assert result.loc[0, "has_target"] == False  # noqa: E712
    assert pd.isna(result.loc[0, "final_quali_time"])


def test_coalesce_final_quali_time_output_columns():
    tcols_source = pd.DataFrame({
        "Year": [2026], "RoundNumber": [13], "Driver": ["VER"],
        "Q1": [91.0], "Q2": [90.2], "Q3": [90.1],
    })
    tcols = resolve_target_columns(tcols_source)
    result = coalesce_final_quali_time(tcols_source, tcols)
    assert list(result.columns) == ["Year", "RoundNumber", "Driver", "final_quali_time", "has_target"]
