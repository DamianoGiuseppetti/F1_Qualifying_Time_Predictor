"""Tests for f1qp.modeling.sequences. Uses the `features_df`/`targets_df`
fixtures from tests/conftest.py (built from WEEKENDS, which includes one
sprint weekend - 2024 round 2 - specifically so the padding/mask logic has
something real to exercise) plus the larger `wide_df_and_feature_cols`
fixture for shape checks at a size big enough to matter.
"""

from __future__ import annotations

import numpy as np
import pytest

from f1qp.modeling.dataset import (
    add_practice_reference_and_gaps,
    assemble_dataset,
    build_weekend_split,
    get_feature_columns,
    pivot_to_weekend_features,
    resolve_feature_columns,
    resolve_target_columns,
)
from f1qp.modeling.sequences import (
    FeatureImputer,
    FeatureScaler,
    MAX_SESSIONS,
    build_lstm_sequences,
)


@pytest.fixture
def wide_df_with_sprint_and_feature_cols(features_df, targets_df):
    """Built from the small WEEKENDS fixture (tests/conftest.py), which
    includes one sprint weekend (2024, round 2) with only FP1/SQ - the case
    build_lstm_sequences' padding logic exists for."""
    fcols = resolve_feature_columns(features_df)
    tcols = resolve_target_columns(targets_df)
    merged = assemble_dataset(features_df, targets_df, fcols, tcols)
    merged = build_weekend_split(merged, fcols, val_fraction=0.3, seed=0)
    feature_cols = get_feature_columns(merged)
    merged = add_practice_reference_and_gaps(merged, fcols)
    wide_df = pivot_to_weekend_features(merged, feature_cols, fcols)
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    return wide_df, feature_cols, fcols


def test_shapes_match_input_rows_and_max_sessions(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)

    n = len(wide_df)
    assert batch.X.shape == (n, MAX_SESSIONS, len(feature_cols))
    assert batch.mask.shape == (n, MAX_SESSIONS)
    assert batch.static.shape == (n, 2)
    assert batch.lengths.shape == (n,)
    assert batch.y_gap.shape == (n,)


def test_sprint_weekend_has_last_slot_masked_out(wide_df_with_sprint_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_with_sprint_and_feature_cols
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)

    is_sprint_row = wide_df[fcols.is_sprint].astype(bool).to_numpy()
    assert is_sprint_row.any(), "fixture should contain at least one sprint weekend"

    # Sprint weekends: only slots 0 and 1 real (FP1, SQ), slot 2 padded.
    assert (batch.mask[is_sprint_row, 0] == 1).all()
    assert (batch.mask[is_sprint_row, 1] == 1).all()
    assert (batch.mask[is_sprint_row, 2] == 0).all()
    assert (batch.lengths[is_sprint_row] == 2).all()

    # Normal weekends: all 3 slots real.
    assert (batch.mask[~is_sprint_row] == 1).all()
    assert (batch.lengths[~is_sprint_row] == 3).all()


def test_padded_slot_values_are_zero_before_and_after_scaling(wide_df_with_sprint_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_with_sprint_and_feature_cols
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)
    is_sprint_row = wide_df[fcols.is_sprint].astype(bool).to_numpy()

    assert np.all(batch.X[is_sprint_row, 2, :] == 0.0)

    scaler = FeatureScaler.fit(batch.X, batch.mask)
    scaled = scaler.transform(batch.X, batch.mask)
    assert np.all(scaled[is_sprint_row, 2, :] == 0.0)


def test_no_nan_anywhere_in_output_array(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)
    assert not np.isnan(batch.X).any()


def test_missing_pivoted_column_raises_keyerror(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    broken = wide_df.drop(columns=[f"session0_{feature_cols[0]}"])
    with pytest.raises(KeyError, match="missing expected pivoted columns"):
        build_lstm_sequences(broken, feature_cols, fcols)


def test_feature_scaler_fits_only_on_real_timesteps():
    # Two samples, 1 timestep of history each, 1 feature. Sample 0's real
    # value is 10, sample 1 is fully padded (mask 0) with a huge decoy
    # value that must NOT influence the fitted mean/std.
    X = np.array([[[10.0]], [[999.0]]], dtype=np.float32)
    mask = np.array([[1.0], [0.0]], dtype=np.float32)
    scaler = FeatureScaler.fit(X, mask)
    assert scaler.mean[0] == pytest.approx(10.0)
    assert scaler.std[0] == pytest.approx(1.0)  # guarded: a single real value has std 0 -> fallback 1.0


def test_zero_real_sessions_raises_valueerror(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    broken = wide_df.copy()
    for slot in range(MAX_SESSIONS):
        for feat in feature_cols:
            broken.loc[broken.index[0], f"session{slot}_{feat}"] = np.nan
    with pytest.raises(ValueError, match="zero real practice sessions"):
        build_lstm_sequences(broken, feature_cols, fcols)


def test_real_slot_with_one_missing_feature_is_kept_real_not_padded(wide_df_and_feature_cols):
    """Real-data regression test (Aug 23 2026): a session that happened can
    still have one individual feature come back NaN (e.g. no long run that
    session) without the whole slot being absent - confirmed on the real
    features.parquet, where the original all-or-nothing assumption raised.
    """
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    broken = wide_df.copy()
    target_row = broken.index[0]
    # NaN out exactly ONE feature in slot 0, leave the rest of that slot intact.
    broken.loc[target_row, f"session0_{feature_cols[0]}"] = np.nan

    batch = build_lstm_sequences(broken, feature_cols, fcols)
    row_pos = broken.index.get_loc(target_row)

    assert batch.mask[row_pos, 0] == 1  # still counted as a real session
    assert batch.lengths[row_pos] == 3
    assert np.isnan(batch.X[row_pos, 0, 0])  # the one missing feature stays NaN
    # every other feature in that same slot is untouched
    assert not np.isnan(batch.X[row_pos, 0, 1:]).any()


def test_feature_imputer_fills_with_train_median_ignoring_padding():
    # 3 samples x 1 timestep x 1 feature. Sample 2 is padded (huge decoy
    # value that must be ignored); sample 1 has a real NaN to fill.
    X = np.array([[[10.0]], [[np.nan]], [[999.0]]], dtype=np.float32)
    mask = np.array([[1.0], [1.0], [0.0]], dtype=np.float32)

    imputer = FeatureImputer.fit(X, mask)
    assert imputer.median[0] == pytest.approx(10.0)  # only sample 0 is real+non-NaN
    assert imputer.n_imputed_per_feature[0] == 1

    filled = imputer.transform(X, mask)
    assert filled[1, 0, 0] == pytest.approx(10.0)  # NaN filled with the median
    assert filled[2, 0, 0] == 0.0  # padded position stays 0, never gets the median
    assert not np.isnan(filled).any()


def test_feature_scaler_raises_on_unimputed_nan():
    X = np.array([[[10.0]], [[np.nan]]], dtype=np.float32)
    mask = np.array([[1.0], [1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="run FeatureImputer first"):
        FeatureScaler.fit(X, mask)


def test_impute_then_scale_pipeline_produces_no_nan(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    broken = wide_df.copy()
    broken.loc[broken.index[0], f"session0_{feature_cols[0]}"] = np.nan
    broken.loc[broken.index[3], f"session1_{feature_cols[2]}"] = np.nan

    batch = build_lstm_sequences(broken, feature_cols, fcols)
    train_rows = batch.split == "train"

    imputer = FeatureImputer.fit(batch.X[train_rows], batch.mask[train_rows])
    X_imputed = imputer.transform(batch.X, batch.mask)
    assert not np.isnan(X_imputed).any()

    scaler = FeatureScaler.fit(X_imputed[train_rows], batch.mask[train_rows])
    X_scaled = scaler.transform(X_imputed, batch.mask)
    assert not np.isnan(X_scaled).any()
