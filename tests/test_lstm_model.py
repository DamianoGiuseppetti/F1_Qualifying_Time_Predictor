"""Tests for f1qp.modeling.lstm_model. Plumbing checks only (does a forward
pass run with variable-length sequences, does a short training loop
complete and produce finite metrics, does early stopping actually stop) -
not real predictive accuracy, which needs the actual data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from f1qp.modeling.lstm_model import (
    QualifyingLSTM,
    leave_one_round_out_cv_lstm,
    train_final_model,
    train_lstm,
)
from f1qp.modeling.sequences import FeatureImputer, FeatureScaler, build_lstm_sequences

_FAST_TRAIN_KWARGS = {"max_epochs": 3, "patience": 10, "batch_size": 16}


def _prepare_full(wide_df_and_feature_cols):
    """Like `_prepare` but for train_final_model's use case: ALL usable
    rows, no train/val split at all (imputer/scaler fit on everything,
    since this function is only ever meant to be called that way)."""
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)
    imputer = FeatureImputer.fit(batch.X, batch.mask)
    X_imputed = imputer.transform(batch.X, batch.mask)
    scaler = FeatureScaler.fit(X_imputed, batch.mask)
    X_scaled = scaler.transform(X_imputed, batch.mask)
    return batch, X_scaled, len(feature_cols)


def _prepare(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    batch = build_lstm_sequences(wide_df, feature_cols, fcols)
    train_rows = batch.split == "train"
    val_rows = batch.split == "val"
    imputer = FeatureImputer.fit(batch.X[train_rows], batch.mask[train_rows])
    X_imputed = imputer.transform(batch.X, batch.mask)
    scaler = FeatureScaler.fit(X_imputed[train_rows], batch.mask[train_rows])
    X_scaled = scaler.transform(X_imputed, batch.mask)
    return batch, train_rows, val_rows, X_scaled, len(feature_cols)


def test_forward_pass_handles_variable_lengths():
    torch.manual_seed(0)
    model = QualifyingLSTM(n_features=5, n_static=2, hidden_size=8)
    X = torch.zeros(4, 3, 5)
    X[:, 0, :] = 1.0  # every sample has at least FP1
    X[:2, 1, :] = 1.0  # first 2 samples also have FP2
    X[0, 2, :] = 1.0  # only sample 0 has FP3
    lengths = torch.tensor([3, 2, 1, 1])
    static = torch.zeros(4, 2)

    out = model(X, lengths, static)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_short_training_run_produces_finite_metrics(wide_df_and_feature_cols):
    batch, train_rows, val_rows, X_scaled, n_features = _prepare(wide_df_and_feature_cols)

    result = train_lstm(
        train_X=X_scaled[train_rows],
        train_lengths=batch.lengths[train_rows],
        train_static=batch.static[train_rows],
        train_y_gap=batch.y_gap[train_rows],
        val_X=X_scaled[val_rows],
        val_lengths=batch.lengths[val_rows],
        val_static=batch.static[val_rows],
        val_y_abs=batch.y_abs[val_rows],
        val_practice_reference=batch.practice_reference[val_rows],
        val_era=batch.era[val_rows],
        n_features=n_features,
        max_epochs=3,
        patience=10,
        batch_size=16,
        verbose=False,
    )

    assert len(result.history) == 3
    assert np.isfinite(result.best_val_mape)
    assert result.best_val_mape >= 0
    assert result.total_seconds > 0
    for m in result.history:
        assert np.isfinite(m.train_loss)
        assert np.isfinite(m.val_mape)


def test_early_stopping_actually_stops_early(wide_df_and_feature_cols):
    batch, train_rows, val_rows, X_scaled, n_features = _prepare(wide_df_and_feature_cols)

    result = train_lstm(
        train_X=X_scaled[train_rows],
        train_lengths=batch.lengths[train_rows],
        train_static=batch.static[train_rows],
        train_y_gap=batch.y_gap[train_rows],
        val_X=X_scaled[val_rows],
        val_lengths=batch.lengths[val_rows],
        val_static=batch.static[val_rows],
        val_y_abs=batch.y_abs[val_rows],
        val_practice_reference=batch.practice_reference[val_rows],
        val_era=batch.era[val_rows],
        n_features=n_features,
        max_epochs=200,
        patience=1,  # stop immediately after any non-improving epoch
        batch_size=16,
        verbose=False,
    )
    assert len(result.history) < 200


def test_best_state_is_reloaded_into_returned_model(wide_df_and_feature_cols):
    batch, train_rows, val_rows, X_scaled, n_features = _prepare(wide_df_and_feature_cols)

    result = train_lstm(
        train_X=X_scaled[train_rows],
        train_lengths=batch.lengths[train_rows],
        train_static=batch.static[train_rows],
        train_y_gap=batch.y_gap[train_rows],
        val_X=X_scaled[val_rows],
        val_lengths=batch.lengths[val_rows],
        val_static=batch.static[val_rows],
        val_y_abs=batch.y_abs[val_rows],
        val_practice_reference=batch.practice_reference[val_rows],
        val_era=batch.era[val_rows],
        n_features=n_features,
        max_epochs=5,
        patience=10,
        batch_size=16,
        verbose=False,
    )
    # Re-run the val forward pass with the returned (best-epoch) model and
    # confirm it reproduces the reported best_val_mape, not the last epoch's.
    from f1qp.modeling.baseline import mape as mape_fn

    result.model.eval()
    with torch.no_grad():
        pred_gap = result.model(
            torch.as_tensor(X_scaled[val_rows], dtype=torch.float32),
            torch.as_tensor(batch.lengths[val_rows], dtype=torch.int64),
            torch.as_tensor(batch.static[val_rows], dtype=torch.float32),
        ).numpy()
    pred_abs = pred_gap + batch.practice_reference[val_rows]
    recomputed_mape = mape_fn(batch.y_abs[val_rows], pred_abs)
    assert recomputed_mape == pytest.approx(result.best_val_mape, abs=1e-3)


def test_loro_lstm_covers_every_round_exactly_once(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)

    result = leave_one_round_out_cv_lstm(
        wide_df, feature_cols, fcols, era_value=1,
        train_kwargs=_FAST_TRAIN_KWARGS, verbose=False,
    )

    expected_rounds = sorted(wide_df.loc[wide_df[fcols.era] == 1, fcols.round_number].unique())
    assert sorted(result["per_round"].keys()) == expected_rounds
    assert len(expected_rounds) > 1  # otherwise this "CV" would be meaningless


def test_loro_lstm_pooled_matches_sum_of_folds(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)

    result = leave_one_round_out_cv_lstm(
        wide_df, feature_cols, fcols, era_value=1,
        train_kwargs=_FAST_TRAIN_KWARGS, verbose=False,
    )

    n_era1 = (wide_df[fcols.era] == 1).sum()
    assert result["pooled"]["n_test"] == n_era1
    assert sum(m["n_test"] for m in result["per_round"].values()) == result["pooled"]["n_test"]


def test_loro_lstm_excludes_holdout_rows(wide_df_with_holdout_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_with_holdout_and_feature_cols
    assert "holdout" in wide_df["split"].unique()  # sanity: fixture actually has one

    wide_df = wide_df[
        (wide_df["split"] != "holdout") & (wide_df["has_target"].astype(bool))
    ].reset_index(drop=True)
    result = leave_one_round_out_cv_lstm(
        wide_df, feature_cols, fcols, era_value=1,
        train_kwargs=_FAST_TRAIN_KWARGS, verbose=False,
    )

    holdout_rounds = set(
        wide_df.loc[wide_df["split"] == "holdout", fcols.round_number].unique()
    )
    assert not (set(result["per_round"].keys()) & holdout_rounds)


def test_loro_lstm_raises_on_era_with_no_rounds(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)
    with pytest.raises(ValueError, match="No rounds found"):
        leave_one_round_out_cv_lstm(
            wide_df, feature_cols, fcols, era_value=999,
            train_kwargs=_FAST_TRAIN_KWARGS, verbose=False,
        )


def test_loro_lstm_residuals_by_round_matches_per_round_n_test(wide_df_and_feature_cols):
    """Regression test for the residuals_by_round key added Aug 24 2026 for
    f1qp.modeling.conformal's split-conformal calibration - each round's
    residual array should line up exactly with that round's own per_round
    test count, and contain only finite (signed) values."""
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)

    result = leave_one_round_out_cv_lstm(
        wide_df, feature_cols, fcols, era_value=1,
        train_kwargs=_FAST_TRAIN_KWARGS, verbose=False,
    )

    assert set(result["residuals_by_round"].keys()) == set(result["per_round"].keys())
    for round_number, metrics in result["per_round"].items():
        residuals = result["residuals_by_round"][round_number]
        assert len(residuals) == metrics["n_test"]
        assert np.isfinite(residuals).all()


def test_loro_lstm_per_round_reports_internal_val_info(wide_df_and_feature_cols):
    wide_df, feature_cols, fcols = wide_df_and_feature_cols
    wide_df = wide_df[wide_df["has_target"].astype(bool)].reset_index(drop=True)

    result = leave_one_round_out_cv_lstm(
        wide_df, feature_cols, fcols, era_value=1,
        train_kwargs=_FAST_TRAIN_KWARGS, verbose=False,
    )
    for metrics in result["per_round"].values():
        assert np.isfinite(metrics["mape"])
        assert metrics["best_epoch"] >= 1
        assert np.isfinite(metrics["internal_val_mape"])


def test_train_final_model_runs_for_exact_epoch_count(wide_df_and_feature_cols):
    batch, X_scaled, n_features = _prepare_full(wide_df_and_feature_cols)

    result = train_final_model(
        train_X=X_scaled,
        train_lengths=batch.lengths,
        train_static=batch.static,
        train_y_gap=batch.y_gap,
        n_features=n_features,
        n_epochs=4,
        batch_size=16,
        verbose=False,
    )

    assert result.n_epochs == 4
    assert len(result.train_loss_history) == 4
    assert result.total_seconds > 0


def test_train_final_model_produces_finite_loss_every_epoch(wide_df_and_feature_cols):
    batch, X_scaled, n_features = _prepare_full(wide_df_and_feature_cols)

    result = train_final_model(
        train_X=X_scaled,
        train_lengths=batch.lengths,
        train_static=batch.static,
        train_y_gap=batch.y_gap,
        n_features=n_features,
        n_epochs=3,
        batch_size=16,
        verbose=False,
    )

    for loss in result.train_loss_history:
        assert np.isfinite(loss)


def test_train_final_model_returned_model_forward_pass_works(wide_df_and_feature_cols):
    batch, X_scaled, n_features = _prepare_full(wide_df_and_feature_cols)

    result = train_final_model(
        train_X=X_scaled,
        train_lengths=batch.lengths,
        train_static=batch.static,
        train_y_gap=batch.y_gap,
        n_features=n_features,
        n_epochs=2,
        batch_size=16,
        verbose=False,
    )

    result.model.eval()
    with torch.no_grad():
        pred = result.model(
            torch.as_tensor(X_scaled, dtype=torch.float32),
            torch.as_tensor(batch.lengths, dtype=torch.int64),
            torch.as_tensor(batch.static, dtype=torch.float32),
        )
    assert pred.shape == (len(batch.y_gap),)
    assert torch.isfinite(pred).all()


def test_train_final_model_has_no_validation_parameters():
    """Structural guard: train_final_model must never grow a `val_*`
    parameter - the whole point is that no held-out split is carved out
    of the final training data (see its docstring). If this test starts
    failing, someone added early stopping back in and should reconsider."""
    import inspect

    params = inspect.signature(train_final_model).parameters
    assert not any(name.startswith("val_") for name in params)
