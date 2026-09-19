"""Tests for f1qp.modeling.tracking - Phase 5's MLflow logging wrapper.

Assumes mlflow IS installed (it's in requirements.txt, same convention as
torch/shap for the modules that need them) - these exercise the real
mlflow local SQLite-backed store against a tmp_path, not a mock, since
mlflow's own API surface is exactly what needs confidence here (this is
also what caught the real FileStore-deprecation bug this module was
originally written with - see f1qp.modeling.tracking's own module
docstring). The ImportError-degradation path is tested separately by
simulating mlflow being absent (`sys.modules["mlflow"] = None` - see
test_log_retrain_run_degrades_gracefully_without_mlflow), without needing
an environment that actually lacks it.
"""

from __future__ import annotations

import sys

import pytest
import torch

from f1qp.modeling.tracking import (
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    _sqlite_tracking_uri,
    log_retrain_run,
    search_runs,
)


def test_log_retrain_run_returns_a_run_id_and_persists_params_metrics(tmp_path):
    run_id = log_retrain_run(
        params={"n_train": 1602, "epoch_count": 8, "holdout_included": False},
        metrics={"pooled_mape": 1.053, "pooled_r2": 0.989, "interval_50pct": 0.752},
        mlruns_dir=tmp_path,
    )
    assert run_id is not None

    runs = search_runs(mlruns_dir=tmp_path)
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["run_id"] == run_id
    assert row["params.n_train"] == "1602"  # mlflow stores params as strings
    assert float(row["metrics.pooled_mape"]) == pytest.approx(1.053)


def test_log_retrain_run_drops_none_metrics_instead_of_raising(tmp_path):
    run_id = log_retrain_run(
        params={"n_train": 1602},
        metrics={"pooled_mape": 1.053, "interval_90pct": None},
        mlruns_dir=tmp_path,
    )
    assert run_id is not None
    runs = search_runs(mlruns_dir=tmp_path)
    assert "metrics.interval_90pct" not in runs.columns or runs["metrics.interval_90pct"].isna().all()


def test_log_retrain_run_registers_a_model_version(tmp_path):
    # Also the regression guard for the real Aug 25 2026 "Correction #2" bug:
    # this installed mlflow version defaults `serialization_format` to
    # "pt2", which requires an `input_example` to trace `model.forward` -
    # our LSTM's variable-length/packed-sequence signature isn't a fit for
    # that, so log_retrain_run pins `serialization_format="pickle"`
    # explicitly. Before that fix, this exact assertion (a version actually
    # getting registered) failed - log_model raised, was swallowed by the
    # inner try/except, and zero versions existed.
    tiny_model = torch.nn.Linear(4, 1)
    run_id = log_retrain_run(
        params={"n_train": 1602},
        metrics={"pooled_mape": 1.053},
        model=tiny_model,
        mlruns_dir=tmp_path,
    )
    assert run_id is not None

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(_sqlite_tracking_uri(tmp_path))
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    assert len(versions) >= 1
    assert any(v.run_id == run_id for v in versions)


def test_log_retrain_run_skips_missing_artifact_paths_without_raising(tmp_path):
    run_id = log_retrain_run(
        params={"n_train": 1602},
        metrics={"pooled_mape": 1.053},
        artifact_paths=[tmp_path / "does_not_exist.json"],
        mlruns_dir=tmp_path,
    )
    assert run_id is not None


def test_log_retrain_run_degrades_gracefully_without_mlflow(tmp_path, monkeypatch):
    # sys.modules[name] = None makes `import mlflow` raise ImportError
    # immediately (a documented CPython behaviour), without needing an
    # environment where mlflow is genuinely absent.
    monkeypatch.setitem(sys.modules, "mlflow", None)
    monkeypatch.setitem(sys.modules, "mlflow.pytorch", None)
    run_id = log_retrain_run(
        params={"n_train": 1602}, metrics={"pooled_mape": 1.053}, mlruns_dir=tmp_path
    )
    assert run_id is None


def test_log_retrain_run_still_logs_params_when_model_registration_fails(tmp_path, monkeypatch):
    # Regression guard: `mlflow.pytorch` (needed only to register a model)
    # imports torch, and a broken torch install can fail with something
    # other than ImportError (an OSError loading a shared library, observed
    # for real while verifying this module). That failure must cost only
    # the model-registration step - params/metrics logged just above it in
    # the same run must still go through, and the run must still get an id.
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "mlflow.pytorch" or name.startswith("mlflow.pytorch."):
            raise OSError("simulated broken torch shared library")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    tiny_model = torch.nn.Linear(4, 1)
    run_id = log_retrain_run(
        params={"n_train": 1602},
        metrics={"pooled_mape": 1.053},
        model=tiny_model,
        mlruns_dir=tmp_path,
    )
    assert run_id is not None  # the run itself still succeeded

    runs = search_runs(mlruns_dir=tmp_path)
    assert len(runs) == 1
    assert float(runs.iloc[0]["metrics.pooled_mape"]) == pytest.approx(1.053)


def test_search_runs_returns_empty_dataframe_when_tracking_dir_missing(tmp_path):
    missing_dir = tmp_path / "never_created"
    runs = search_runs(mlruns_dir=missing_dir)
    assert runs.empty


def test_search_runs_returns_empty_dataframe_when_experiment_has_no_runs(tmp_path):
    # Tracking dir exists (e.g. created by an earlier unrelated run) but no
    # run has ever been logged under EXPERIMENT_NAME here.
    tmp_path.mkdir(exist_ok=True)
    runs = search_runs(mlruns_dir=tmp_path)
    assert runs.empty


def test_sqlite_tracking_uri_not_a_deprecated_file_store(tmp_path):
    # Regression guard for the real Aug 25 2026 bug: a bare `file:...` URI
    # hits newer mlflow's FileStore-deprecation exception at
    # mlflow.set_tracking_uri/start_run time. Pin the scheme so a future
    # edit can't silently reintroduce it.
    uri = _sqlite_tracking_uri(tmp_path)
    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")


def test_experiment_and_model_name_constants_are_stable():
    # Regression guard: the dashboard and retrain_pipeline.py both import
    # these by name - a rename here would silently orphan old runs.
    assert EXPERIMENT_NAME == "f1_qualifying_predictor"
    assert REGISTERED_MODEL_NAME == "f1_qualifying_lstm"
