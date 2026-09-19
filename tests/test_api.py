"""Tests for f1qp.api.main - the FastAPI layer over f1qp.serving.predict.

Doesn't touch real production artifacts or a real features.parquet: the
lifespan startup hook is monkeypatched to load the same deterministic
"zeroed" model tests/test_predict.py uses (see its module docstring for
why), and `f1qp.serving.predict.FEATURES_PATH` is monkeypatched to a small
parquet file this test writes itself. This keeps these tests fast and
exact, and keeps them from silently depending on whatever happens to be in
the real models/ or data/ directories on whichever machine runs them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

import f1qp.api.main as api_main
import f1qp.serving.history as history_mod
import f1qp.serving.predict as predict_mod
from f1qp.modeling.lstm_model import QualifyingLSTM
from f1qp.modeling.sequences import FeatureImputer, FeatureScaler
from f1qp.serving.predict import ProductionArtifacts

FEATURE_COLS = ["best_lap_time", "feat_b"]
CONSTANT_GAP_PRED = 5.0


def _zeroed_model() -> QualifyingLSTM:
    model = QualifyingLSTM(n_features=len(FEATURE_COLS), n_static=2, hidden_size=8)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.head[3].bias.fill_(CONSTANT_GAP_PRED)
    model.eval()
    return model


def _fake_artifacts() -> ProductionArtifacts:
    n = len(FEATURE_COLS)
    return ProductionArtifacts(
        model=_zeroed_model(),
        imputer=FeatureImputer(median=np.zeros(n, dtype=np.float32)),
        scaler=FeatureScaler(mean=np.zeros(n, dtype=np.float32), std=np.ones(n, dtype=np.float32)),
        feature_cols=FEATURE_COLS,
        metadata={
            "trained_at_utc": "2026-08-24T21:23:34+00:00",
            "n_train": 1602,
            "n_epochs": 8,
            "reference_leave_one_round_out_pooled_mape": 1.053,
            "reference_leave_one_round_out_pooled_r2": 0.989,
            "note": "test fixture",
        },
        deployment_quantile_seconds=0.5,
        deployment_quantile_exact=True,
        coverage_target_pct=50.0,
    )


def _write_features_parquet(path, year=2099, round_number=13, include_test_driver=False):
    rows = []
    best_lap_time = {
        "FP1": {"VER": 90.0, "HAM": 91.0},
        "FP2": {"VER": 89.5, "HAM": 90.5},
        "FP3": {"VER": 89.0, "HAM": 90.0},
    }
    for session in ("FP1", "FP2", "FP3"):
        for driver in ("VER", "HAM"):
            rows.append({
                "Year": year,
                "RoundNumber": round_number,
                "Driver": driver,
                "SessionCode": session,
                "IsSprint": False,
                "Era": 1,
                "best_lap_time": best_lap_time[session][driver],
                "feat_b": 1.0,
            })
    if include_test_driver:
        rows.append({
            "Year": year, "RoundNumber": round_number, "Driver": "TST", "SessionCode": "FP1",
            "IsSprint": False, "Era": 1, "best_lap_time": 95.0, "feat_b": 1.0,
        })
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient with the lifespan startup patched to load
    `_fake_artifacts()` instead of touching the real models/lstm/,
    `f1qp.serving.predict.FEATURES_PATH` patched to a tmp parquet file this
    fixture writes with one known event (year=2099, round=13), and
    `f1qp.serving.history`'s launches dir / targets path both patched to
    tmp locations - keeps every launch/history test isolated from the
    real data/predictions/launches/ and data/processed/
    qualifying_targets.parquet, same isolation convention as the two
    monkeypatches above."""
    features_path = tmp_path / "features.parquet"
    _write_features_parquet(features_path)
    monkeypatch.setattr(predict_mod, "FEATURES_PATH", features_path)
    monkeypatch.setattr(api_main, "load_production_artifacts", lambda: _fake_artifacts())

    launches_dir = tmp_path / "launches"
    targets_path = tmp_path / "qualifying_targets.parquet"
    pd.DataFrame(columns=["Year", "RoundNumber", "Driver", "Q1", "Q2", "Q3"]).to_parquet(
        targets_path, index=False
    )
    monkeypatch.setattr(history_mod, "LAUNCHES_DIR", launches_dir)
    monkeypatch.setattr(history_mod, "TARGETS_PATH", targets_path)

    with TestClient(api_main.app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_trained_at_utc"] == "2026-08-24T21:23:34+00:00"


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_train"] == 1602
    assert body["n_epochs"] == 8
    assert body["deployment_quantile_seconds"] == pytest.approx(0.5)
    assert body["interval_level_pct"] == pytest.approx(50.0)
    assert body["interval_label"] == "typical range"


def test_predict_known_event_returns_both_drivers(client):
    resp = client.post("/predict/2099/13")
    assert resp.status_code == 200
    body = resp.json()
    assert body["year"] == 2099
    assert body["round_number"] == 13
    assert body["n_drivers"] == 2
    assert body["excluded_test_drivers"] == []

    expected_pred = CONSTANT_GAP_PRED + 89.0  # practice_reference = min best_lap_time (VER's FP3)
    by_driver = {p["driver"]: p for p in body["predictions"]}
    for driver in ("VER", "HAM"):
        p = by_driver[driver]
        assert p["n_practice_sessions"] == 3
        assert p["predicted_quali_time_seconds"] == pytest.approx(expected_pred)
        assert p["interval_low_seconds"] == pytest.approx(expected_pred - 0.5)
        assert p["interval_high_seconds"] == pytest.approx(expected_pred + 0.5)
        assert p["interval_label"] == "typical range"


def test_predict_unknown_event_returns_404(client):
    resp = client.post("/predict/2099/1")
    assert resp.status_code == 404
    assert "2099" in resp.json()["detail"]


def test_predict_without_loaded_artifacts_returns_503():
    """`TestClient(app)` used WITHOUT the `with` context manager never runs
    the lifespan startup hook (Starlette only runs it on __enter__), so
    `api_main._state["artifacts"]` stays whatever it's forced to here -
    exercises the `_artifacts()` guard's 503 branch directly, independent
    of the `client` fixture above and of any real models/lstm/ artifacts
    that may or may not exist on whatever machine runs this test."""
    api_main._state["artifacts"] = None
    plain_client = TestClient(api_main.app)
    resp = plain_client.get("/health")
    assert resp.status_code == 503


@pytest.fixture
def client_with_test_driver(tmp_path, monkeypatch):
    """Same isolation as `client` above, but `features.parquet` also has an
    FP1-only reserve/test driver (TST) - exercises the excluded_test_drivers
    field end-to-end through the API layer."""
    features_path = tmp_path / "features.parquet"
    _write_features_parquet(features_path, include_test_driver=True)
    monkeypatch.setattr(predict_mod, "FEATURES_PATH", features_path)
    monkeypatch.setattr(api_main, "load_production_artifacts", lambda: _fake_artifacts())

    launches_dir = tmp_path / "launches"
    targets_path = tmp_path / "qualifying_targets.parquet"
    pd.DataFrame(columns=["Year", "RoundNumber", "Driver", "Q1", "Q2", "Q3"]).to_parquet(
        targets_path, index=False
    )
    monkeypatch.setattr(history_mod, "LAUNCHES_DIR", launches_dir)
    monkeypatch.setattr(history_mod, "TARGETS_PATH", targets_path)

    with TestClient(api_main.app) as c:
        yield c


def test_predict_reports_excluded_test_drivers(client_with_test_driver):
    resp = client_with_test_driver.post("/predict/2099/13")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_drivers"] == 2
    assert body["excluded_test_drivers"] == ["TST"]
    assert "TST" not in {p["driver"] for p in body["predictions"]}


def test_launch_persists_and_appears_in_history_current(client):
    launch_resp = client.post("/predict/2099/13/launch")
    assert launch_resp.status_code == 200
    body = launch_resp.json()
    assert body["year"] == 2099
    assert body["round_number"] == 13
    assert body["excluded_test_drivers"] == []
    assert "launched_at_utc" in body and body["launched_at_utc"]

    current_resp = client.get("/history/current")
    assert current_resp.status_code == 200
    current = current_resp.json()
    assert len(current) == 1
    assert current[0]["year"] == 2099
    assert current[0]["round_number"] == 13
    assert current[0]["scored"] is False
    assert len(current[0]["rows"]) == 2


def test_launch_unknown_event_returns_404(client):
    resp = client.post("/predict/2099/1/launch")
    assert resp.status_code == 404


def test_history_current_empty_before_any_launch(client):
    resp = client.get("/history/current")
    assert resp.status_code == 200
    assert resp.json() == []


def test_history_empty_before_any_launch(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert resp.json() == {"rows": []}


def test_history_filters_by_season_and_round(client):
    client.post("/predict/2099/13/launch")

    all_rows = client.get("/history").json()["rows"]
    assert len(all_rows) == 2  # VER + HAM

    matching = client.get("/history", params={"season": 2099, "round_number": 13}).json()["rows"]
    assert len(matching) == 2

    no_match = client.get("/history", params={"season": 2099, "round_number": 1}).json()["rows"]
    assert no_match == []


def test_relaunching_same_round_keeps_history_to_one_entry(client):
    client.post("/predict/2099/13/launch")
    client.post("/predict/2099/13/launch")

    rows = client.get("/history").json()["rows"]
    # Still just VER + HAM once - a re-run replaces, not duplicates (see
    # f1qp.serving.history.record_launch's docstring).
    assert len(rows) == 2


def test_model_runs_empty_when_no_tracking_store(client, monkeypatch):
    """No mlruns/mlflow.db under wherever F1QP_MLRUNS_DIR points (or
    mlflow not installed) is search_runs()'s normal "no history yet"
    case (see its own docstring) - GET /model/runs should surface that as
    [], not an error, since this is expected early in a season before any
    retrain_pipeline.py run has ever happened."""
    monkeypatch.setattr(api_main, "search_runs", lambda mlruns_dir=None: pd.DataFrame())
    resp = client.get("/model/runs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_model_runs_returns_oldest_first_with_params_and_metrics(client, monkeypatch):
    """Two logged runs, deliberately inserted newest-first, to exercise
    the endpoint's own oldest-first sort - the Performance tab's
    "Retrain history" timeline expects chronological order. Also checks
    that params./metrics. column prefixes are stripped and NaN cells
    (a run that didn't log a given key) are dropped rather than
    surfacing as "NaN" in the response."""

    fake_runs = pd.DataFrame(
        [
            {
                "run_id": "run-2",
                "start_time": pd.Timestamp("2026-08-25T10:00:00+00:00"),
                "params.n_epochs": "8",
                "metrics.pooled_mape": 1.053,
                "metrics.pooled_r2": 0.989,
            },
            {
                "run_id": "run-1",
                "start_time": pd.Timestamp("2026-08-20T10:00:00+00:00"),
                "params.n_epochs": "6",
                "metrics.pooled_mape": float("nan"),
                "metrics.pooled_r2": 0.95,
            },
        ]
    )
    monkeypatch.setattr(api_main, "search_runs", lambda mlruns_dir=None: fake_runs)

    resp = client.get("/model/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert [r["run_id"] for r in body] == ["run-1", "run-2"]

    first = body[0]
    assert first["params"] == {"n_epochs": "6"}
    assert first["metrics"] == {"pooled_r2": pytest.approx(0.95)}
    assert "pooled_mape" not in first["metrics"]  # NaN cell dropped, not sent as null

    second = body[1]
    assert second["params"] == {"n_epochs": "8"}
    assert second["metrics"]["pooled_mape"] == pytest.approx(1.053)
    assert second["metrics"]["pooled_r2"] == pytest.approx(0.989)
