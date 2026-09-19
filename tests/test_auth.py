"""Tests for the Sep 13 2026 admin-token gating (f1qp.api.main.require_admin).

Point 4 of the "final version of the application" plan: guests get
Preview, only an admin token unlocks Launch and the on-demand data-fetch
triggers. Reuses the exact isolation pattern tests/test_api.py's own
`client` fixture already established (fake zeroed model, tmp_path
features/targets/launches, no real models/lstm/ or data/ touched) -
these tests only add ADMIN_TOKEN into the mix via monkeypatch.setenv, so
they never depend on (or affect) whatever's actually in a real shell's
environment.
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
ADMIN_TOKEN = "s3cr3t-test-token"


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


def _write_features_parquet(path, year=2099, round_number=13):
    rows = []
    best_lap_time = {"FP1": {"VER": 90.0, "HAM": 91.0}, "FP2": {"VER": 89.5, "HAM": 90.5}, "FP3": {"VER": 89.0, "HAM": 90.0}}
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
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Same isolation as test_api.py's `client` fixture - see that file's
    docstring. ADMIN_TOKEN is explicitly NOT set here (delenv, in case a
    real shell happens to export one) - individual tests opt into gating
    via monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN) themselves."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)

    features_path = tmp_path / "features.parquet"
    _write_features_parquet(features_path)
    monkeypatch.setattr(predict_mod, "FEATURES_PATH", features_path)
    monkeypatch.setattr(api_main, "load_production_artifacts", lambda: _fake_artifacts())
    monkeypatch.setattr(api_main, "start_fetch_job", lambda year, round_number: "fake-fetch-job")
    monkeypatch.setattr(api_main, "start_results_job", lambda year, round_number: "fake-results-job")

    launches_dir = tmp_path / "launches"
    targets_path = tmp_path / "qualifying_targets.parquet"
    pd.DataFrame(columns=["Year", "RoundNumber", "Driver", "Q1", "Q2", "Q3"]).to_parquet(targets_path, index=False)
    monkeypatch.setattr(history_mod, "LAUNCHES_DIR", launches_dir)
    monkeypatch.setattr(history_mod, "TARGETS_PATH", targets_path)

    with TestClient(api_main.app) as c:
        yield c


def test_admin_not_required_by_default(client):
    resp = client.get("/auth/status")
    assert resp.status_code == 200
    assert resp.json() == {"admin_required": False}


def test_launch_works_without_any_token_when_admin_not_required(client):
    """No ADMIN_TOKEN set at all (the local docker-compose case) - Launch
    must keep working with zero friction, exactly as it did before this
    gating existed."""
    resp = client.post("/predict/2099/13/launch")
    assert resp.status_code == 200


def test_data_fetch_works_without_any_token_when_admin_not_required(client):
    resp = client.post("/data/fetch/2026/14")
    assert resp.status_code == 200
    assert resp.json() == {"job_id": "fake-fetch-job"}


def test_admin_required_once_token_configured(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    resp = client.get("/auth/status")
    assert resp.status_code == 200
    assert resp.json() == {"admin_required": True}


def test_preview_never_gated_even_when_admin_required(client, monkeypatch):
    """Preview (POST /predict/... WITHOUT /launch) must stay open to
    everyone regardless of ADMIN_TOKEN - only persisting a launch, or
    triggering a fetch, is restricted."""
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    resp = client.post("/predict/2099/13")
    assert resp.status_code == 200


def test_launch_blocked_without_token_when_admin_required(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    resp = client.post("/predict/2099/13/launch")
    assert resp.status_code == 401
    assert "admin token" in resp.json()["detail"].lower()


def test_launch_blocked_with_wrong_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    resp = client.post(
        "/predict/2099/13/launch", headers={"X-Admin-Token": "not-the-right-token"}
    )
    assert resp.status_code == 401


def test_launch_allowed_with_correct_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    resp = client.post(
        "/predict/2099/13/launch", headers={"X-Admin-Token": ADMIN_TOKEN}
    )
    assert resp.status_code == 200


def test_data_fetch_and_fetch_results_blocked_without_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    assert client.post("/data/fetch/2026/14").status_code == 401
    assert client.post("/data/fetch-results/2026/14").status_code == 401


def test_data_fetch_and_fetch_results_allowed_with_correct_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    headers = {"X-Admin-Token": ADMIN_TOKEN}
    assert client.post("/data/fetch/2026/14", headers=headers).status_code == 200
    assert client.post("/data/fetch-results/2026/14", headers=headers).status_code == 200


def test_auth_check_endpoint_reflects_token_validity(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    assert client.post("/auth/check").status_code == 401
    assert client.post("/auth/check", headers={"X-Admin-Token": "wrong"}).status_code == 401
    ok_resp = client.post("/auth/check", headers={"X-Admin-Token": ADMIN_TOKEN})
    assert ok_resp.status_code == 200
    assert ok_resp.json() == {"ok": True}


def test_auth_check_always_ok_when_admin_not_required(client):
    """Gating off entirely (no ADMIN_TOKEN) - /auth/check is trivially ok,
    same as every gated route in that state."""
    resp = client.post("/auth/check")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_read_only_routes_never_gated(client, monkeypatch):
    """GET routes (health, history, model info/runs, readiness, job
    status) were never in scope for this gating - confirms none of them
    accidentally picked up the dependency."""
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    assert client.get("/health").status_code == 200
    assert client.get("/model/info").status_code == 200
    assert client.get("/history").status_code == 200
    assert client.get("/history/current").status_code == 200
    assert client.get("/data/readiness/2026/14").status_code == 200
