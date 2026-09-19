"""Tests for f1qp.serving.predict.

Same convention `tests/test_lstm_model.py` already uses for anything that
touches `QualifyingLSTM`: plumbing/shape/exception checks, not hand-derived
LSTM math (infeasible by hand for a real gated recurrence). Where an exact
expected NUMBER is worth asserting - the practice_reference/interval
arithmetic this module adds on top of the model - the model itself is
made deterministic by zeroing every parameter except the final layer's
bias, so its forward pass is a known constant `B` for any input (h_last is
always 0 -> ReLU(0)=0 -> Dropout is a no-op in eval() -> final Linear(32,1)
with weight=0 collapses to just its bias). That isolates exactly the new
logic in this module (practice_reference lookup, conformal interval
arithmetic, per-driver session counts) from the untestable-by-hand part
(what the LSTM itself computes), the same separation of concerns
test_lstm_model.py already draws.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from f1qp.modeling.lstm_model import QualifyingLSTM
from f1qp.modeling.sequences import FeatureImputer, FeatureScaler
from f1qp.serving.predict import (
    ArtifactsNotFoundError,
    DriverPrediction,
    EventNotFoundError,
    ProductionArtifacts,
    identify_test_drivers,
    load_features_for_event,
    load_production_artifacts,
    predict_event,
)

FEATURE_COLS = ["best_lap_time", "feat_b"]
CONSTANT_GAP_PRED = 5.0  # B - see module docstring


def _zeroed_model(constant_pred: float = CONSTANT_GAP_PRED) -> QualifyingLSTM:
    # No hidden_size override here on purpose (Sep 10 2026 fix): this model's
    # state_dict is what _write_fake_artifacts() torch.save()s and what
    # load_production_artifacts() then torch.load()s back into a FRESH
    # QualifyingLSTM(n_features=...) - i.e. the real production default
    # (hidden_size=64, see lstm_model.py). A stale hardcoded hidden_size=8
    # here (leftover from before the production default was set to 64) is
    # exactly what caused load_state_dict's "size mismatch" RuntimeError in
    # both tests that round-trip through the real files (happy_path and
    # missing_era1_entry) - the other tests using this helper never
    # serialize/reload, so they never surfaced it. Always inherit
    # QualifyingLSTM's own default instead of a second hardcoded number that
    # can drift out of sync with it again.
    model = QualifyingLSTM(n_features=len(FEATURE_COLS), n_static=2)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.head[3].bias.fill_(constant_pred)  # head[3] = final Linear(32, 1)
    model.eval()
    return model


def _identity_artifacts(
    constant_pred: float = CONSTANT_GAP_PRED, quantile: float = 0.5
) -> ProductionArtifacts:
    n = len(FEATURE_COLS)
    return ProductionArtifacts(
        model=_zeroed_model(constant_pred),
        imputer=FeatureImputer(median=np.zeros(n, dtype=np.float32)),
        scaler=FeatureScaler(mean=np.zeros(n, dtype=np.float32), std=np.ones(n, dtype=np.float32)),
        feature_cols=FEATURE_COLS,
        metadata={"trained_at_utc": "2026-08-24T00:00:00+00:00"},
        deployment_quantile_seconds=quantile,
        deployment_quantile_exact=True,
        coverage_target_pct=50.0,
    )


def _event_features_df(sessions=("FP1", "FP2", "FP3"), year=2099, round_number=99) -> pd.DataFrame:
    """VER and HAM, `sessions` real practice sessions each, best_lap_time
    chosen so the weekend's practice_reference (min across everyone) is
    known by construction."""
    best_lap_time = {
        "FP1": {"VER": 90.0, "HAM": 91.0},
        "FP2": {"VER": 89.5, "HAM": 90.5},
        "FP3": {"VER": 89.0, "HAM": 90.0},
    }
    rows = []
    for session in sessions:
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
    return pd.DataFrame(rows)


def test_predict_event_full_weekend_matches_hand_computed_values():
    artifacts = _identity_artifacts(constant_pred=5.0, quantile=0.5)
    event_df = _event_features_df(sessions=("FP1", "FP2", "FP3"))

    results = predict_event(event_df, artifacts)

    assert len(results) == 2
    by_driver = {r.driver: r for r in results}
    # practice_reference = min best_lap_time across the WHOLE weekend (both
    # drivers, all 3 sessions) = 89.0 (VER's FP3). Same for both drivers.
    expected_pred = 5.0 + 89.0
    for driver in ("VER", "HAM"):
        r = by_driver[driver]
        assert r.n_practice_sessions == 3
        assert r.is_sprint is False
        assert r.era == 1
        assert r.predicted_quali_time_seconds == pytest.approx(expected_pred)
        assert r.interval_low_seconds == pytest.approx(expected_pred - 0.5)
        assert r.interval_high_seconds == pytest.approx(expected_pred + 0.5)
        assert r.interval_width_seconds == pytest.approx(1.0)
        assert r.interval_level_pct == pytest.approx(50.0)
        assert r.interval_exact is True
        assert r.interval_label == "typical range"


def test_predict_event_partial_weekend_uses_fewer_sessions_and_recomputed_reference():
    """FP3 hasn't happened yet for anyone (a live, in-progress weekend) -
    per the module docstring, this should fall out of the SAME padding
    mechanism a sprint weekend's missing 3rd session already uses, not a
    special case. n_practice_sessions should drop to 2 for both drivers,
    and practice_reference should be recomputed from only the sessions
    actually present (no FP3 rows to pull a minimum from)."""
    artifacts = _identity_artifacts(constant_pred=5.0, quantile=0.5)
    event_df = _event_features_df(sessions=("FP1", "FP2"))

    results = predict_event(event_df, artifacts)

    assert len(results) == 2
    by_driver = {r.driver: r for r in results}
    expected_pred = 5.0 + 89.5  # min(FP1, FP2) across both drivers = VER's FP2
    for driver in ("VER", "HAM"):
        r = by_driver[driver]
        assert r.n_practice_sessions == 2
        assert r.predicted_quali_time_seconds == pytest.approx(expected_pred)


def test_predict_event_output_is_a_driver_prediction_list():
    artifacts = _identity_artifacts()
    event_df = _event_features_df()
    results = predict_event(event_df, artifacts)
    assert all(isinstance(r, DriverPrediction) for r in results)


def test_load_features_for_event_raises_for_unknown_event(tmp_path):
    features_path = tmp_path / "features.parquet"
    _event_features_df(year=2099, round_number=99).to_parquet(features_path, index=False)

    with pytest.raises(EventNotFoundError):
        load_features_for_event(2099, 1, features_path=features_path)


def test_load_features_for_event_raises_when_file_missing(tmp_path):
    with pytest.raises(ArtifactsNotFoundError):
        load_features_for_event(2099, 99, features_path=tmp_path / "does_not_exist.parquet")


def test_load_features_for_event_returns_only_the_requested_event(tmp_path):
    features_path = tmp_path / "features.parquet"
    df = pd.concat([
        _event_features_df(year=2099, round_number=99),
        _event_features_df(year=2099, round_number=100),
    ], ignore_index=True)
    df.to_parquet(features_path, index=False)

    event = load_features_for_event(2099, 99, features_path=features_path)
    assert (event["RoundNumber"] == 99).all()
    assert len(event) == 6  # 2 drivers x 3 sessions


def _write_fake_artifacts(models_dir, *, quantile=0.752, exact=True, include_era1=True):
    models_dir.mkdir(parents=True, exist_ok=True)
    n = len(FEATURE_COLS)
    model = _zeroed_model()
    torch.save(model.state_dict(), models_dir / "lstm_final_production.pt")
    np.savez(
        models_dir / "final_preprocessing.npz",
        imputer_median=np.zeros(n, dtype=np.float32),
        scaler_mean=np.zeros(n, dtype=np.float32),
        scaler_std=np.ones(n, dtype=np.float32),
        feature_cols=np.array(FEATURE_COLS),
        static_cols=np.array(["IsSprint", "Era"]),
    )
    with open(models_dir / "final_model_metadata.json", "w") as f:
        json.dump({"trained_at_utc": "2026-08-24T21:23:34+00:00", "n_train": 1602, "n_epochs": 8}, f)
    alphas = {}
    if include_era1:
        alphas["0.5"] = {
            "coverage_target_pct": 50,
            "era_1": {"deployment_quantile_seconds": quantile, "deployment_quantile_exact": exact},
        }
    with open(models_dir / "conformal_intervals.json", "w") as f:
        json.dump({"alphas": alphas}, f)


def test_load_production_artifacts_happy_path(tmp_path):
    models_dir = tmp_path / "lstm"
    _write_fake_artifacts(models_dir, quantile=0.752, exact=True)

    artifacts = load_production_artifacts(models_dir=models_dir)

    assert artifacts.feature_cols == FEATURE_COLS
    assert artifacts.deployment_quantile_seconds == pytest.approx(0.752)
    assert artifacts.deployment_quantile_exact is True
    assert artifacts.coverage_target_pct == pytest.approx(50.0)
    assert artifacts.metadata["n_train"] == 1602
    # The zeroed model should actually be usable - forward pass on a tiny
    # made-up batch returns the constant it was built with.
    out = artifacts.model(
        torch.zeros(1, 3, len(FEATURE_COLS)),
        torch.tensor([3]),
        torch.zeros(1, 2),
    )
    assert out.item() == pytest.approx(CONSTANT_GAP_PRED)


def test_load_production_artifacts_raises_when_model_file_missing(tmp_path):
    models_dir = tmp_path / "lstm"
    _write_fake_artifacts(models_dir)
    (models_dir / "lstm_final_production.pt").unlink()

    with pytest.raises(ArtifactsNotFoundError):
        load_production_artifacts(models_dir=models_dir)


def test_load_production_artifacts_raises_when_conformal_missing_era1_entry(tmp_path):
    models_dir = tmp_path / "lstm"
    _write_fake_artifacts(models_dir, include_era1=False)

    with pytest.raises(ArtifactsNotFoundError):
        load_production_artifacts(models_dir=models_dir)


def test_load_production_artifacts_raises_when_dir_is_empty(tmp_path):
    with pytest.raises(ArtifactsNotFoundError):
        load_production_artifacts(models_dir=tmp_path / "nonexistent")


def _event_features_df_with_test_driver(
    real_sessions=("FP1", "FP2", "FP3"), year=2099, round_number=99
) -> pd.DataFrame:
    """VER and HAM run a full weekend (`real_sessions`); a third driver,
    TST, only ever appears in an FP1 row - a reserve/test-driver outing,
    per identify_test_drivers's docstring on the verified real-data rule
    (2+ sessions, or any single non-FP1 session, counts as a real
    entrant; FP1-only does not)."""
    df = _event_features_df(sessions=real_sessions, year=year, round_number=round_number)
    tst_row = {
        "Year": year,
        "RoundNumber": round_number,
        "Driver": "TST",
        "SessionCode": "FP1",
        "IsSprint": False,
        "Era": 1,
        "best_lap_time": 95.0,
        "feat_b": 1.0,
    }
    return pd.concat([df, pd.DataFrame([tst_row])], ignore_index=True)


def test_identify_test_drivers_flags_fp1_only_driver():
    event_df = _event_features_df_with_test_driver()
    assert identify_test_drivers(event_df) == ["TST"]


def test_identify_test_drivers_empty_when_everyone_has_a_real_entry():
    event_df = _event_features_df(sessions=("FP1", "FP2", "FP3"))
    assert identify_test_drivers(event_df) == []


def test_identify_test_drivers_does_not_flag_a_driver_whose_only_session_is_not_fp1():
    """A driver who only ran e.g. SQ (a sprint-quali-only appearance) is
    still a real entrant by the verified rule - only an FP1-only session
    set is excluded."""
    event_df = _event_features_df(sessions=("FP1", "FP2"))
    sq_only_row = {
        "Year": 2099, "RoundNumber": 99, "Driver": "SQO", "SessionCode": "SQ",
        "IsSprint": True, "Era": 1, "best_lap_time": 94.0, "feat_b": 1.0,
    }
    event_df = pd.concat([event_df, pd.DataFrame([sq_only_row])], ignore_index=True)
    assert identify_test_drivers(event_df) == []


def test_predict_event_excludes_fp1_only_test_driver():
    """The Round 11 2026 bug this fixes: a reserve/test driver's FP1-only
    outing must never produce a prediction (there is no qualifying result
    for them to ever be compared against)."""
    artifacts = _identity_artifacts(constant_pred=5.0, quantile=0.5)
    event_df = _event_features_df_with_test_driver()

    results = predict_event(event_df, artifacts)

    assert {r.driver for r in results} == {"VER", "HAM"}
    assert len(results) == 2


def test_predict_event_excluded_test_driver_does_not_affect_practice_reference():
    """TST's slower FP1 lap (95.0s) must not pull the weekend's
    practice_reference away from the real field's actual fastest lap -
    excluding test drivers has to happen BEFORE the reference/pivot step,
    not just at the end of the result list."""
    artifacts = _identity_artifacts(constant_pred=5.0, quantile=0.5)
    event_df = _event_features_df_with_test_driver(real_sessions=("FP1", "FP2", "FP3"))

    results = predict_event(event_df, artifacts)

    expected_pred = 5.0 + 89.0  # VER's FP3, same as the no-test-driver case
    for r in results:
        assert r.predicted_quali_time_seconds == pytest.approx(expected_pred)
