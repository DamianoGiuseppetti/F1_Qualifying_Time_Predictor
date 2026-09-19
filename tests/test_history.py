"""Tests for f1qp.serving.history - the launch/scoring persistence layer
behind the Prediction tab's switcher and the History tab (Phase 5 UI
redesign; see the module's own docstring for the full "why").

Uses `tmp_path` for both `launches_dir` and a small synthetic
`qualifying_targets.parquet` in every test - never touches the real
`data/predictions/launches/` or `data/processed/qualifying_targets.parquet`,
same isolation convention tests/test_predict.py and tests/test_api.py
already use for their own tmp fixtures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1qp.serving.history import (
    is_scored,
    latest_two_launches,
    list_launches,
    prediction_history,
    record_launch,
    score_launch,
)
from f1qp.serving.predict import DriverPrediction


def _pred(driver, predicted, low, high, era=1, is_sprint=False, n_sessions=3) -> DriverPrediction:
    return DriverPrediction(
        driver=driver,
        era=era,
        is_sprint=is_sprint,
        n_practice_sessions=n_sessions,
        predicted_quali_time_seconds=predicted,
        interval_low_seconds=low,
        interval_high_seconds=high,
        interval_width_seconds=high - low,
        interval_level_pct=50.0,
        interval_exact=True,
    )


def _write_targets(path, rows):
    """rows: list of (year, round_number, driver, q1, q2, q3)."""
    df = pd.DataFrame(rows, columns=["Year", "RoundNumber", "Driver", "Q1", "Q2", "Q3"])
    df.to_parquet(path, index=False)


def test_record_launch_round_trips_through_list_launches(tmp_path):
    launches_dir = tmp_path / "launches"
    preds = [_pred("VER", 90.0, 89.5, 90.5), _pred("HAM", 91.0, 90.5, 91.5)]

    record_launch(
        year=2026, round_number=13, predictions=preds, excluded_test_drivers=["ARO"],
        model_trained_at_utc="2026-08-25T21:20:21+00:00", launched_at_utc="2026-08-26T10:00:00+00:00",
        launches_dir=launches_dir,
    )

    records = list_launches(launches_dir)
    assert len(records) == 1
    r = records[0]
    assert r.year == 2026
    assert r.round_number == 13
    assert r.excluded_test_drivers == ["ARO"]
    assert len(r.predictions) == 2
    assert {p["driver"] for p in r.predictions} == {"VER", "HAM"}


def test_relaunching_same_round_overwrites_not_duplicates(tmp_path):
    launches_dir = tmp_path / "launches"
    record_launch(
        year=2026, round_number=13, predictions=[_pred("VER", 90.0, 89.5, 90.5)],
        excluded_test_drivers=[], model_trained_at_utc="t0", launched_at_utc="2026-08-26T09:00:00+00:00",
        launches_dir=launches_dir,
    )
    record_launch(
        year=2026, round_number=13, predictions=[_pred("VER", 91.0, 90.5, 91.5)],
        excluded_test_drivers=[], model_trained_at_utc="t1", launched_at_utc="2026-08-26T10:00:00+00:00",
        launches_dir=launches_dir,
    )

    records = list_launches(launches_dir)
    assert len(records) == 1
    assert records[0].launched_at_utc == "2026-08-26T10:00:00+00:00"
    assert records[0].predictions[0]["predicted_quali_time_seconds"] == 91.0


def test_list_launches_empty_dir_returns_empty_list_not_error(tmp_path):
    assert list_launches(tmp_path / "does_not_exist") == []


def test_score_launch_unscored_round_has_no_official_result(tmp_path):
    launches_dir = tmp_path / "launches"
    targets_path = tmp_path / "qualifying_targets.parquet"
    _write_targets(targets_path, [])  # no rows for this round yet

    record_launch(
        year=2026, round_number=13, predictions=[_pred("VER", 90.0, 89.5, 90.5)],
        excluded_test_drivers=[], model_trained_at_utc="t0", launched_at_utc="2026-08-26T10:00:00+00:00",
        launches_dir=launches_dir,
    )
    record = list_launches(launches_dir)[0]
    scored_df = score_launch(record, targets_path=targets_path)

    assert len(scored_df) == 1
    assert scored_df.loc[0, "has_target"] == False  # noqa: E712
    assert pd.isna(scored_df.loc[0, "final_quali_time"])
    assert not is_scored(record, targets_path=targets_path)


def test_score_launch_computes_abs_error_and_within_interval(tmp_path):
    launches_dir = tmp_path / "launches"
    targets_path = tmp_path / "qualifying_targets.parquet"
    # VER's official time (Q3) lands inside [89.5, 90.5]; HAM's lands outside.
    _write_targets(targets_path, [
        (2026, 13, "VER", 91.0, 90.2, 90.1),
        (2026, 13, "HAM", 92.0, 91.5, 93.0),
    ])

    record_launch(
        year=2026, round_number=13,
        predictions=[_pred("VER", 90.0, 89.5, 90.5), _pred("HAM", 91.0, 90.5, 91.5)],
        excluded_test_drivers=[], model_trained_at_utc="t0", launched_at_utc="2026-08-26T10:00:00+00:00",
        launches_dir=launches_dir,
    )
    record = list_launches(launches_dir)[0]
    scored_df = score_launch(record, targets_path=targets_path).set_index("driver")

    assert scored_df.loc["VER", "has_target"] == True  # noqa: E712
    assert scored_df.loc["VER", "final_quali_time"] == pytest.approx(90.1)
    assert scored_df.loc["VER", "abs_error_seconds"] == pytest.approx(0.1)
    assert scored_df.loc["VER", "within_interval"] == True  # noqa: E712

    assert scored_df.loc["HAM", "final_quali_time"] == pytest.approx(93.0)
    assert scored_df.loc["HAM", "within_interval"] == False  # noqa: E712

    assert is_scored(record, targets_path=targets_path)


def test_is_scored_false_when_only_some_drivers_have_a_result(tmp_path):
    launches_dir = tmp_path / "launches"
    targets_path = tmp_path / "qualifying_targets.parquet"
    _write_targets(targets_path, [(2026, 13, "VER", 91.0, 90.2, 90.1)])  # HAM missing

    record_launch(
        year=2026, round_number=13,
        predictions=[_pred("VER", 90.0, 89.5, 90.5), _pred("HAM", 91.0, 90.5, 91.5)],
        excluded_test_drivers=[], model_trained_at_utc="t0", launched_at_utc="2026-08-26T10:00:00+00:00",
        launches_dir=launches_dir,
    )
    record = list_launches(launches_dir)[0]
    assert not is_scored(record, targets_path=targets_path)


def test_latest_two_launches_orders_newest_first_and_caps_at_two(tmp_path):
    launches_dir = tmp_path / "launches"
    targets_path = tmp_path / "qualifying_targets.parquet"
    _write_targets(targets_path, [])

    for round_number, ts in [(11, "2026-08-01T00:00:00+00:00"), (12, "2026-08-15T00:00:00+00:00"), (13, "2026-08-26T00:00:00+00:00")]:
        record_launch(
            year=2026, round_number=round_number, predictions=[_pred("VER", 90.0, 89.5, 90.5)],
            excluded_test_drivers=[], model_trained_at_utc="t0", launched_at_utc=ts,
            launches_dir=launches_dir,
        )

    current = latest_two_launches(launches_dir=launches_dir, targets_path=targets_path)
    assert [c["round_number"] for c in current] == [13, 12]


def test_prediction_history_filters_by_season_and_round(tmp_path):
    launches_dir = tmp_path / "launches"
    targets_path = tmp_path / "qualifying_targets.parquet"
    _write_targets(targets_path, [])

    record_launch(
        year=2026, round_number=12, predictions=[_pred("VER", 90.0, 89.5, 90.5)],
        excluded_test_drivers=[], model_trained_at_utc="t0", launched_at_utc="2026-08-15T00:00:00+00:00",
        launches_dir=launches_dir,
    )
    record_launch(
        year=2026, round_number=13, predictions=[_pred("VER", 91.0, 90.5, 91.5)],
        excluded_test_drivers=[], model_trained_at_utc="t0", launched_at_utc="2026-08-26T00:00:00+00:00",
        launches_dir=launches_dir,
    )

    all_rows = prediction_history(launches_dir=launches_dir, targets_path=targets_path)
    assert set(all_rows["round_number"]) == {12, 13}

    only_13 = prediction_history(season=2026, round_number=13, launches_dir=launches_dir, targets_path=targets_path)
    assert set(only_13["round_number"]) == {13}


def test_prediction_history_empty_when_no_launches(tmp_path):
    df = prediction_history(launches_dir=tmp_path / "empty")
    assert df.empty


def test_list_launches_excludes_pre_min_season_launches(tmp_path):
    """A launch from before MIN_SEASON (2026) - there was exactly one on
    Damiano's real data/predictions/launches/, a leftover test launch
    from before this filter existed, cleaned up Aug 30 2026 alongside
    adding it - must never surface, in list_launches or anything built on
    top of it. "history only for 2026 i don't want to go back more than
    this" (Damiano, Aug 30 2026)."""
    launches_dir = tmp_path / "launches"
    targets_path = tmp_path / "qualifying_targets.parquet"
    _write_targets(targets_path, [])
    record_launch(
        year=2023, round_number=8, predictions=[_pred("VER", 90.0, 89.5, 90.5)],
        excluded_test_drivers=[], model_trained_at_utc="t0", launched_at_utc="2023-05-01T00:00:00+00:00",
        launches_dir=launches_dir,
    )
    record_launch(
        year=2026, round_number=13, predictions=[_pred("VER", 91.0, 90.5, 91.5)],
        excluded_test_drivers=[], model_trained_at_utc="t1", launched_at_utc="2026-08-26T00:00:00+00:00",
        launches_dir=launches_dir,
    )

    records = list_launches(launches_dir)
    assert [r.year for r in records] == [2026]

    current = latest_two_launches(launches_dir=launches_dir, targets_path=targets_path)
    assert [c["year"] for c in current] == [2026]

    all_rows = prediction_history(launches_dir=launches_dir, targets_path=targets_path)
    assert set(all_rows["year"]) == {2026}
