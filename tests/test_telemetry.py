import pandas as pd
import pytest

from f1qp.features.runs import add_run_features
from f1qp.features.telemetry import (
    representative_laps,
    telemetry_row_for_lap,
    telemetry_trend_features,
)


def _telemetry(throttle: list[float], brake: list[bool], speed: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Throttle": throttle, "Brake": brake, "Speed": speed})


def test_telemetry_trend_features_full_throttle_straight():
    tel = _telemetry(
        throttle=[100.0] * 10,
        brake=[False] * 10,
        speed=[300.0] * 10,
    )
    feats = telemetry_trend_features(tel)
    assert feats["throttle_full_pct"] == pytest.approx(1.0)
    assert feats["braking_events"] == 0
    assert feats["max_speed"] == pytest.approx(300.0)


def test_telemetry_trend_features_counts_brake_applications_not_samples():
    # Two separate braking zones, each held for 3 samples - must count as 2 events, not 6.
    tel = _telemetry(
        throttle=[100, 0, 0, 0, 100, 100, 0, 0, 0, 100],
        brake=[False, True, True, True, False, False, True, True, True, False],
        speed=[300, 200, 150, 120, 150, 300, 200, 150, 120, 150],
    )
    feats = telemetry_trend_features(tel)
    assert feats["braking_events"] == 2


def test_telemetry_trend_features_raises_on_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        telemetry_trend_features(pd.DataFrame({"Throttle": [100.0]}))


def test_telemetry_trend_features_raises_on_empty_frame():
    empty = pd.DataFrame({c: [] for c in ["Throttle", "Brake", "Speed"]})
    with pytest.raises(ValueError, match="empty"):
        telemetry_trend_features(empty)


def test_telemetry_row_for_lap_includes_join_keys():
    tel = _telemetry(throttle=[100.0] * 5, brake=[False] * 5, speed=[300.0] * 5)
    row = telemetry_row_for_lap(
        tel, year=2026, round_number=10, session_code="FP2", driver="VER", lap_number=14.0
    )
    assert row["Year"] == 2026
    assert row["Driver"] == "VER"
    assert row["LapNumber"] == 14.0
    assert "avg_speed" in row


def _laps_for_representative_test() -> pd.DataFrame:
    df = pd.DataFrame({
        "Driver": ["VER"] * 4,
        "LapNumber": [10, 11, 12, 13],
        "Stint": [3, 3, 3, 3],
        "LapStartTime": pd.to_timedelta([2705.82, 2817.23, 2929.44, 3041.09], unit="s"),
        "LapTime": pd.to_timedelta([111.405, 112.212, 109.5, 111.523], unit="s"),
        "PitInTime": pd.NaT,
        "Deleted": False,
        "IsAccurate": True,
        "Compound": "MEDIUM",
    })
    return df


def test_representative_laps_picks_fastest_flying_lap_per_run():
    laps = add_run_features(_laps_for_representative_test())
    rep = representative_laps(laps)
    assert len(rep) == 1
    assert rep.iloc[0]["LapNumber"] == 12  # 109.5s is the fastest


def test_representative_laps_requires_expected_columns():
    with pytest.raises(ValueError):
        representative_laps(pd.DataFrame({"Driver": ["VER"]}))


def test_representative_laps_empty_when_no_flying_laps():
    df = _laps_for_representative_test()
    df["IsAccurate"] = False  # nothing qualifies as flying
    laps = add_run_features(df)
    rep = representative_laps(laps)
    assert rep.empty
