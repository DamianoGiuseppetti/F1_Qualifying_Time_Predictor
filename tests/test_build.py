import pandas as pd
import pytest

from f1qp.features.build import (
    CORE_FEATURE_COLUMNS,
    build_session_driver_features,
    join_telemetry_features,
    join_weather_features,
)


def _two_driver_session() -> pd.DataFrame:
    rows = []
    # VER: one clean long run, best lap on lap 3 (mid-run)
    ver_laps = [
        (1, None, False),
        (2, 91.2, True),
        (3, 90.5, True),
        (4, 91.8, True),
    ]
    for lap, laptime, _flying in ver_laps:
        rows.append({
            "Driver": "VER", "LapNumber": lap, "Stint": 1,
            "LapStartTime": lap * 92, "LapTime": laptime,
            "PitInTime": None, "Deleted": False, "IsAccurate": True,
            "Compound": "SOFT", "TyreLife": lap, "Position": 1,
            "Sector1Time": (laptime / 3) if laptime else None,
            "Sector2Time": (laptime / 3) if laptime else None,
            "Sector3Time": (laptime / 3) if laptime else None,
        })
    # NOR: shorter run, slower
    nor_laps = [(1, None), (2, 92.5)]
    for lap, laptime in nor_laps:
        rows.append({
            "Driver": "NOR", "LapNumber": lap, "Stint": 1,
            "LapStartTime": lap * 93, "LapTime": laptime,
            "PitInTime": None, "Deleted": False, "IsAccurate": True,
            "Compound": "MEDIUM", "TyreLife": lap, "Position": 2,
            "Sector1Time": (laptime / 3) if laptime else None,
            "Sector2Time": (laptime / 3) if laptime else None,
            "Sector3Time": (laptime / 3) if laptime else None,
        })
    df = pd.DataFrame(rows)
    df["LapStartTime"] = pd.to_timedelta(df["LapStartTime"], unit="s")
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        df[col] = df[col].apply(lambda v: pd.to_timedelta(v, unit="s") if pd.notna(v) else pd.NaT)
    return df


def test_build_session_driver_features_basic_shape_and_values():
    laps = _two_driver_session()
    feats = build_session_driver_features(laps, fuel_burn_seconds_per_lap=0.05)
    assert set(feats["Driver"]) == {"VER", "NOR"}
    for col in CORE_FEATURE_COLUMNS:
        assert col in feats.columns

    ver = feats[feats["Driver"] == "VER"].iloc[0]
    assert ver["best_lap_time"] == pytest.approx(90.5)
    assert ver["gap_to_session_best"] == pytest.approx(0.0)  # VER set the session's overall best
    assert ver["n_flying_laps"] == 3
    assert ver["compound_on_best_lap"] == 2  # SOFT

    nor = feats[feats["Driver"] == "NOR"].iloc[0]
    assert nor["gap_to_session_best"] == pytest.approx(92.5 - 90.5)


def test_build_session_driver_features_empty_laps_returns_empty_frame():
    out = build_session_driver_features(pd.DataFrame(), fuel_burn_seconds_per_lap=0.05)
    assert out.empty


def test_build_session_driver_features_no_flying_laps_returns_empty_frame():
    laps = _two_driver_session()
    laps["IsAccurate"] = False
    out = build_session_driver_features(laps, fuel_burn_seconds_per_lap=0.05)
    assert out.empty


def test_build_session_driver_features_missing_tyre_life_falls_back_to_median():
    # VER's best (fastest) lap has a missing TyreLife reading, but another
    # flying lap in the same session has one - the feature should backfill
    # from that median rather than surface a lone NaN in an otherwise
    # fully-populated row (see f1qp.features.build.build_session_driver_features).
    laps = _two_driver_session()
    laps.loc[(laps["Driver"] == "VER") & (laps["LapNumber"] == 3), "TyreLife"] = None
    feats = build_session_driver_features(laps, fuel_burn_seconds_per_lap=0.05)
    ver = feats[feats["Driver"] == "VER"].iloc[0]
    assert ver["best_lap_time"] == pytest.approx(90.5)  # still VER's fastest lap
    assert pd.notna(ver["tyre_life_on_best_lap"])
    assert ver["tyre_life_on_best_lap"] == pytest.approx(3.0)  # median of laps 2 and 4 (TyreLife 2, 4)


def test_build_session_driver_features_long_run_without_flying_laps_falls_back_to_median_pace():
    # SAR's longest run by lap COUNT is all non-flying (deleted) laps; their
    # only flying laps sit in a shorter run. long_run_avg_pace should fall
    # back to the driver's overall median pace instead of NaN.
    rows = []
    long_run_laps = [(1, None, False), (2, 95.0, True), (3, 95.5, True), (4, 96.0, True)]
    for lap, laptime, deleted in long_run_laps:
        rows.append({
            "Driver": "SAR", "LapNumber": lap, "Stint": 1,
            "LapStartTime": lap * 92, "LapTime": laptime,
            "PitInTime": None, "Deleted": deleted, "IsAccurate": True,
            "Compound": "HARD", "TyreLife": lap, "Position": 1,
            "Sector1Time": None, "Sector2Time": None, "Sector3Time": None,
        })
    short_run_laps = [(5, None, False), (6, 90.0, False)]
    for lap, laptime, deleted in short_run_laps:
        rows.append({
            "Driver": "SAR", "LapNumber": lap, "Stint": 2,
            "LapStartTime": lap * 92, "LapTime": laptime,
            "PitInTime": None, "Deleted": deleted, "IsAccurate": True,
            "Compound": "SOFT", "TyreLife": lap, "Position": 1,
            "Sector1Time": None, "Sector2Time": None, "Sector3Time": None,
        })
    df = pd.DataFrame(rows)
    df["LapStartTime"] = pd.to_timedelta(df["LapStartTime"], unit="s")
    df["LapTime"] = df["LapTime"].apply(lambda v: pd.to_timedelta(v, unit="s") if pd.notna(v) else pd.NaT)

    feats = build_session_driver_features(df, fuel_burn_seconds_per_lap=0.05)
    sar = feats[feats["Driver"] == "SAR"].iloc[0]
    assert sar["best_lap_time"] == pytest.approx(90.0)
    assert pd.notna(sar["long_run_avg_pace"])
    assert sar["long_run_avg_pace"] == pytest.approx(sar["median_flying_lap_time"])


def test_join_weather_features_broadcasts_to_every_row():
    feats = pd.DataFrame({"Driver": ["VER", "NOR"], "best_lap_time": [90.5, 92.5]})
    weather_row = {
        "air_temp_mean": 23.5, "track_temp_mean": 38.0, "humidity_mean": 50.0,
        "rainfall_share": 0.0, "wind_speed_mean": 2.0,
    }
    out = join_weather_features(feats, weather_row)
    assert (out["track_temp_mean"] == 38.0).all()


def test_join_weather_features_nan_when_no_data_yet():
    feats = pd.DataFrame({"Driver": ["VER"], "best_lap_time": [90.5]})
    out = join_weather_features(feats, None)
    assert out["track_temp_mean"].isna().all()


def test_join_telemetry_features_matches_on_driver_and_best_lap():
    feats = pd.DataFrame({"Driver": ["VER", "NOR"], "best_lap_number": [3, 2]})
    telemetry = pd.DataFrame({
        "Driver": ["VER", "NOR"],
        "LapNumber": [3, 2],
        "throttle_full_pct": [0.7, 0.65],
        "throttle_mean": [80.0, 75.0],
        "braking_events": [6, 7],
        "avg_speed": [220.0, 210.0],
        "max_speed": [320.0, 310.0],
    })
    out = join_telemetry_features(feats, telemetry)
    ver = out[out["Driver"] == "VER"].iloc[0]
    assert ver["throttle_full_pct"] == pytest.approx(0.7)
    assert "LapNumber" not in out.columns


def test_join_telemetry_features_nan_when_no_data_yet():
    feats = pd.DataFrame({"Driver": ["VER"], "best_lap_number": [3]})
    out = join_telemetry_features(feats, None)
    assert out["avg_speed"].isna().all()


def test_join_telemetry_features_backfills_from_drivers_other_lap():
    # VER's best lap (3) wasn't extracted (a per-lap extraction gap), but VER
    # has telemetry for another lap in the same session - use that rather
    # than leaving these 5 columns NaN in an otherwise fully-populated row.
    feats = pd.DataFrame({"Driver": ["VER"], "best_lap_number": [3]})
    telemetry = pd.DataFrame({
        "Driver": ["VER"],
        "LapNumber": [5],
        "throttle_full_pct": [0.7], "throttle_mean": [80.0],
        "braking_events": [6], "avg_speed": [220.0], "max_speed": [320.0],
    })
    out = join_telemetry_features(feats, telemetry)
    ver = out.iloc[0]
    assert ver["avg_speed"] == pytest.approx(220.0)
    assert ver["throttle_full_pct"] == pytest.approx(0.7)


def test_join_telemetry_features_backfills_from_session_mean_when_driver_has_none():
    # NOR has no telemetry at all for this session (extraction failed for
    # every one of their laps), but the session has telemetry for other
    # drivers - fall back to the session-wide mean rather than NaN.
    feats = pd.DataFrame({"Driver": ["VER", "NOR"], "best_lap_number": [3, 2]})
    telemetry = pd.DataFrame({
        "Driver": ["VER"],
        "LapNumber": [3],
        "throttle_full_pct": [0.7], "throttle_mean": [80.0],
        "braking_events": [6], "avg_speed": [220.0], "max_speed": [320.0],
    })
    out = join_telemetry_features(feats, telemetry)
    nor = out[out["Driver"] == "NOR"].iloc[0]
    assert nor["avg_speed"] == pytest.approx(220.0)  # session mean == VER's only value
