import pandas as pd
import pytest

from f1qp.features.runs import add_run_features, identify_runs, run_summary


def _laps(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal laps frame with the columns identify_runs needs.

    Each row dict may omit columns - sensible defaults fill in, and
    LapStartTime/LapTime accept plain seconds (int/float) for readability.
    """
    df = pd.DataFrame(rows)
    df["Driver"] = df["Driver"] if "Driver" in df.columns else "VER"
    for col in ["PitInTime", "PitOutTime"]:
        if col not in df.columns:
            df[col] = pd.NaT
        else:
            df[col] = df[col].apply(lambda v: pd.to_timedelta(v, unit="s") if pd.notna(v) else pd.NaT)
    if "Deleted" not in df.columns:
        df["Deleted"] = False
    else:
        df["Deleted"] = df["Deleted"].map(lambda v: v if pd.notna(v) else False)
    if "IsAccurate" not in df.columns:
        df["IsAccurate"] = True
    else:
        df["IsAccurate"] = df["IsAccurate"].astype("object").where(df["IsAccurate"].notna(), True)
    if "Compound" not in df.columns:
        df["Compound"] = "MEDIUM"
    else:
        df["Compound"] = df["Compound"].fillna("MEDIUM")
    df["LapStartTime"] = pd.to_timedelta(df["LapStartTime"], unit="s")
    df["LapTime"] = df["LapTime"].apply(lambda v: pd.to_timedelta(v, unit="s") if pd.notna(v) else pd.NaT)
    return df


def test_stint_change_always_starts_a_new_run():
    laps = _laps([
        {"LapNumber": 1, "Stint": 1, "LapStartTime": 0, "LapTime": None},
        {"LapNumber": 2, "Stint": 1, "LapStartTime": 90, "LapTime": 90},
        {"LapNumber": 3, "Stint": 2, "LapStartTime": 300, "LapTime": None},
        {"LapNumber": 4, "Stint": 2, "LapStartTime": 390, "LapTime": 90},
    ])
    out = identify_runs(laps)
    assert list(out.sort_values("LapNumber")["RunId"]) == [1, 1, 2, 2]


def test_contiguous_stint_stays_one_run_real_data_shape():
    # Mirrors the real Belgian GP 2026 FP2 shape used to validate this module:
    # 6 back-to-back laps in one stint, no gaps - must stay a single run.
    rows = []
    t = 0
    for lap in range(1, 7):
        rows.append({"LapNumber": lap, "Stint": 3, "LapStartTime": t, "LapTime": 111})
        t += 111
    laps = _laps(rows)
    out = identify_runs(laps)
    assert out["RunId"].nunique() == 1


def test_long_gap_within_same_stint_splits_the_run():
    # Same stint throughout (no pit stop), but a ~10 minute gap mid-stint -
    # a red flag or blocked track, not a tyre change.
    laps = _laps([
        {"LapNumber": 1, "Stint": 1, "LapStartTime": 0, "LapTime": None},
        {"LapNumber": 2, "Stint": 1, "LapStartTime": 90, "LapTime": 90},
        {"LapNumber": 3, "Stint": 1, "LapStartTime": 780, "LapTime": None},  # +690s gap
        {"LapNumber": 4, "Stint": 1, "LapStartTime": 870, "LapTime": 90},
    ])
    out = identify_runs(laps)
    assert list(out.sort_values("LapNumber")["RunId"]) == [1, 1, 2, 2]


def test_ordinary_out_lap_gap_does_not_split_the_run():
    # An out-lap is slower than a flying lap but not a red-flag-scale gap -
    # should not spuriously split the run.
    laps = _laps([
        {"LapNumber": 1, "Stint": 1, "LapStartTime": 0, "LapTime": None},
        {"LapNumber": 2, "Stint": 1, "LapStartTime": 130, "LapTime": 90},  # slow out-lap, ~130s
        {"LapNumber": 3, "Stint": 1, "LapStartTime": 220, "LapTime": 90},
    ])
    out = identify_runs(laps)
    assert out["RunId"].nunique() == 1


def test_pit_exit_out_lap_gap_does_not_split_the_run_real_data_shape():
    # Real Belgian GP 2026 FP2 (VER, stint 3): LapStartTime for the out-lap
    # sits ~428s before the next lap's start (garage dwell time before the
    # car actually leaves the pits), well past a naive gap threshold - but
    # the out-lap's LapTime is null, so this must NOT split the run.
    laps = _laps([
        {"LapNumber": 9, "Stint": 3, "LapStartTime": 2277.83, "LapTime": None},
        {"LapNumber": 10, "Stint": 3, "LapStartTime": 2705.82, "LapTime": 111.405},
        {"LapNumber": 11, "Stint": 3, "LapStartTime": 2817.23, "LapTime": 112.212},
    ])
    out = identify_runs(laps)
    assert out["RunId"].nunique() == 1


def test_identify_runs_requires_expected_columns():
    with pytest.raises(ValueError):
        identify_runs(pd.DataFrame({"Driver": ["VER"]}))


def test_flying_lap_excludes_out_lap_in_lap_deleted_and_inaccurate():
    laps = _laps([
        {"LapNumber": 1, "Stint": 1, "LapStartTime": 0, "LapTime": None},  # out-lap
        {"LapNumber": 2, "Stint": 1, "LapStartTime": 90, "LapTime": 90},  # flying
        {"LapNumber": 3, "Stint": 1, "LapStartTime": 180, "LapTime": 91, "Deleted": True},  # deleted
        {"LapNumber": 4, "Stint": 1, "LapStartTime": 271, "LapTime": 95, "IsAccurate": False},  # inaccurate
        {"LapNumber": 5, "Stint": 1, "LapStartTime": 366, "LapTime": 120, "PitInTime": 366 + 120},  # in-lap
    ])
    out = add_run_features(laps)
    flying = out.set_index("LapNumber")["IsFlyingLap"]
    assert flying.loc[2] is True or flying.loc[2] == True  # noqa: E712
    for lap in (1, 3, 4, 5):
        assert not flying.loc[lap]


def test_run_summary_counts_laps_and_flying_laps_per_run():
    laps = _laps([
        {"LapNumber": 1, "Stint": 1, "LapStartTime": 0, "LapTime": None},
        {"LapNumber": 2, "Stint": 1, "LapStartTime": 90, "LapTime": 90},
        {"LapNumber": 3, "Stint": 1, "LapStartTime": 180, "LapTime": 91},
        {"LapNumber": 4, "Stint": 1, "LapStartTime": 271, "LapTime": 120, "PitInTime": 271 + 120},
    ])
    out = add_run_features(laps)
    summary = run_summary(out)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_laps"] == 4
    assert row["n_flying_laps"] == 2  # laps 2 and 3; 1 is out-lap, 4 is in-lap
