import pandas as pd

from f1qp.data.schema import validate_laps


def test_validate_laps_flags_missing_columns():
    df = pd.DataFrame({"Driver": ["VER", "NOR"], "LapTime": [None, None]})
    report = validate_laps(df, year=2026, round_number=1, session_code="FP1")
    assert not report.is_clean
    assert "LapNumber" in report.missing_columns


def test_validate_laps_clean_frame():
    cols = [
        "Driver", "LapTime", "LapNumber", "Stint", "Compound", "TyreLife",
        "Sector1Time", "Sector2Time", "Sector3Time", "IsPersonalBest", "Deleted",
    ]
    df = pd.DataFrame({c: [1, 2] for c in cols})
    df["LapTime"] = pd.to_timedelta(["0:01:30", "0:01:31"])
    report = validate_laps(df, year=2026, round_number=1, session_code="FP1")
    assert report.is_clean


def test_validate_laps_flags_thin_qualifying_field():
    df = pd.DataFrame({"Driver": [f"D{i}" for i in range(8)], "LapTime": [1] * 8})
    report = validate_laps(df, year=2026, round_number=12, session_code="Q")
    assert any("red flag" in n for n in report.notes)


def _full_column_frame(n_rows: int, n_null_laptime: int) -> pd.DataFrame:
    cols = [
        "Driver", "LapNumber", "Stint", "Compound", "TyreLife",
        "Sector1Time", "Sector2Time", "Sector3Time", "IsPersonalBest", "Deleted",
    ]
    df = pd.DataFrame({c: [1] * n_rows for c in cols})
    df["Driver"] = [f"D{i % 20}" for i in range(n_rows)]
    times = pd.to_timedelta(["0:01:30"] * n_rows)
    df["LapTime"] = times.where(pd.Series(range(n_rows)) >= n_null_laptime, pd.NaT)
    return df


def test_fp_session_with_30pct_null_laptime_is_clean():
    # Out-laps / in-laps routinely leave ~30% of FP laps untimed - this is
    # normal and should NOT be flagged.
    df = _full_column_frame(n_rows=20, n_null_laptime=6)  # 30% null
    report = validate_laps(df, year=2023, round_number=13, session_code="FP2")
    assert report.is_clean


def test_qualifying_session_with_40pct_null_laptime_is_clean():
    # Confirmed against real downloaded data: every push lap in Q brackets
    # an out-lap and an in-lap, so 35-43% null is the NORMAL range, not an
    # anomaly - the download-time check only exists to catch something
    # obviously dead, not this.
    df = _full_column_frame(n_rows=20, n_null_laptime=8)  # 40% null
    report = validate_laps(df, year=2023, round_number=13, session_code="Q")
    assert report.is_clean


def test_qualifying_session_with_65pct_null_laptime_is_flagged():
    # Well past even the real Baku-2025 outlier (60.9%) - this should still
    # trip the coarse download-time ceiling.
    df = _full_column_frame(n_rows=20, n_null_laptime=13)  # 65% null
    report = validate_laps(df, year=2023, round_number=13, session_code="Q")
    assert not report.is_clean
