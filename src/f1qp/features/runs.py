"""Run identification: cluster laps into runs and flag flying laps.

A "run" is a stretch of consecutive laps on one set of tyres with no break
long enough to change the driving context. Two independent boundary rules,
either one starts a new run - see docs/feature_engineering.md for the full
reasoning:

  1. `Stint` changes - a real pit stop, already present in the raw data.
  2. A time-gap break within the same stint, anchored on the PREVIOUS lap
     only when that previous lap has a real (non-null) `LapTime`: the
     elapsed time between one lap's start and the previous lap's start is
     more than `gap_multiplier` times the session's median lap time. This
     catches a red flag or a long blocked-track pause that doesn't show up
     as a stint change (the car stays out on the same tyres but sits
     stationary for minutes). The previous-lap-timed restriction matters -
     an out-lap's `LapStartTime` can sit many minutes before its
     `PitOutTime` (real FP data shows cars idling in the garage between
     "lap start" and actually leaving the pits), and an out-lap's
     `LapTime` is always null, so without this restriction every pit-exit
     out-lap would spuriously split itself into its own one-lap "run".

A lap is a **flying lap** only if it is not the first lap of its run (an
out-lap), not immediately followed by a pit-in on the same lap
(`PitInTime` set - an in-lap), not `Deleted`, and `IsAccurate`. This
mirrors the out-lap/in-lap reasoning already validated against real data in
Phase 1 (see docs/data_strategy.md's null-LapTime investigation), applied
per lap instead of as a session-level aggregate.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_GAP_MULTIPLIER = 2.5
# Absolute floor for the gap threshold so a session with an unusually low
# median lap time (e.g. a street circuit) doesn't trip on perfectly normal
# out-lap/in-lap pace alone.
MIN_GAP_SECONDS = 90.0

REQUIRED_COLUMNS = [
    "Driver", "LapNumber", "Stint", "LapTime", "LapStartTime", "PitInTime", "Deleted", "IsAccurate",
]


def _session_median_lap_seconds(laps: pd.DataFrame) -> float:
    valid = laps["LapTime"].dropna()
    if valid.empty:
        return MIN_GAP_SECONDS / DEFAULT_GAP_MULTIPLIER  # arbitrary but harmless fallback
    return valid.dt.total_seconds().median()


def identify_runs(laps: pd.DataFrame, *, gap_multiplier: float = DEFAULT_GAP_MULTIPLIER) -> pd.DataFrame:
    """Return a copy of `laps` with a new integer `RunId` column.

    `RunId` is unique within (Driver) but not across drivers or sessions -
    join on Driver (and Year/RoundNumber/SessionCode if present) alongside
    it, don't treat it as a global identifier.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in laps.columns]
    if missing:
        raise ValueError(f"identify_runs is missing required columns: {missing}")

    median_lap_seconds = _session_median_lap_seconds(laps)
    gap_threshold_seconds = max(gap_multiplier * median_lap_seconds, MIN_GAP_SECONDS)

    laps = laps.sort_values(["Driver", "LapNumber"]).copy()
    run_ids = pd.Series(index=laps.index, dtype="int64")

    for driver, idx in laps.groupby("Driver", sort=False).groups.items():
        grp = laps.loc[idx]
        run_id = 0
        prev_stint = None
        prev_start = None
        prev_lap_timed = False
        ids = []
        for _, row in grp.iterrows():
            is_new_run = prev_stint is None or row["Stint"] != prev_stint
            if (
                not is_new_run
                and prev_lap_timed  # the anchor lap must be a real timed lap - see module docstring
                and prev_start is not None
                and pd.notna(row["LapStartTime"])
            ):
                gap_seconds = (row["LapStartTime"] - prev_start).total_seconds()
                if gap_seconds > gap_threshold_seconds:
                    is_new_run = True
            if is_new_run:
                run_id += 1
            ids.append(run_id)
            prev_stint = row["Stint"]
            prev_lap_timed = pd.notna(row["LapTime"])
            if pd.notna(row["LapStartTime"]):
                prev_start = row["LapStartTime"]
        run_ids.loc[idx] = ids

    laps["RunId"] = run_ids
    return laps


def flag_flying_laps(laps_with_runs: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean `IsFlyingLap` column. Requires `RunId` from `identify_runs` first."""
    if "RunId" not in laps_with_runs.columns:
        raise ValueError("flag_flying_laps requires RunId - call identify_runs first")

    laps = laps_with_runs.copy()
    first_lap_of_run = laps.groupby(["Driver", "RunId"])["LapNumber"].transform("min")
    is_out_lap = laps["LapNumber"] == first_lap_of_run
    is_in_lap = laps["PitInTime"].notna()
    is_deleted = laps["Deleted"].map(lambda v: bool(v) if pd.notna(v) else False)
    if "IsAccurate" in laps.columns:
        is_accurate = laps["IsAccurate"].map(lambda v: bool(v) if pd.notna(v) else False)
    else:
        is_accurate = True
    has_time = laps["LapTime"].notna()

    laps["IsFlyingLap"] = (~is_out_lap) & (~is_in_lap) & (~is_deleted) & is_accurate & has_time
    return laps


def add_run_features(laps: pd.DataFrame, *, gap_multiplier: float = DEFAULT_GAP_MULTIPLIER) -> pd.DataFrame:
    """Convenience wrapper: identify_runs + flag_flying_laps in one call."""
    return flag_flying_laps(identify_runs(laps, gap_multiplier=gap_multiplier))


def run_summary(laps_with_runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (Driver, RunId): run length and flying-lap count.

    Requires `add_run_features` to have been run first (needs `IsFlyingLap`).
    """
    if "IsFlyingLap" not in laps_with_runs.columns:
        raise ValueError("run_summary requires IsFlyingLap - call add_run_features first")

    return (
        laps_with_runs.groupby(["Driver", "RunId"])
        .agg(
            n_laps=("LapNumber", "count"),
            n_flying_laps=("IsFlyingLap", "sum"),
            compound=("Compound", "first"),
            start_lap=("LapNumber", "min"),
            end_lap=("LapNumber", "max"),
        )
        .reset_index()
    )
