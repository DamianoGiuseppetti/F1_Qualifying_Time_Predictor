"""Telemetry trend features, computed for a small representative-lap subset only.

Phase 1 deliberately never pulled car telemetry (`telemetry=False`
everywhere in `f1qp.data.loader`) - full car telemetry is the single
biggest driver of download time/disk usage across 82+ weekends, and only
this feature needs it. `representative_laps` picks the one lap per run
that's worth the telemetry pull (the run's fastest flying lap) instead of
every lap of every session; `scripts/extract_telemetry.py` is what
actually fetches telemetry for those laps and calls
`telemetry_row_for_lap` per lap - this module has no network dependency of
its own, only pandas.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_TELEMETRY_COLUMNS = ["Throttle", "Brake", "Speed"]
FULL_THROTTLE_THRESHOLD = 99.0  # Throttle is 0-100; treat >=99 as "flat out"


def representative_laps(laps_with_runs: pd.DataFrame) -> pd.DataFrame:
    """The fastest flying lap of every (Driver, RunId) - the only laps worth pulling telemetry for.

    Requires `IsFlyingLap` and `RunId` - run `f1qp.features.runs.add_run_features` first.
    """
    required = ["Driver", "RunId", "IsFlyingLap", "LapTime"]
    missing = [c for c in required if c not in laps_with_runs.columns]
    if missing:
        raise ValueError(f"representative_laps is missing required columns: {missing}")

    flying = laps_with_runs[laps_with_runs["IsFlyingLap"]]
    if flying.empty:
        return flying.copy()
    idx = flying.groupby(["Driver", "RunId"])["LapTime"].idxmin()
    return laps_with_runs.loc[idx].copy()


def _count_braking_events(brake: pd.Series) -> int:
    """Number of times the brake channel transitions from off to on (brake applications, not samples)."""
    brake_bool = brake.astype(bool)
    onsets = brake_bool & ~brake_bool.shift(1, fill_value=False)
    return int(onsets.sum())


def telemetry_trend_features(telemetry: pd.DataFrame) -> dict:
    """Aggregate one lap's car telemetry channels into a few driving-style features.

    Raises ValueError on missing columns or an empty frame.
    """
    missing = [c for c in REQUIRED_TELEMETRY_COLUMNS if c not in telemetry.columns]
    if missing:
        raise ValueError(f"telemetry_trend_features is missing required columns: {missing}")
    if telemetry.empty:
        raise ValueError("telemetry_trend_features got an empty telemetry frame")

    return {
        "throttle_full_pct": float((telemetry["Throttle"] >= FULL_THROTTLE_THRESHOLD).mean()),
        "throttle_mean": float(telemetry["Throttle"].mean()),
        "braking_events": _count_braking_events(telemetry["Brake"]),
        "avg_speed": float(telemetry["Speed"].mean()),
        "max_speed": float(telemetry["Speed"].max()),
    }


def telemetry_row_for_lap(
    telemetry: pd.DataFrame,
    *,
    year: int,
    round_number: int,
    session_code: str,
    driver: str,
    lap_number: float,
) -> dict:
    """`telemetry_trend_features` plus the join keys, ready to append into a rows list."""
    row = {
        "Year": year,
        "RoundNumber": round_number,
        "SessionCode": session_code,
        "Driver": driver,
        "LapNumber": lap_number,
    }
    row.update(telemetry_trend_features(telemetry))
    return row
