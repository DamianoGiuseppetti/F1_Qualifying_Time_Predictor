"""Assemble one feature row per (Driver, Session) - the practice-session step
vector described in docs/feature_engineering.md.

`build_session_driver_features` only needs laps (already on disk from
Phase 1) plus a fuel-burn factor from `f1qp.features.fuel`. Weather and
telemetry are joined in as separate steps (`join_weather_features`,
`join_telemetry_features`) because neither is available until
`scripts/extract_weather.py` / `scripts/extract_telemetry.py` have run -
see docs/feature_engineering.md for why Phase 1 never saved them. Building
the core laps-only columns never blocks on those two scripts having run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from f1qp.features.fuel import fuel_corrected_pace
from f1qp.features.runs import add_run_features, run_summary

COMPOUND_ORDER = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}
DEFAULT_COMPOUND_ORD = 1

CORE_FEATURE_COLUMNS = [
    "best_lap_time", "gap_to_session_best", "median_flying_lap_time", "lap_time_std",
    "n_runs", "n_flying_laps", "avg_run_length", "longest_run_length",
    "compound_on_best_lap", "tyre_life_on_best_lap", "fuel_corrected_pace", "long_run_avg_pace",
    "best_sector1_time", "best_sector2_time", "best_sector3_time", "best_lap_session_position",
]
WEATHER_FEATURE_COLUMNS = ["air_temp_mean", "track_temp_mean", "humidity_mean", "rainfall_share", "wind_speed_mean"]
TELEMETRY_FEATURE_COLUMNS = ["throttle_full_pct", "throttle_mean", "braking_events", "avg_speed", "max_speed"]


def _best_sector_seconds(flying_laps: pd.DataFrame, column: str) -> float:
    if column not in flying_laps.columns:
        return np.nan
    valid = flying_laps[column].dropna()
    if valid.empty:
        return np.nan
    return float(valid.min().total_seconds())


def build_session_driver_features(
    laps: pd.DataFrame, *, fuel_burn_seconds_per_lap: float, gap_multiplier: float = 2.5
) -> pd.DataFrame:
    """One row per Driver, from one session's laps (already filtered to a single SessionCode).

    Returns an empty DataFrame (not an error) if there are no laps or no
    flying laps at all - a session with nothing usable is a real, expected
    case (e.g. fully red-flagged), not a bug to raise on.
    """
    if laps.empty:
        return pd.DataFrame(columns=["Driver", *CORE_FEATURE_COLUMNS, "best_lap_number"])

    laps = add_run_features(laps, gap_multiplier=gap_multiplier)
    flying = laps[laps["IsFlyingLap"]].copy()
    if flying.empty:
        return pd.DataFrame(columns=["Driver", *CORE_FEATURE_COLUMNS, "best_lap_number"])

    flying["lap_time_seconds"] = flying["LapTime"].dt.total_seconds()
    session_best = flying["lap_time_seconds"].min()
    session_start = laps["LapStartTime"].min()
    session_span = (laps["LapStartTime"].max() - session_start).total_seconds()
    summaries = run_summary(laps)

    rows = []
    for driver, grp in flying.groupby("Driver"):
        grp = grp.sort_values("lap_time_seconds")
        best = grp.iloc[0]
        driver_runs = summaries[summaries["Driver"] == driver]

        # Rank of the best lap among this run's FLYING laps only - matches how
        # f1qp.features.fuel ranks lap_in_run when fitting the coefficient
        # (also pre-filtered to flying laps). It additionally restricts to
        # green-flag laps when fitting, which this rank does not re-apply, so
        # this is an approximate within-run position, not an exact replay of
        # the fit - fine for a correction that isn't claiming exact physical
        # fuel load in the first place (see f1qp.features.fuel docstring).
        run_laps = flying[(flying["Driver"] == driver) & (flying["RunId"] == best["RunId"])].sort_values("LapNumber")
        lap_in_run = int((run_laps["LapNumber"] <= best["LapNumber"]).sum())

        longest_run_id = driver_runs.loc[driver_runs["n_laps"].idxmax(), "RunId"] if len(driver_runs) else None
        long_run_laps = flying[(flying["Driver"] == driver) & (flying["RunId"] == longest_run_id)]

        best_lap_position = (
            (best["LapStartTime"] - session_start).total_seconds() / session_span if session_span > 0 else 0.0
        )

        # TyreLife on the specific best lap is sometimes missing from FastF1
        # even when the lap itself is fine - fall back to this driver's own
        # median (other flying laps), then the session-wide median, rather
        # than leaving a lone NaN in an otherwise fully-populated row (which
        # would break sequences.build_lstm_sequences's all-or-nothing check).
        tyre_life = best["TyreLife"]
        if pd.isna(tyre_life):
            tyre_life = grp["TyreLife"].median()
        if pd.isna(tyre_life):
            tyre_life = flying["TyreLife"].median()

        # A driver with no run meeting the long-run length threshold has no
        # "long_run_avg_pace" to compute - their overall median flying-lap
        # pace is the best available proxy, and keeps this feature populated
        # whenever best_lap_time is (same all-or-nothing reasoning as above).
        long_run_avg_pace = (
            float(long_run_laps["lap_time_seconds"].mean())
            if len(long_run_laps)
            else float(grp["lap_time_seconds"].median())
        )

        rows.append({
            "Driver": driver,
            "best_lap_time": best["lap_time_seconds"],
            "gap_to_session_best": best["lap_time_seconds"] - session_best,
            "median_flying_lap_time": float(grp["lap_time_seconds"].median()),
            "lap_time_std": float(grp["lap_time_seconds"].std()) if len(grp) > 1 else 0.0,
            "n_runs": int(driver_runs["RunId"].nunique()) if len(driver_runs) else 0,
            "n_flying_laps": int(len(grp)),
            "avg_run_length": float(driver_runs["n_laps"].mean()) if len(driver_runs) else 0.0,
            "longest_run_length": int(driver_runs["n_laps"].max()) if len(driver_runs) else 0,
            "compound_on_best_lap": COMPOUND_ORDER.get(str(best["Compound"]).upper(), DEFAULT_COMPOUND_ORD),
            "tyre_life_on_best_lap": float(tyre_life) if pd.notna(tyre_life) else 0.0,
            "fuel_corrected_pace": fuel_corrected_pace(
                best["lap_time_seconds"], lap_in_run, fuel_burn_seconds_per_lap
            ),
            "long_run_avg_pace": long_run_avg_pace,
            "best_sector1_time": _best_sector_seconds(grp, "Sector1Time"),
            "best_sector2_time": _best_sector_seconds(grp, "Sector2Time"),
            "best_sector3_time": _best_sector_seconds(grp, "Sector3Time"),
            "best_lap_session_position": best_lap_position,
            # Not a model feature - the join key join_telemetry_features needs.
            "best_lap_number": best["LapNumber"],
        })
    return pd.DataFrame(rows)


def join_weather_features(features: pd.DataFrame, weather_row: dict | None) -> pd.DataFrame:
    """Broadcast one session's weather aggregate onto every driver row.

    `weather_row` comes from `f1qp.features.weather.weather_row_for_session`
    (or None if that session isn't in data/processed/weather.parquet yet) -
    every driver in a session shares the same weather, it isn't per-driver.
    """
    out = features.copy()
    for col in WEATHER_FEATURE_COLUMNS:
        out[col] = weather_row[col] if weather_row else np.nan
    return out


def join_telemetry_features(features: pd.DataFrame, telemetry_by_driver_lap: pd.DataFrame | None) -> pd.DataFrame:
    """Join telemetry trend features onto each driver's own best lap (`best_lap_number`).

    `telemetry_by_driver_lap` is the concatenated output of
    `f1qp.features.telemetry.telemetry_row_for_lap` calls (one row per
    representative lap) for this session, or None if
    scripts/extract_telemetry.py hasn't produced it yet for this session.
    A driver whose best lap isn't in the telemetry table (extraction
    failed for that one lap, or hasn't reached it yet) gets NaN telemetry
    columns rather than dropping the row - unless this driver has telemetry
    for other laps in the same session, in which case those are used as a
    fallback (see below) so telemetry stays NaN only when the whole session
    has none yet, matching every other feature's all-or-nothing presence.
    """
    out = features.copy()
    if telemetry_by_driver_lap is None or telemetry_by_driver_lap.empty:
        for col in TELEMETRY_FEATURE_COLUMNS:
            out[col] = np.nan
        return out

    merged = out.merge(
        telemetry_by_driver_lap[["Driver", "LapNumber", *TELEMETRY_FEATURE_COLUMNS]],
        left_on=["Driver", "best_lap_number"],
        right_on=["Driver", "LapNumber"],
        how="left",
    )
    merged = merged.drop(columns=["LapNumber"])

    # The driver's exact best lap can be missing from telemetry.parquet
    # (a per-lap extraction failure) even though the session has telemetry
    # overall. Backfill from this driver's other laps first, then the
    # session-wide mean, rather than leaving these 5 columns NaN while the
    # rest of the row (pace/weather/tyre) is populated - a partial-NaN row
    # that sequences.build_lstm_sequences would reject as a data bug.
    missing = merged[TELEMETRY_FEATURE_COLUMNS[0]].isna()
    if missing.any():
        driver_means = telemetry_by_driver_lap.groupby("Driver")[TELEMETRY_FEATURE_COLUMNS].mean()
        session_means = telemetry_by_driver_lap[TELEMETRY_FEATURE_COLUMNS].mean()
        for col in TELEMETRY_FEATURE_COLUMNS:
            fallback = merged.loc[missing, "Driver"].map(driver_means[col]).fillna(session_means[col])
            merged.loc[missing, col] = merged.loc[missing, col].fillna(fallback)
    return merged
