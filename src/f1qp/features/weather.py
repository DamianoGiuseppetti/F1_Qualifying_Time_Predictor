"""Session-level weather aggregates from FastF1's `Session.weather_data`.

Phase 1 requested weather (`session.load(..., weather=True, ...)`) but never
saved it - see docs/feature_engineering.md for the full story. This module
only does the aggregation; `scripts/extract_weather.py` is what actually
reopens each cached session and calls it, since that needs a live `Session`
object this module has no business constructing.

Deliberately NOT merged onto lap rows: `weather_data` is a session-level
time series (~1 sample/minute), so joining it onto every lap would repeat
the same handful of numbers dozens of times. One aggregate row per session
is what actually gets joined onto the feature table.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_WEATHER_COLUMNS = ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]


def aggregate_session_weather(weather_data: pd.DataFrame) -> dict:
    """Collapse one session's weather time series into a handful of numbers.

    Raises ValueError on missing columns or an empty frame - callers should
    treat "no weather data for this session" as a case worth knowing about,
    not a silently-zero feature row.
    """
    missing = [c for c in REQUIRED_WEATHER_COLUMNS if c not in weather_data.columns]
    if missing:
        raise ValueError(f"aggregate_session_weather is missing required columns: {missing}")
    if weather_data.empty:
        raise ValueError("aggregate_session_weather got an empty weather_data frame")

    return {
        "air_temp_mean": float(weather_data["AirTemp"].mean()),
        "air_temp_max": float(weather_data["AirTemp"].max()),
        "track_temp_mean": float(weather_data["TrackTemp"].mean()),
        "track_temp_max": float(weather_data["TrackTemp"].max()),
        "humidity_mean": float(weather_data["Humidity"].mean()),
        "rainfall_share": float(weather_data["Rainfall"].astype(bool).mean()),
        "wind_speed_mean": float(weather_data["WindSpeed"].mean()),
    }


def weather_row_for_session(
    weather_data: pd.DataFrame, *, year: int, round_number: int, session_code: str
) -> dict:
    """`aggregate_session_weather` plus the join keys, ready to append into a rows list."""
    row = {"Year": year, "RoundNumber": round_number, "SessionCode": session_code}
    row.update(aggregate_session_weather(weather_data))
    return row
