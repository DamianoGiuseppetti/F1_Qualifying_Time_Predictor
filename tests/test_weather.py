import pandas as pd
import pytest

from f1qp.features.weather import aggregate_session_weather, weather_row_for_session


def _weather_data(n: int = 10, rain_share: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "Time": pd.to_timedelta(range(0, n * 60, 60), unit="s"),
        "AirTemp": [22.0 + i * 0.1 for i in range(n)],
        "TrackTemp": [34.0 + i * 0.2 for i in range(n)],
        "Humidity": [55.0] * n,
        "Pressure": [1013.0] * n,
        "Rainfall": [i < n * rain_share for i in range(n)],
        "WindDirection": [180] * n,
        "WindSpeed": [1.5] * n,
    })


def test_aggregate_session_weather_dry_session():
    agg = aggregate_session_weather(_weather_data())
    assert agg["rainfall_share"] == pytest.approx(0.0)
    assert 20 < agg["air_temp_mean"] < 25
    assert 30 < agg["track_temp_mean"] < 40
    assert set(agg) == {
        "air_temp_mean", "air_temp_max", "track_temp_mean", "track_temp_max",
        "humidity_mean", "rainfall_share", "wind_speed_mean",
    }


def test_aggregate_session_weather_partial_rain():
    agg = aggregate_session_weather(_weather_data(n=10, rain_share=0.4))
    assert agg["rainfall_share"] == pytest.approx(0.4)


def test_aggregate_session_weather_raises_on_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_session_weather(pd.DataFrame({"AirTemp": [20.0]}))


def test_aggregate_session_weather_raises_on_empty_frame():
    empty = pd.DataFrame({c: [] for c in ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]})
    with pytest.raises(ValueError, match="empty"):
        aggregate_session_weather(empty)


def test_weather_row_for_session_includes_join_keys():
    row = weather_row_for_session(_weather_data(), year=2026, round_number=10, session_code="FP2")
    assert row["Year"] == 2026
    assert row["RoundNumber"] == 10
    assert row["SessionCode"] == "FP2"
    assert "track_temp_mean" in row
