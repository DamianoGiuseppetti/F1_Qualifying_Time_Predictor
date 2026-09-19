"""Phase 2: extract weather aggregates for every training + offline-test session.

Cache hit only - `session.load(weather=True)` reuses Phase 1's fastf1 disk
cache (data/cache/, already warmed by download_2026.py /
download_historical.py), so this makes no new network calls and should
finish in well under a minute. See docs/feature_engineering.md for why
this script exists: Phase 1 requested weather but never saved it.

Run from the repo root:
    python scripts/extract_weather.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import pandas as pd

from f1qp.config import OFFLINE_TEST_ROUND, all_training_events, get_event_info
from f1qp.data.cache import enable_cache
from f1qp.features.weather import weather_row_for_session
from f1qp.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "weather.parquet"


def _events() -> list[tuple[int, int]]:
    events = all_training_events()
    events.append((2026, OFFLINE_TEST_ROUND))  # Zandvoort - needed for the offline validation too
    return events


def main() -> None:
    enable_cache()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for year, round_number in _events():
        event = get_event_info(year, round_number)
        for session_code in event.practice_sessions + ["Q"]:
            try:
                session = fastf1.get_session(year, round_number, session_code)
                session.load(laps=False, telemetry=False, weather=True, messages=False)
                rows.append(
                    weather_row_for_session(
                        session.weather_data, year=year, round_number=round_number, session_code=session_code
                    )
                )
            except Exception as exc:
                logger.warning("Weather extraction failed %s R%s %s: %s", year, round_number, session_code, exc)
                failures.append((year, round_number, session_code, str(exc)))

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH)
    logger.info("Wrote %s - %s sessions, %s failures.", OUT_PATH, len(rows), len(failures))
    if failures:
        logger.warning("Failed sessions (re-run this script to retry - it isn't idempotent-by-skip, it's fast enough to just redo):")
        for year, round_number, session_code, err in failures:
            logger.warning("  - %s R%s %s: %s", year, round_number, session_code, err)


if __name__ == "__main__":
    main()
