"""Session loading helpers.

Defaults are deliberately conservative: laps + weather, no car telemetry.
Telemetry is expensive - multiple channels at several Hz per car for a
whole session - and only Phase 2's "telemetry trend" feature needs it, and
only for a subset of laps, not the full historical backfill. Turn it on
explicitly (`telemetry=True`) at the one call site that needs it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import fastf1
from fastf1.core import Session

from f1qp.config import EventInfo, get_event_info

logger = logging.getLogger(__name__)

# Be polite to the upstream data source across a long backfill.
REQUEST_PAUSE_SECONDS = 1.5
MAX_RETRIES = 3


@dataclass
class LoadedSession:
    event: EventInfo
    session_code: str
    session: Session


def load_session(year: int, round_number: int, session_code: str, *, telemetry: bool = False) -> LoadedSession:
    """Load one session with retry/backoff. Raises after MAX_RETRIES failed attempts."""
    event = get_event_info(year, round_number)
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session = fastf1.get_session(year, round_number, session_code)
            session.load(laps=True, telemetry=telemetry, weather=True, messages=False)
            return LoadedSession(event=event, session_code=session_code, session=session)
        except Exception as exc:  # fastf1 raises a handful of different error types upstream
            last_exc = exc
            logger.warning(
                "Load failed (%s R%s %s), attempt %s/%s: %s",
                year, round_number, session_code, attempt, MAX_RETRIES, exc,
            )
            time.sleep(REQUEST_PAUSE_SECONDS * attempt)
    raise RuntimeError(f"Giving up on {year} R{round_number} {session_code}") from last_exc


def load_weekend(year: int, round_number: int, *, telemetry: bool = False) -> list[LoadedSession]:
    """Load every practice + qualifying session for one weekend, normal or sprint."""
    event = get_event_info(year, round_number)
    codes = event.practice_sessions + ["Q"]
    loaded = []
    for code in codes:
        loaded.append(load_session(year, round_number, code, telemetry=telemetry))
        time.sleep(REQUEST_PAUSE_SECONDS)
    return loaded
