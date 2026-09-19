"""
Season / round configuration for the F1 Qualifying Predictor.

Sprint-vs-normal format is NOT hardcoded here. The 2026 calendar has
already changed once this season (Bahrain and Saudi Arabia were dropped,
cutting the season from 24 to 22 rounds), so round numbers are not safe as
compile-time constants. `get_event_info()` reads the format straight off
FastF1's own event schedule instead, which stays correct as long as
fastf1 itself is kept up to date.

Known and confirmed as of Aug 2026:
  - Round 12 ("Olanda") = Dutch GP, Zandvoort, Aug 21-23 - a sprint weekend.
    This is the offline validation round.
  - Round 13 = Italian GP, Monza, Sep 4-6 - a standard (non-sprint) weekend.
    This is the live-deployment round.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import fastf1
import pandas as pd

TRAINING_SEASONS = [2023, 2024, 2025]
SEASON_2026_TRAIN_ROUNDS = list(range(1, 12))  # R1-R11
OFFLINE_TEST_ROUND = 12  # "Olanda" - Zandvoort
LIVE_DEPLOY_ROUND = 13  # Monza

NORMAL_WEEKEND_PRACTICE = ["FP1", "FP2", "FP3"]
# The Saturday sprint-quali session's FastF1 identifier is NOT stable across seasons:
# 2023 used the "sprint_shootout" format (session named "Sprint Shootout", identifier "SS");
# 2024+ renamed it to the "sprint_qualifying" format (session named "Sprint Qualifying",
# identifier "SQ"). Using the wrong one raises "Session type '...' does not exist for this
# event" from fastf1.get_session() - see get_event_info() below for the format-aware pick.
SPRINT_SHOOTOUT_PRACTICE = ["FP1", "SS"]  # 2023-style "sprint_shootout" weekends
SPRINT_WEEKEND_PRACTICE = ["FP1", "SQ"]  # 2024+ "sprint_qualifying" weekends
QUALIFYING_SESSION = "Q"


@dataclass(frozen=True)
class EventInfo:
    year: int
    round_number: int
    event_name: str
    is_sprint: bool
    practice_sessions: list[str]


@functools.lru_cache(maxsize=8)
def _schedule(year: int):
    """Cached wrapper so a whole script only hits the schedule endpoint once per year."""
    return fastf1.get_event_schedule(year, include_testing=False)


def get_event_info(year: int, round_number: int) -> EventInfo:
    """Look up whether a round is a sprint weekend directly from FastF1's own schedule.

    Do not hardcode a sprint-round list here - it has already shifted once
    this season and the exact `EventFormat` string fastf1 uses has varied
    across versions ("sprint_qualifying", "sprint_shootout", "sprint").
    Matching on "sprint" in the lowercased string is deliberately loose to
    absorb that.
    """
    schedule = _schedule(year)
    row = schedule.loc[schedule["RoundNumber"] == round_number].iloc[0]
    event_format = str(row["EventFormat"]).lower()
    is_sprint = "sprint" in event_format
    if "shootout" in event_format:
        sessions = SPRINT_SHOOTOUT_PRACTICE
    elif is_sprint:
        sessions = SPRINT_WEEKEND_PRACTICE
    else:
        sessions = NORMAL_WEEKEND_PRACTICE
    return EventInfo(
        year=year,
        round_number=round_number,
        event_name=str(row["EventName"]),
        is_sprint=is_sprint,
        practice_sessions=sessions,
    )


def get_event_name(year: int, round_number: int) -> Optional[str]:
    """Best-effort Grand Prix name for display purposes ("Italian Grand
    Prix" rather than just "Round 16") - Damiano, Aug 30 2026: "Add the
    name of the GP not only the round." Wraps `get_event_info` in a
    try/except: a FastF1 schedule lookup failure (network hiccup, a round
    number the schedule doesn't recognize) must never break /predict,
    /predict/.../launch, or /history - it should just mean no GP name is
    shown, exactly as before this existed. Used by f1qp.api.main (the
    live prediction responses) and f1qp.serving.history (the History
    rows) so both pages get the same name from the same source.
    """
    try:
        return get_event_info(year, round_number).event_name
    except Exception:
        return None


# Conservative, DOCUMENTED GUESS, not a measured guarantee (Sep 2026,
# Damiano: "consider to make some type of control before start running
# data fetching... ask the user to confirm... warn that for that round
# data must be fetched and so we need to be sure that practices are over
# with a temporal range of security... you advise how many time to
# wait"). A practice or sprint-qualifying session can itself run up to
# ~90 minutes including any red-flag delays, and this project's own
# experience with FastF1's underlying timing feed is that it can take
# anywhere from a few minutes up to roughly an hour to be fully populated
# once a session ends. 150 = 90 (session) + 60 (feed lag) minutes after
# the session's SCHEDULED START (not end - FastF1's schedule only gives
# start times reliably, and this project deliberately doesn't try to be
# cleverer than the schedule data actually supports - see get_event_info's
# docstring on the same principle). Purely advisory: session_readiness()
# below never blocks a fetch, it only informs the warning shown before
# one - see f1qp.api.main's /data/readiness endpoint.
DATA_READY_BUFFER_MINUTES = 150


def session_readiness(year: int, round_number: int) -> dict:
    """Pre-flight check for "is it safe to fetch this round's practice
    data yet".

    Checks the SCHEDULED START time of the LAST session this project
    actually uses as model input - the final entry of
    EventInfo.practice_sessions (FP3 on a normal weekend, Sprint
    Qualifying on a sprint weekend) - against DATA_READY_BUFFER_MINUTES
    above. FastF1 numbers sessions in weekend-chronological order
    (Session1DateUtc, Session2DateUtc, ...) and practice_sessions is built
    as exactly the leading prefix of that order (see get_event_info), so
    the Nth practice_sessions entry lines up with SessionN - checked
    directly against fastf1's own session-name construction for both the
    "conventional" (Practice 1/2/3) and "sprint_qualifying" (Practice 1,
    Sprint Qualifying, Sprint, Qualifying, Race) formats.

    Deliberately does NOT attempt the equivalent check for the QUALIFYING
    session (what scripts/extract_qualifying_targets.py needs) - on a
    sprint weekend there's a Sprint race sitting between Sprint Qualifying
    and Qualifying, so "the next session after practice" is NOT Qualifying
    there, and guessing further risks a confidently wrong answer instead
    of a merely absent one. The app's "check for official result" action
    skips this warning and asks a plain confirmation instead.

    Never raises - a lookup failure (unrecognized round, schedule fetch
    failure) comes back as ready=True, checked=False with an explanatory
    message, so a readiness-check problem never blocks the fetch it was
    only ever meant to advise on.
    """
    try:
        event = get_event_info(year, round_number)
        n = len(event.practice_sessions)
        schedule = _schedule(year)
        row = schedule.loc[schedule["RoundNumber"] == round_number].iloc[0]
        start_raw = row[f"Session{n}DateUtc"]
        if pd.isna(start_raw):
            return {
                "ready": True,
                "checked": False,
                "message": "No scheduled session time found for this round - proceeding without a readiness check.",
            }
        start_utc = pd.Timestamp(start_raw)
        if start_utc.tzinfo is None:
            start_utc = start_utc.tz_localize("UTC")
        now_utc = pd.Timestamp.now(tz="UTC")
        minutes_since_start = (now_utc - start_utc).total_seconds() / 60.0
        ready = minutes_since_start >= DATA_READY_BUFFER_MINUTES
        recommended_at = start_utc + timedelta(minutes=DATA_READY_BUFFER_MINUTES)
        last_session_name = event.practice_sessions[-1]

        if minutes_since_start < 0:
            message = (
                f"{last_session_name} for this round hasn't started yet "
                f"(scheduled {start_utc.isoformat()}) - fetching now will find nothing."
            )
        elif not ready:
            message = (
                f"Only {int(minutes_since_start)} min since {last_session_name}'s scheduled start "
                f"({start_utc.isoformat()}) - data may be missing or incomplete this soon. "
                f"Recommended: wait until {recommended_at.isoformat()}."
            )
        else:
            message = f"{last_session_name} started {int(minutes_since_start)} min ago - safe to fetch."

        return {
            "ready": ready,
            "checked": True,
            "minutes_since_session_start": round(minutes_since_start, 1),
            "buffer_minutes": DATA_READY_BUFFER_MINUTES,
            "recommended_fetch_at_utc": recommended_at.isoformat(),
            "message": message,
        }
    except Exception as exc:
        return {
            "ready": True,
            "checked": False,
            "message": f"Readiness check failed ({exc}) - proceeding without it.",
        }


def all_training_events() -> list[tuple[int, int]]:
    """(year, round_number) pairs for every training weekend: full 2023-2025 seasons + 2026 R1-11."""
    pairs: list[tuple[int, int]] = []
    for year in TRAINING_SEASONS:
        schedule = _schedule(year)
        pairs += [(year, int(r)) for r in schedule["RoundNumber"] if r > 0]
    pairs += [(2026, r) for r in SEASON_2026_TRAIN_ROUNDS]
    return pairs
