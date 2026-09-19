"""Phase 2: telemetry trend features for each run's representative lap.

Practice sessions only (FP1-FP3, or FP1+SQ/SS on sprint weekends) - never Q
or the race. This is the one Phase 2 extraction script that needs real
network calls: Phase 1 never requested telemetry (`telemetry=False`
everywhere), so nothing here is cached yet. Worth being precise about what
"only a representative subset" actually saves: FastF1 fetches one driver's
whole-session car telemetry in a single call regardless of how many laps
you read off it afterward, so the fetch itself isn't lap-selective - what
IS deliberately limited is (a) practice sessions only, never Q or the race,
and (b) only the aggregate trend for each run's single fastest flying lap
(f1qp.features.telemetry.representative_laps) gets computed and saved, not
the raw per-sample channels for every lap. That is still the meaningful
scope cut from a full telemetry backfill.

This is the slow script - budget real time for it, and consider running it
in the background the way download_historical.py ran in Phase 1.

Resumable and rate-limit aware (see Pit Wall > Incident Report - a run got
cut off by FastF1's own client-side "any API: 500 calls/h" self-throttle
partway through a backfill, and a second concurrent invocation raced the
first through the same session list instead of splitting the work):
  - Refuses to start if another instance's lock file is present.
  - Loads whatever is already in OUT_PATH and skips any (year, round,
    session, driver, lap) already covered - a session fully covered by a
    prior run costs no network call at all on a re-run.
  - If FastF1's rate limiter trips, stops the loop immediately (instead of
    logging 60+ doomed per-session failures) and still writes out whatever
    was collected this run merged with what was already on disk, so a
    later re-run picks up exactly where this one stopped.

Run from the repo root:
    python scripts/extract_telemetry.py
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from f1qp.config import OFFLINE_TEST_ROUND, all_training_events, get_event_info
from f1qp.data.cache import enable_cache
from f1qp.features.runs import add_run_features
from f1qp.features.telemetry import representative_laps, telemetry_row_for_lap
from f1qp.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "telemetry.parquet"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
LOCK_PATH = OUT_PATH.with_suffix(".lock")
REQUEST_PAUSE_SECONDS = 1.5


def _events() -> list[tuple[int, int]]:
    events = all_training_events()
    events.append((2026, OFFLINE_TEST_ROUND))
    return events


def _laps_for(year: int, round_number: int, session_code: str) -> pd.DataFrame | None:
    year_dir = RAW_DIR / str(year)
    matches = list(year_dir.glob(f"r{round_number:02d}_*.parquet"))
    if not matches:
        return None
    df = pd.read_parquet(matches[0])
    return df[df["SessionCode"] == session_code]


def _load_existing() -> pd.DataFrame:
    if not OUT_PATH.exists():
        return pd.DataFrame(columns=["Year", "RoundNumber", "SessionCode", "Driver", "LapNumber"])
    return pd.read_parquet(OUT_PATH)


def _row_key(year: int, round_number: int, session_code: str, driver: str, lap_number: float) -> tuple:
    return (int(year), int(round_number), str(session_code), str(driver), float(lap_number))


def _existing_keys(existing: pd.DataFrame) -> set[tuple]:
    return {
        _row_key(r.Year, r.RoundNumber, r.SessionCode, r.Driver, r.LapNumber)
        for r in existing.itertuples(index=False)
    }


def _acquire_lock() -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive create ("x") is atomic at the OS level - unlike a separate
        # exists()-then-write_text(), two processes starting within microseconds
        # of each other can't both pass this check.
        with LOCK_PATH.open("x") as f:
            f.write(str(time.time()))
    except FileExistsError:
        raise SystemExit(
            f"Lock file {LOCK_PATH} already exists - another extract_telemetry.py run looks to be in "
            "progress (or a previous one crashed without cleaning up). Running two instances at once "
            "doesn't split the work: both walk the same event list and race each other for the same "
            "FastF1 rate-limit budget. Wait for the other run to finish, or delete the lock file "
            "yourself if you're sure none is actually running."
        ) from None


def _release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def main() -> None:
    _acquire_lock()
    try:
        _run()
    finally:
        _release_lock()


def _run() -> None:
    enable_cache()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing()
    done_keys = _existing_keys(existing)
    logger.info("Resuming with %s representative laps already on disk from a previous run.", len(existing))

    rows = []
    failures = []
    rate_limited = False
    for year, round_number in _events():
        if rate_limited:
            break
        event = get_event_info(year, round_number)
        for session_code in event.practice_sessions:
            laps = _laps_for(year, round_number, session_code)
            if laps is None or laps.empty:
                if (year, round_number) == (2026, OFFLINE_TEST_ROUND):
                    # Not a gap to fix: this is the held-out offline validation round, and its
                    # raw laps only exist once Phase 1's download has been (re-)run for it - which
                    # can't happen until the weekend actually finishes. Expected to be "missing"
                    # here for as long as the round is still in progress.
                    logger.info(
                        "%s R%s %s: offline validation round not finished yet - raw laps aren't "
                        "downloaded until Phase 1 is re-run for it after the weekend concludes. "
                        "Skipping for now, not a gap in the training data.",
                        year, round_number, session_code,
                    )
                else:
                    logger.info("No raw laps on disk for %s R%s %s - skipping (re-run Phase 1 downloads first?)",
                                year, round_number, session_code)
                continue

            reps = representative_laps(add_run_features(laps))
            if reps.empty:
                continue

            reps = reps[
                ~reps.apply(
                    lambda rep: _row_key(year, round_number, session_code, rep["Driver"], rep["LapNumber"])
                    in done_keys,
                    axis=1,
                )
            ]
            if reps.empty:
                logger.info("%s R%s %s already fully covered on disk - skipping (no network call).",
                            year, round_number, session_code)
                continue

            try:
                session = fastf1.get_session(year, round_number, session_code)
                session.load(laps=True, telemetry=True, weather=False, messages=False)
            except RateLimitExceededError as exc:
                logger.warning(
                    "FastF1 rate limit hit (%s) while loading %s R%s %s - stopping here instead of "
                    "burning through the rest of the event list. Progress so far will still be saved; "
                    "re-run this script later (once the rate-limit window has reset) to pick up where "
                    "this stopped.",
                    exc, year, round_number, session_code,
                )
                rate_limited = True
                break
            except Exception as exc:
                logger.warning("Telemetry session load failed %s R%s %s: %s", year, round_number, session_code, exc)
                failures.append((year, round_number, session_code, "session load", str(exc)))
                continue

            for _, rep in reps.iterrows():
                driver, lap_number = rep["Driver"], rep["LapNumber"]
                try:
                    driver_laps = session.laps[session.laps["Driver"] == driver]
                    lap_row = driver_laps[driver_laps["LapNumber"] == lap_number]
                    if lap_row.empty:
                        logger.warning("Lap not found in reloaded session: %s R%s %s %s L%s",
                                        year, round_number, session_code, driver, lap_number)
                        continue
                    car_data = lap_row.iloc[0].get_car_data()
                    rows.append(telemetry_row_for_lap(
                        car_data, year=year, round_number=round_number, session_code=session_code,
                        driver=driver, lap_number=lap_number,
                    ))
                except RateLimitExceededError as exc:
                    logger.warning(
                        "FastF1 rate limit hit (%s) mid-session on %s R%s %s %s L%s - stopping here; "
                        "progress so far will still be saved.",
                        exc, year, round_number, session_code, driver, lap_number,
                    )
                    rate_limited = True
                    break
                except Exception as exc:
                    logger.warning("Telemetry extraction failed %s R%s %s %s L%s: %s",
                                    year, round_number, session_code, driver, lap_number, exc)
                    failures.append((year, round_number, session_code, f"{driver} L{lap_number}", str(exc)))

            if rate_limited:
                break

            time.sleep(REQUEST_PAUSE_SECONDS)

    new_rows = pd.DataFrame(rows)
    out = pd.concat([existing, new_rows], ignore_index=True) if len(new_rows) else existing
    out.to_parquet(OUT_PATH)
    logger.info("Wrote %s - %s representative laps total (%s new this run), %s failures.",
                OUT_PATH, len(out), len(new_rows), len(failures))
    if rate_limited:
        logger.warning("Stopped early due to the FastF1 rate limit - re-run this script later to resume.")
    if failures:
        for year, round_number, session_code, what, err in failures:
            logger.warning("  - %s R%s %s (%s): %s", year, round_number, session_code, what, err)


if __name__ == "__main__":
    main()
