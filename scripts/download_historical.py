"""Phase 1: 2023-2025 backfill.

This is the big pull - 72 weekends x 3-4 sessions each - and the #1
timeline risk in the project plan. Built to run as a background job
across Day 1-2, not inside a single sitting:

  - Idempotent: a round already saved to data/raw/<year>/ is skipped, so
    it's safe to Ctrl-C and restart, or run it again tomorrow to pick up
    where it left off.
  - No telemetry by default - laps + weather only (see loader.py).
  - Failures on one round are logged and skipped rather than aborting the
    whole run; re-running later retries just the missing ones.
  - Rounds download in parallel via a ThreadPoolExecutor (MAX_WORKERS
    threads). The work is I/O-bound (waiting on the FastF1/Ergast API),
    not CPU-bound, so threads are enough - and they share the single
    enable_cache() call made below, which matters because FastF1's cache
    is class-level, process-wide state backed by one shared sqlite file
    that isn't documented as safe to hit from separate OS processes.
  - At the end, every weekend that failed to download is collected and
    logged as one summary list (in addition to being logged individually
    as it happens), so it's obvious what to look at before re-running.
  - FastF1 enforces its own client-side rate limit (500 calls/h to any one
    API, sliding 1h window - see fastf1.req). Once that trips, every
    remaining call fails the same way until the window drains, so the
    first RateLimitExceededError seen stops new downloads for the rest of
    this run instead of burning retries/time on rounds that can't
    possibly succeed yet - those get reported separately as "rate
    limited" rather than lumped in with real failures. Idempotency means
    simply re-running later (once the window has drained) picks up
    exactly those rounds and nothing else.

Run from the repo root, ideally in a background terminal:
    python scripts/download_historical.py
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from f1qp.config import all_training_events, get_event_info
from f1qp.data.cache import enable_cache
from f1qp.data.loader import load_weekend
from f1qp.data.schema import validate_laps
from f1qp.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MAX_WORKERS = 3  # modest on purpose - politeness to the API + the shared sqlite cache


@dataclass
class RoundResult:
    year: int
    round_number: int
    event_name: str | None
    status: str  # "downloaded" | "skipped" | "failed" | "rate_limited"
    error: str | None = None


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True if exc (or anything chained under it via 'raise ... from') is FastF1's rate-limit error."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, RateLimitExceededError):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def _download_one_round(year: int, round_number: int, rate_limited: threading.Event) -> RoundResult:
    """Download and save one weekend. Runs in a worker thread; safe to call concurrently."""
    event = get_event_info(year, round_number)
    year_dir = RAW_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    out_path = year_dir / f"r{round_number:02d}_{event.event_name.replace(' ', '_')}.parquet"
    if out_path.exists():
        return RoundResult(year, round_number, event.event_name, "skipped")

    if rate_limited.is_set():
        # Another thread already tripped the API's rate limit this run - it won't have
        # drained in the meantime, so don't waste a real request finding that out again.
        return RoundResult(year, round_number, event.event_name, "rate_limited")

    try:
        sessions = load_weekend(year, round_number, telemetry=False)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            rate_limited.set()
            logger.warning(
                "Hit FastF1's API rate limit (500 calls/h) on %s R%s (%s) - "
                "no new downloads will start this run; wait for the window to drain and re-run.",
                year, round_number, event.event_name,
            )
            return RoundResult(year, round_number, event.event_name, "rate_limited", str(exc))
        logger.exception(
            "Giving up on %s R%s (%s) for now - will retry next run",
            year, round_number, event.event_name,
        )
        return RoundResult(year, round_number, event.event_name, "failed", str(exc))

    frames = []
    for loaded in sessions:
        laps = loaded.session.laps.copy()
        laps["SessionCode"] = loaded.session_code
        laps["Year"] = year
        laps["RoundNumber"] = round_number
        laps["IsSprint"] = event.is_sprint
        report = validate_laps(laps, year=year, round_number=round_number, session_code=loaded.session_code)
        if not report.is_clean:
            logger.warning(
                "Schema issue %s R%s %s: missing=%s notes=%s",
                year, round_number, loaded.session_code, report.missing_columns, report.notes,
            )
        frames.append(laps)

    pd.concat(frames, ignore_index=True).to_parquet(out_path)
    logger.info("Saved %s", out_path)
    return RoundResult(year, round_number, event.event_name, "downloaded")


def main() -> None:
    enable_cache()
    events = [(y, r) for y, r in all_training_events() if y != 2026]  # 2026 handled by download_2026.py
    logger.info(
        "Backfilling %s historical weekends (2023-2025) with %s parallel workers.",
        len(events), MAX_WORKERS,
    )

    rate_limited = threading.Event()
    results: list[RoundResult] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_event = {
            executor.submit(_download_one_round, year, round_number, rate_limited): (year, round_number)
            for year, round_number in events
        }
        for i, future in enumerate(as_completed(future_to_event), start=1):
            year, round_number = future_to_event[future]
            try:
                result = future.result()
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    rate_limited.set()
                    result = RoundResult(year, round_number, None, "rate_limited", str(exc))
                else:
                    logger.exception(
                        "Unexpected error downloading %s R%s - will retry next run", year, round_number
                    )
                    result = RoundResult(year, round_number, None, "failed", str(exc))
            results.append(result)
            logger.info("[%s/%s] %s R%s: %s", i, len(events), year, round_number, result.status)

    downloaded = [r for r in results if r.status == "downloaded"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    rate_limited_rounds = [r for r in results if r.status == "rate_limited"]

    logger.info(
        "Backfill complete: %s downloaded, %s already present, %s failed, %s rate-limited.",
        len(downloaded), len(skipped), len(failed), len(rate_limited_rounds),
    )

    if failed:
        failed_sorted = sorted(failed, key=lambda r: (r.year, r.round_number))
        logger.warning(
            "Failed to download %s weekend(s) - rerun this script to retry just these:", len(failed)
        )
        for r in failed_sorted:
            label = r.event_name or "unknown event"
            logger.warning("  - %s R%02d (%s): %s", r.year, r.round_number, label, r.error)

    if rate_limited_rounds:
        rl_sorted = sorted(rate_limited_rounds, key=lambda r: (r.year, r.round_number))
        logger.warning(
            "Skipped %s weekend(s) due to FastF1's API rate limit (500 calls/h) - "
            "wait for the window to drain (up to ~1h from your last burst of calls) "
            "and re-run to pick these up:",
            len(rate_limited_rounds),
        )
        for r in rl_sorted:
            label = r.event_name or "unknown event"
            logger.warning("  - %s R%02d (%s)", r.year, r.round_number, label)


if __name__ == "__main__":
    main()
