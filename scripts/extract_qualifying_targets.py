"""Phase 2: extract the official Q1/Q2/Q3 classification - the model's actual target.

Cache hit only, same reasoning as extract_weather.py. Phase 1 saved raw Q
laps but never `session.results`, which is where FastF1 publishes each
driver's official per-segment best time. Raw laps alone don't reliably say
where Q1 ends and Q2 begins (a timing-system decision, not a fixed lap
count), so this can't be reconstructed after the fact from what's on disk.
See docs/feature_engineering.md.

**--round N (Aug 30 2026)**: `_events()` was hardcoded to the training
seasons + Round 12 only, so once a new round - Round 13 (Monza) onward -
actually runs qualifying, its official result never made it into
qualifying_targets.parquet without this flag, and the History tab could
never show it as scored. Pass the round once its Q session has happened
(repeat --round for more than one); a round whose Q hasn't run yet just
logs a lookup failure and is skipped, same as always.

**Incremental when --round is passed (Sep 13 2026)**: the bare command
(no --round) is UNCHANGED - it still re-fetches every event in
`_events()` fresh from FastF1 every time, exactly as documented above.
But `--round N` is exactly what the app's own on-demand "check for
official result" button uses (f1qp.serving.data_fetch.start_results_job),
and on a Hugging Face Space that command used to re-fetch the ENTIRE
2023-2025 + 2026 R1-R12 history from FastF1 over the network on every
single click - the Space never bakes in the (huge, training-only) FastF1
session cache, so every one of those historical events would be a cold
network fetch, easily overrunning data_fetch.py's own 900s per-step
timeout just to check one new round's result. Now, whenever --round is
passed at least once, an event already sitting in the existing
qualifying_targets.parquet is skipped (carried forward as-is) UNLESS it
was itself one of the requested --round values - so "check for official
result" only ever actually contacts FastF1 for the one round being
checked.

Run from the repo root:
    python scripts/extract_qualifying_targets.py
    python scripts/extract_qualifying_targets.py --round 13
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import fastf1
import pandas as pd

from f1qp.config import OFFLINE_TEST_ROUND, all_training_events
from f1qp.data.cache import enable_cache
from f1qp.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "qualifying_targets.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--round",
        type=int,
        action="append",
        default=[],
        dest="extra_rounds",
        metavar="N",
        help=(
            "Additional 2026 round number to extract the official "
            "qualifying result for, beyond the default R1-R12 set (repeat "
            "--round for more than one) - pass it once that round's Q "
            "session has actually happened."
        ),
    )
    return parser.parse_args()


def _events(extra_rounds: list[int] = ()) -> list[tuple[int, int]]:
    events = all_training_events()
    events.append((2026, OFFLINE_TEST_ROUND))
    for r in extra_rounds:
        pair = (2026, r)
        if pair not in events:
            events.append(pair)
    return events


def _segment_seconds(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return value.total_seconds()


def _needs_fetch(
    year: int, round_number: int, extra_rounds: list[int], existing_keys: set[tuple[int, int]]
) -> bool:
    """See module docstring ("Incremental when --round is passed").

    Sep 13 2026 fix: the bare/no---round invocation (`extra_rounds` empty)
    always fetches everything, full stop - this is guaranteed here
    directly rather than relied on via main() happening to pass an empty
    `existing_keys` in that case, so this function can never accidentally
    skip a fetch no matter what a caller passes for `existing_keys`.

    Otherwise: an explicitly requested --round is always (re)fetched -
    Damiano is asking for that round's result right now. Everything else
    is fetched only if it ISN'T already in `existing_keys` (the previous
    run's output)."""
    if not extra_rounds:
        return True
    if (year, round_number) in {(2026, r) for r in extra_rounds}:
        return True
    return (year, round_number) not in existing_keys


def main() -> None:
    args = parse_args()
    enable_cache()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Sep 13 2026 addition - only engaged when --round is passed (see
    # module docstring); the bare invocation's full-refetch behavior is
    # untouched (existing_keys stays empty below).
    incremental = bool(args.extra_rounds)
    existing_df = None
    existing_keys: set[tuple[int, int]] = set()
    if incremental and OUT_PATH.exists():
        try:
            existing_df = pd.read_parquet(OUT_PATH)
            existing_keys = set(zip(existing_df["Year"], existing_df["RoundNumber"]))
        except Exception as exc:
            logger.warning(
                "Couldn't read existing %s (%s) - falling back to a full refetch this run.",
                OUT_PATH, exc,
            )
            existing_df = None
            existing_keys = set()

    rows = []
    failures = []
    skipped = 0
    for year, round_number in _events(args.extra_rounds):
        if not _needs_fetch(year, round_number, args.extra_rounds, existing_keys):
            skipped += 1
            continue
        try:
            session = fastf1.get_session(year, round_number, "Q")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            results = session.results
            for _, r in results.iterrows():
                rows.append({
                    "Year": year,
                    "RoundNumber": round_number,
                    "Driver": r["Abbreviation"],
                    "Q1": _segment_seconds(r.get("Q1")),
                    "Q2": _segment_seconds(r.get("Q2")),
                    "Q3": _segment_seconds(r.get("Q3")),
                })
        except Exception as exc:
            logger.warning("Qualifying target extraction failed %s R%s: %s", year, round_number, exc)
            failures.append((year, round_number, str(exc)))

    fetched = pd.DataFrame(rows)
    fetched_keys = set(zip(fetched["Year"], fetched["RoundNumber"])) if not fetched.empty else set()
    if existing_df is not None and not existing_df.empty:
        # Carry forward every existing row EXCEPT ones for an event we
        # just (re)fetched above - a fresh fetch always wins over a
        # carried-forward row for the same event.
        carried = existing_df[
            ~existing_df.apply(lambda r: (r["Year"], r["RoundNumber"]) in fetched_keys, axis=1)
        ]
        out = pd.concat([carried, fetched], ignore_index=True) if not fetched.empty else carried
    else:
        out = fetched

    out.to_parquet(OUT_PATH)
    logger.info(
        "Wrote %s - %s driver-round rows (%s freshly fetched this run, %s event(s) skipped/carried "
        "forward from disk), %s failures.",
        OUT_PATH, len(out), len(rows), skipped, len(failures),
    )
    if failures:
        for year, round_number, err in failures:
            logger.warning("  - %s R%s: %s", year, round_number, err)


if __name__ == "__main__":
    main()
