"""Phase 1: pull the small, high-value dataset first - 2026 rounds 1-11,
PLUS Round 12 (Zandvoort/Olanda, f1qp.config.OFFLINE_TEST_ROUND).

**Bug fixed here (Aug 25 2026)**: this script originally looped only over
SEASON_2026_TRAIN_ROUNDS (R1-R11) - Round 12's raw laps were NEVER
downloaded by anything in Phase 1, even though every Phase 2 script
(extract_weather.py, extract_qualifying_targets.py, extract_telemetry.py,
build_features.py) already assumed it would be there (their own
`_events()` helpers all append OFFLINE_TEST_ROUND). Damiano ran
scripts/evaluate_holdout.py and hit a real FileNotFoundError - phase3_
dataset.parquet had 0 Round 12 rows. Root cause, confirmed by inspecting
the actual files: qualifying_targets.parquet DID have all 22 Round 12
rows (extract_qualifying_targets.py fetches session.results directly from
FastF1, independent of local raw files) but features.parquet had 0 Round
12 rows, because build_features.py only builds features from whatever
raw laps already exist on disk under data/raw/2026/ - and no r12_*.parquet
file was ever there to build from. Fixed by downloading Round 12 here too,
same as every downstream script already expected.

**--round N (Aug 30 2026)**: this script (like build_features.py and
extract_qualifying_targets.py) was hardcoded to R1-R12 only, so a brand
new round - Round 13 (Monza) onward - silently had nothing to download
until this flag existed; the app's own "run scripts/download_2026.py +
scripts/build_features.py for this round first" 404 message wasn't
actually achievable for a new GP without it. Pass the round you want to
predict (repeat --round for more than one); already-downloaded rounds are
skipped as before.

**--only-round N (Sep 16 2026)**: Damiano testing Round 14 found the
app's own "Fetch data now" taking 10+ minutes and eventually timing out.
Root cause: f1qp.serving.data_fetch's on-demand fetch used plain --round,
which APPENDS to the R1-R12 default rather than replacing it - and none
of deploy/render/ or deploy/huggingface/'s Dockerfiles bake data/raw/ in
(deliberately, to keep the image small - see their own comments), so
every fresh container had nothing cached and re-downloaded the ENTIRE
R1-R12 season from FastF1 before ever reaching the round actually asked
for. --only-round bypasses ROUNDS_TO_DOWNLOAD entirely - see
f1qp.serving.data_fetch.start_fetch_job, the only caller that needs this.
Mutually exclusive with --round; local full-bootstrap runs should keep
using --round, which still gets you the whole R1-R12 baseline.

Run from the repo root:
    python scripts/download_2026.py
    python scripts/download_2026.py --round 13
    python scripts/download_2026.py --only-round 14
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from f1qp.config import OFFLINE_TEST_ROUND, SEASON_2026_TRAIN_ROUNDS, get_event_info
from f1qp.data.cache import enable_cache
from f1qp.data.loader import load_weekend
from f1qp.data.schema import validate_laps
from f1qp.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "2026"

# R1-R11 (training) + R12 (Zandvoort/Olanda, the offline-test holdout round -
# see module docstring for why this was missing before). Order matters only
# for log readability; downstream code never assumes it. Any round(s) passed
# via --round are appended to this at run time (see main()), not hardcoded
# here, so a brand-new GP doesn't need a code change to a constant.
ROUNDS_TO_DOWNLOAD = list(SEASON_2026_TRAIN_ROUNDS) + [OFFLINE_TEST_ROUND]


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
            "Additional 2026 round number to download beyond the default "
            "R1-R12 set (repeat --round for more than one) - e.g. a new "
            "GP weekend you're about to predict for the first time."
        ),
    )
    parser.add_argument(
        "--only-round",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Download ONLY this round - skips the R1-R12 baseline entirely "
            "instead of appending to it. See this module's docstring "
            "(Sep 16 2026) for why f1qp.serving.data_fetch's on-demand "
            "fetch needs this. Mutually exclusive with --round."
        ),
    )
    args = parser.parse_args()
    if args.only_round is not None and args.extra_rounds:
        parser.error("--only-round cannot be combined with --round")
    return args


def main() -> None:
    args = parse_args()
    enable_cache()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reports = []

    if args.only_round is not None:
        rounds_to_download = [args.only_round]
    else:
        rounds_to_download = list(ROUNDS_TO_DOWNLOAD)
        for r in args.extra_rounds:
            if r not in rounds_to_download:
                rounds_to_download.append(r)

    for round_number in rounds_to_download:
        event = get_event_info(2026, round_number)
        out_path = RAW_DIR / f"r{round_number:02d}_{event.event_name.replace(' ', '_')}.parquet"
        if out_path.exists():
            logger.info("Skipping R%s (%s) - already cached at %s", round_number, event.event_name, out_path)
            continue

        role = "OFFLINE-TEST HOLDOUT" if round_number == OFFLINE_TEST_ROUND else "training"
        logger.info(
            "Loading R%s: %s (%s, %s)",
            round_number, event.event_name, "sprint" if event.is_sprint else "normal", role,
        )
        sessions = load_weekend(2026, round_number, telemetry=False)

        frames = []
        for loaded in sessions:
            laps = loaded.session.laps.copy()
            laps["SessionCode"] = loaded.session_code
            laps["Year"] = 2026
            laps["RoundNumber"] = round_number
            laps["IsSprint"] = event.is_sprint
            report = validate_laps(laps, year=2026, round_number=round_number, session_code=loaded.session_code)
            if not report.is_clean:
                logger.warning(
                    "Schema issue R%s %s: missing=%s notes=%s",
                    round_number, loaded.session_code, report.missing_columns, report.notes,
                )
            reports.append(report)
            frames.append(laps)

        pd.concat(frames, ignore_index=True).to_parquet(out_path)
        logger.info("Saved %s", out_path)

    logger.info("Done. %s sessions checked, %s with schema issues.", len(reports), sum(not r.is_clean for r in reports))


if __name__ == "__main__":
    main()
