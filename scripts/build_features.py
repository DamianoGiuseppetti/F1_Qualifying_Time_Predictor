"""Phase 2: assemble the final per-(Driver, Session) feature table.

Pure local computation - no network, no fastf1 session objects, just the
raw laps already on disk plus whatever extract_weather.py /
extract_telemetry.py / extract_qualifying_targets.py have produced so far.
Run this last, but it's safe to run before the other three scripts finish:
missing weather/telemetry just means NaN columns for now (join_weather_features
/ join_telemetry_features handle that), not a crash - re-run this script
after those catch up rather than treating this as a hard dependency order.

Practice sessions only (FP1-FP3, or FP1+SQ/SS on sprint weekends) go into
features.parquet - Q laps are never a model input, only the source of
targets in qualifying_targets.parquet (a separate join for Phase 3, not
done here).

**--round N (Aug 30 2026)**: `_events()` was hardcoded to the training
seasons + Round 12 only, so a brand new round - Round 13 (Monza) onward -
had no feature rows built for it even after download_2026.py fetched its
raw laps. Pass the same round(s) you downloaded (repeat --round for more
than one).

**Carry-forward for missing raw laps (Sep 13 2026)**: this script used to
rebuild `features.parquet` from scratch every run, using ONLY whatever
raw laps happen to be under `data/raw/` right now - a weekend with no raw
laps on disk was just silently dropped from the output. That's fine on a
full local checkout (every season's raw laps are there), but it silently
DESTROYED every earlier weekend's rows the one time this ran somewhere
raw laps aren't fully present: a Hugging Face Space, which never bakes in
the (huge, training-only - see Task_List.txt's Sep 13 2026 "final
version" analysis) raw cache, only the already-built features.parquet.
Running this there via the app's own on-demand fetch (f1qp.serving.
data_fetch, `--round <just-fetched-round>`) would wipe every other
weekend's features on the very first click. `_carry_forward_missing`
below fixes this: any (Year, RoundNumber) this run couldn't rebuild (no
raw laps found) is now carried forward from whatever `features.parquet`
already had, instead of being dropped. A no-op on a full local checkout
(nothing is ever missing there, so nothing is ever carried forward) -
this only changes behavior in exactly the partial-raw-data case above.

**Cached fuel-burn factor (Sep 16 2026)**: `_learn_fuel_factor` used to
hard-require R1-R11's raw practice laps on disk, unconditionally, even
though the carry-forward fix above already established that a deployment
running this on-demand may have NONE of them - it would raise
RuntimeError and the whole fetch job would fail. Worse, once
f1qp.serving.data_fetch's on-demand fetch was fixed (Sep 16 2026, see
download_2026.py's own docstring) to stop needlessly re-downloading all
of R1-R12 just to reach one new round, R1-R11's raw laps are now
deliberately NEVER on disk in a deployed container - so this would always
raise there. Damiano's own Round 14 test found this stage taking 10+
minutes and eventually timing out BEFORE that download fix, because the
300-iteration cluster bootstrap (f1qp.features.fuel.learn_fuel_burn_factor)
was re-run from scratch, every fetch, over a full freshly-redownloaded
season - real CPU-bound work that's slow on a free-tier instance.
`FUEL_FACTOR_CACHE_PATH` fixes both: whenever R1-R11 raw laps ARE present
(a real local run with the full season on disk), the freshly-computed
factor is cached to disk after computing it, same as always; whenever
they're NOT present (every on-demand deploy fetch from now on), this
falls back to that cached value instead of recomputing or raising - it's
refreshed by whatever `build_features.py` full run most recently ran
locally, and rides into every image rebuild since it lives in
data/processed/ alongside features.parquet.

Run from the repo root, after at least download_2026.py / download_historical.py:
    python scripts/build_features.py
    python scripts/build_features.py --round 13
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from f1qp.config import OFFLINE_TEST_ROUND, TRAINING_SEASONS, all_training_events, get_event_info
from f1qp.features.build import build_session_driver_features, join_telemetry_features, join_weather_features
from f1qp.features.fuel import learn_fuel_burn_factor
from f1qp.features.runs import add_run_features
from f1qp.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "features.parquet"
# Sep 16 2026 addition - see module docstring ("Cached fuel-burn factor").
FUEL_FACTOR_CACHE_PATH = PROCESSED_DIR / "fuel_burn_factor.json"


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
            "Additional 2026 round number to build features for, beyond "
            "the default R1-R12 set (repeat --round for more than one) - "
            "pass the same round(s) you gave download_2026.py --round."
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


def _laps_for(year: int, round_number: int) -> pd.DataFrame | None:
    year_dir = RAW_DIR / str(year)
    matches = list(year_dir.glob(f"r{round_number:02d}_*.parquet"))
    if not matches:
        return None
    return pd.read_parquet(matches[0])


def _era(year: int) -> int:
    return 0 if year in TRAINING_SEASONS else 1  # 2023-2025 = 0 (pre-2026 regs), 2026 = 1


def _load_cached_fuel_factor() -> float | None:
    """See module docstring ("Cached fuel-burn factor"). Returns None if no
    cache exists yet (the first-ever-run case, local or deployed - nothing
    to fall back to, so the caller still raises as before)."""
    if not FUEL_FACTOR_CACHE_PATH.exists():
        return None
    with open(FUEL_FACTOR_CACHE_PATH) as f:
        cached = json.load(f)
    logger.info(
        "No 2026 practice laps on disk - using the cached fuel burn factor from %s "
        "(%.4f s/lap, computed %s from a full local run).",
        FUEL_FACTOR_CACHE_PATH, cached["effective_seconds_per_lap"], cached["computed_at_utc"],
    )
    return cached["effective_seconds_per_lap"]


def _cache_fuel_factor(result) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(FUEL_FACTOR_CACHE_PATH, "w") as f:
        json.dump(
            {
                "computed_at_utc": datetime.now(timezone.utc).isoformat(),
                "seconds_per_lap": result.seconds_per_lap,
                "effective_seconds_per_lap": result.effective_seconds_per_lap,
                "is_reliable": result.is_reliable,
                "n_laps_used": result.n_laps_used,
                "n_runs_used": result.n_runs_used,
                "n_sessions_used": result.n_sessions_used,
                "note": (
                    "Fallback value read by _load_cached_fuel_factor whenever R1-R11's "
                    "raw practice laps aren't on disk (every deployed on-demand fetch - "
                    "see this module's docstring). Refreshed here every time this script "
                    "runs with the full 2026 season's raw laps actually present."
                ),
            },
            f,
            indent=2,
        )


def _learn_fuel_factor() -> float:
    logger.info("Learning fuel burn factor from 2026 practice data...")
    frames = []
    for round_number in range(1, 12):
        laps = _laps_for(2026, round_number)
        if laps is None:
            continue
        practice = laps[laps["SessionCode"].isin(["FP1", "FP2", "FP3"])]
        if practice.empty:
            continue
        for _, grp in practice.groupby("SessionCode"):
            frames.append(add_run_features(grp))
    if not frames:
        cached = _load_cached_fuel_factor()
        if cached is not None:
            return cached
        raise RuntimeError(
            "No 2026 practice laps found on disk and no cached fuel_burn_factor.json either - "
            "run download_2026.py first (or scripts/build_features.py at least once with the "
            "full 2026 season's raw laps present, to populate the cache)."
        )
    pooled = pd.concat(frames, ignore_index=True)
    result = learn_fuel_burn_factor(pooled)
    logger.info(
        "Fuel burn factor: %.4f s/lap (se=%.4f, t=%.2f, bootstrap sign consistency=%.0f%%) "
        "from %s laps across %s runs / %s sessions - reliable=%s",
        result.seconds_per_lap, result.standard_error, result.t_stat,
        result.bootstrap_sign_consistency * 100, result.n_laps_used, result.n_runs_used,
        result.n_sessions_used, result.is_reliable,
    )
    if not result.is_reliable:
        logger.warning(
            "Fuel burn estimate did NOT pass the reliability guardrail (see f1qp.features.fuel's "
            "docstring for the thresholds and why) - using 0.0 s/lap instead, i.e. fuel_corrected_pace "
            "will equal the raw best lap time for every row rather than applying an untrustworthy "
            "correction. Re-run once more 2026 rounds are available if you want this feature populated."
        )
    _cache_fuel_factor(result)
    return result.effective_seconds_per_lap


def _load_optional(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning(
            "%s not found at %s - features will carry NaN %s columns until that script has run.",
            label, path, label.lower(),
        )
        return None
    return pd.read_parquet(path)


def _carry_forward_missing(rebuilt: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    """See module docstring ("Carry-forward for missing raw laps"). Any
    (Year, RoundNumber) present in `existing` but NOT in `rebuilt` (this
    run couldn't find raw laps for it) is appended from `existing` as-is.
    A (Year, RoundNumber) that WAS just rebuilt always wins outright over
    whatever it had before - this is a real refresh for that weekend, not
    a gap to fill. `existing` being None or empty is the ordinary
    first-ever-run state - just returns `rebuilt` unchanged."""
    if existing is None or existing.empty:
        return rebuilt
    rebuilt_keys = set(zip(rebuilt["Year"], rebuilt["RoundNumber"])) if not rebuilt.empty else set()
    carried = existing[
        ~existing.apply(lambda r: (r["Year"], r["RoundNumber"]) in rebuilt_keys, axis=1)
    ]
    if carried.empty:
        return rebuilt
    if rebuilt.empty:
        return carried
    return pd.concat([carried, rebuilt], ignore_index=True)


def main() -> None:
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fuel_burn_seconds_per_lap = _learn_fuel_factor()

    weather_df = _load_optional(PROCESSED_DIR / "weather.parquet", "Weather")
    telemetry_df = _load_optional(PROCESSED_DIR / "telemetry.parquet", "Telemetry")
    # Sep 13 2026 addition - see module docstring and _carry_forward_missing.
    # Not existing yet is the normal first-ever-run state, not warning-worthy,
    # so this reads OUT_PATH directly rather than going through
    # _load_optional (whose "not found" warning is written for
    # weather/telemetry, not this).
    existing_df = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else None

    all_rows = []
    missing_raw = []
    for year, round_number in _events(args.extra_rounds):
        laps = _laps_for(year, round_number)
        if laps is None:
            missing_raw.append((year, round_number))
            continue
        event = get_event_info(year, round_number)

        for session_code in event.practice_sessions:
            session_laps = laps[laps["SessionCode"] == session_code]
            feats = build_session_driver_features(
                session_laps, fuel_burn_seconds_per_lap=fuel_burn_seconds_per_lap
            )
            if feats.empty:
                continue

            weather_row = None
            if weather_df is not None:
                match = weather_df[
                    (weather_df["Year"] == year)
                    & (weather_df["RoundNumber"] == round_number)
                    & (weather_df["SessionCode"] == session_code)
                ]
                if not match.empty:
                    weather_row = match.iloc[0].to_dict()
            feats = join_weather_features(feats, weather_row)

            session_telemetry = None
            if telemetry_df is not None:
                session_telemetry = telemetry_df[
                    (telemetry_df["Year"] == year)
                    & (telemetry_df["RoundNumber"] == round_number)
                    & (telemetry_df["SessionCode"] == session_code)
                ]
            feats = join_telemetry_features(feats, session_telemetry)

            feats["Year"] = year
            feats["RoundNumber"] = round_number
            feats["SessionCode"] = session_code
            feats["IsSprint"] = event.is_sprint
            feats["Era"] = _era(year)
            all_rows.append(feats)

    if not all_rows and (existing_df is None or existing_df.empty):
        raise RuntimeError("Produced zero feature rows - check data/raw/ has downloaded files.")

    rebuilt = (
        pd.concat(all_rows, ignore_index=True)
        if all_rows
        else pd.DataFrame(columns=(existing_df.columns if existing_df is not None else None))
    )
    out = _carry_forward_missing(rebuilt, existing_df)
    out.to_parquet(OUT_PATH)
    logger.info("Wrote %s - %s (driver, session) rows across %s weekends (%s rows freshly rebuilt this run).",
                OUT_PATH, len(out), out.groupby(["Year", "RoundNumber"]).ngroups, len(rebuilt))
    if missing_raw:
        carried_keys = set(zip(out["Year"], out["RoundNumber"])) - (
            set(zip(rebuilt["Year"], rebuilt["RoundNumber"])) if not rebuilt.empty else set()
        )
        logger.warning(
            "No raw laps on disk for %s weekend(s) this run - %s of them were carried forward "
            "unchanged from the existing %s, the rest have no rows at all yet:",
            len(missing_raw), len(carried_keys), OUT_PATH,
        )
        for year, round_number in missing_raw:
            tag = "carried forward" if (year, round_number) in carried_keys else "NO ROWS YET"
            logger.warning("  - %s R%02d (%s)", year, round_number, tag)


if __name__ == "__main__":
    main()
