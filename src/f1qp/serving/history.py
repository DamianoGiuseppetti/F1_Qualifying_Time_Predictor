"""Phase 5, UI redesign step: persists what the production model actually
predicted for a weekend at the moment someone launched it, so a later
screen can show that exact prediction next to the official result once
qualifying happens - and so a season/round-filterable "History" view has
real data to read instead of only the one-off Round 12 holdout evaluation
(`models/lstm/holdout_evaluation.csv`, a LEAVE-ONE-ROUND-OUT retrospective
score, not a launched live prediction).

**Why this exists (Damiano's requirement, Aug 26 2026 design round)**:
"After we launch prediction and it's done the page prediction must show
the prediction results as in history with the official results comparison.
We will have the opportunity to switch between the last weekend predicted
and the new one. Once the new prediction is launch than move the last one
to history and we will have the last to predict and the new one."

That needs a durable record of every LAUNCH (not every /predict call -
the dashboard may call /predict freely while someone is just looking; a
launch is the deliberate "run this for real" action), keyed by
(year, round_number), so:

  - the Prediction tab's switcher can show the two most recently launched
    rounds (by launched_at_utc) - one still awaiting quali, one already
    scored - without needing to re-run the model;
  - the History tab can filter by season/round and read back ANY past
    launch, scored against whatever official result now exists.

**Storage, deliberately simple**: one JSON file per (year, round_number)
under `LAUNCHES_DIR` (default `data/predictions/launches/`, override via
`F1QP_PREDICTIONS_DIR`) - filename `{year}_{round_number:02d}.json`. A
second "Re-run Prediction" launch for the same round OVERWRITES its file
(updating `launched_at_utc`) rather than creating a second record - re-
running a round that hasn't been scored yet is a correction, not a new
weekend, and there is exactly one "the current prediction for round N"
that makes sense to show. No database: this mirrors every other artifact
in this project (models/lstm/*.json, mlruns/) living as plain files under
`data/`/`models/`, and the launch volume is one file per Grand Prix
weekend - at most ~24/year, nowhere near needing anything heavier.

**Scoring is computed at READ time, never stored**: a launch file records
only what was PREDICTED (this module never mutates it after the fact).
Whether a round is "scored" depends on `qualifying_targets.parquet` having
real Q1/Q2/Q3 rows for it yet, which changes over time as a race weekend
actually happens - recomputing the join on every read means a round that
was "awaiting quali" this morning correctly shows as "scored" this
afternoon with zero extra bookkeeping, and there's no separate "did we
remember to backfill the result" step that can be forgotten.

**Scoring a not-yet-launched preview too (Aug 30 2026)**: Damiano asked for
the official-result comparison ("official position" column included) on
the Prediction page as well as History - not just for launched rounds.
`score_predictions()` below is the same join as `score_launch()`, exposed
for a raw (unlaunched) list of `DriverPrediction`s, so re-previewing an
already-happened round (e.g. re-checking Round 12) shows the same
official comparison the History tab would, without requiring a launch
first. Both now share one `_score_predictions_df()` implementation.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from f1qp.config import get_event_name
from f1qp.modeling.dataset import coalesce_final_quali_time, resolve_target_columns
from f1qp.serving import hf_sync
from f1qp.serving.predict import DriverPrediction

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHES_DIR = Path(
    os.environ.get("F1QP_PREDICTIONS_DIR", REPO_ROOT / "data" / "predictions" / "launches")
)
TARGETS_PATH = Path(
    os.environ.get("F1QP_TARGETS_PATH", REPO_ROOT / "data" / "processed" / "qualifying_targets.parquet")
)

MIN_SEASON = 2026
"""The earliest season the app will ever surface launched-prediction
history for (Damiano, Aug 30 2026: "history only for 2026 i don't want
to go back more than this). The model trains on 2023-2025 data too, but
that's training data, not launches - a LAUNCH only ever matters for the
live season the app is actually being used for. Enforced once, here, in
`list_launches` (the single read path every other function in this
module goes through) rather than in each caller separately, so a stray
pre-2026 launch file is invisible everywhere - History, the switcher,
everything - not just in whichever tab happens to remember to filter."""


@dataclass(frozen=True)
class LaunchRecord:
    """One launched prediction for one (year, round_number). `predictions`
    is stored as plain dicts (not `DriverPrediction` instances) so this
    round-trips through JSON without a custom decoder."""

    year: int
    round_number: int
    launched_at_utc: str
    model_trained_at_utc: str
    excluded_test_drivers: List[str] = field(default_factory=list)
    predictions: List[dict] = field(default_factory=list)


def _launch_path(year: int, round_number: int, launches_dir: Optional[Path] = None) -> Path:
    launches_dir = launches_dir or LAUNCHES_DIR
    return launches_dir / f"{year}_{round_number:02d}.json"


def record_launch(
    year: int,
    round_number: int,
    predictions: List[DriverPrediction],
    excluded_test_drivers: List[str],
    model_trained_at_utc: str,
    launched_at_utc: str,
    launches_dir: Optional[Path] = None,
) -> LaunchRecord:
    """Persist one launch, overwriting any prior launch for the same
    (year, round_number) - see module docstring on why a re-run replaces
    rather than accumulates. `launched_at_utc` is passed in (not stamped
    here with `datetime.utcnow()`) so callers - and tests - control it
    explicitly; the API layer is the one real caller and it stamps this
    itself at request time.
    """
    launches_dir = launches_dir or LAUNCHES_DIR
    launches_dir.mkdir(parents=True, exist_ok=True)
    record = LaunchRecord(
        year=year,
        round_number=round_number,
        launched_at_utc=launched_at_utc,
        model_trained_at_utc=model_trained_at_utc,
        excluded_test_drivers=list(excluded_test_drivers),
        predictions=[asdict(p) for p in predictions],
    )
    path = _launch_path(year, round_number, launches_dir)
    with open(path, "w") as f:
        json.dump(asdict(record), f, indent=2)
    # Sep 13 2026 addition: mirror this launch out to the Hugging Face
    # dataset repo (see f1qp.serving.hf_sync's module docstring) so it
    # survives a Space redeploy without needing Damiano's Mac to carry it
    # forward. A no-op locally (HF_DATASET_REPO/HF_TOKEN unset). Fired in
    # a background thread, not awaited here - a Launch should feel
    # instant; the durability sync is best-effort and can trail behind by
    # a second or two without anyone noticing.
    threading.Thread(
        target=hf_sync.push_file,
        args=(path, f"predictions/launches/{path.name}"),
        daemon=True,
    ).start()
    return record


def list_launches(launches_dir: Optional[Path] = None) -> List[LaunchRecord]:
    """Every launch on disk from `MIN_SEASON` onward, sorted by
    `launched_at_utc` ascending (oldest first) - a launch file from
    before `MIN_SEASON` (there was exactly one, a leftover test launch,
    cleaned up Aug 30 2026) is silently skipped rather than surfaced.
    Returns an empty list (never raises) if `launches_dir` doesn't exist
    yet - "no launches yet" is a normal, expected pre-season state, not
    an error, matching `dashboard._search_runs`'s own convention for the
    equally-empty-at-first MLflow store.
    """
    launches_dir = launches_dir or LAUNCHES_DIR
    if not launches_dir.exists():
        return []
    records = []
    for path in sorted(launches_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        record = LaunchRecord(**data)
        if record.year < MIN_SEASON:
            continue
        records.append(record)
    records.sort(key=lambda r: r.launched_at_utc)
    return records


def _official_results(targets_path: Optional[Path] = None) -> pd.DataFrame:
    """(Year, RoundNumber, Driver, final_quali_time, has_target) for every
    round `qualifying_targets.parquet` currently has rows for - reuses
    `coalesce_final_quali_time` (see its docstring) rather than
    re-deriving the Q3->Q2->Q1 rule. Returns an empty frame (never raises)
    if the targets file doesn't exist yet, or doesn't have this round -
    "not scored yet" is the normal pre-quali state a launch sits in.
    """
    targets_path = targets_path or TARGETS_PATH
    if not targets_path.exists():
        return pd.DataFrame(columns=["Year", "RoundNumber", "Driver", "final_quali_time", "has_target"])
    targets_df = pd.read_parquet(targets_path)
    tcols = resolve_target_columns(targets_df)
    return coalesce_final_quali_time(targets_df, tcols)


def _score_predictions_df(
    preds_df: pd.DataFrame, year: int, round_number: int, targets_path: Optional[Path] = None
) -> pd.DataFrame:
    """The actual join shared by `score_launch` (a persisted launch) and
    `score_predictions` (a raw, possibly-unlaunched prediction list) -
    one implementation of "match these driver predictions against
    whatever official result exists for (year, round_number)" so the two
    callers can never drift apart."""
    if preds_df.empty:
        return preds_df

    official_all = _official_results(targets_path)
    official = official_all[
        (official_all["Year"] == year) & (official_all["RoundNumber"] == round_number)
    ][["Driver", "final_quali_time", "has_target"]]

    merged = preds_df.merge(
        official, left_on="driver", right_on="Driver", how="left", indicator="_merge_indicator"
    ).drop(columns=["Driver"])
    # Sep 16 2026 fix (Damiano, Round 14/Madrid: "history keep showing the
    # previous view without the official results... it says awaiting for
    # official result" even after a successful "Check for official
    # result" fetch): whether THIS driver's row was actually found in
    # qualifying_targets.parquet at all - deliberately NOT the same thing
    # as `has_target` a few lines down. f1qp.modeling.dataset.
    # assemble_dataset documents has_target as false for a driver with NO
    # classified time at all (DNS/DSQ/crashed before a lap) - a normal,
    # PERMANENT per-driver outcome once that driver HAS been extracted,
    # completely different from a driver who hasn't been extracted yet at
    # all (mid-fetch, or the round hasn't happened). Round 14 is the
    # first case: BEA and STR both have a real row in qualifying_
    # targets.parquet (FastF1's session.results classified them, just
    # with no timed lap), so `has_target` is correctly False for them
    # forever, but `official_row_present` below is correctly True - the
    # extraction did account for them.
    official_row_present = merged["_merge_indicator"] == "both"
    merged = merged.drop(columns=["_merge_indicator"])
    merged["has_target"] = merged["has_target"].fillna(False)
    merged["abs_error_seconds"] = (merged["predicted_quali_time_seconds"] - merged["final_quali_time"]).abs()
    merged["within_interval"] = merged["has_target"] & (
        merged["final_quali_time"] >= merged["interval_low_seconds"]
    ) & (merged["final_quali_time"] <= merged["interval_high_seconds"])
    # Round-level (same value on every row): "has EVERY one of this
    # launch's drivers been accounted for by the official-result
    # extraction" - true once every predicted driver has a matched row
    # (whether or not that row itself has a time), false if even one
    # driver has no row yet at all (still awaiting/partial fetch). This -
    # not `has_target.all()` - is what `is_scored` and the History tab's
    # badge should use; see is_scored's own docstring.
    merged["official_results_available"] = bool(official_row_present.all())
    return merged


def score_launch(record: LaunchRecord, targets_path: Optional[Path] = None) -> pd.DataFrame:
    """Join one launch's predictions against the official result, where
    one now exists. Output has one row per predicted driver with columns:
    driver, era, is_sprint, n_practice_sessions,
    predicted_quali_time_seconds, interval_low/high/width_seconds,
    interval_level_pct, interval_exact, interval_label,
    final_quali_time (NaN if not yet scored), has_target (False if not yet
    scored), abs_error_seconds, within_interval - the same shape
    `models/lstm/holdout_evaluation.csv` already uses, so the History tab
    and the Prediction tab's "scored" view can share one rendering path.
    """
    preds_df = pd.DataFrame(record.predictions)
    return _score_predictions_df(preds_df, record.year, record.round_number, targets_path)


def score_predictions(
    year: int,
    round_number: int,
    predictions: List[DriverPrediction],
    targets_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Same join as `score_launch`, for a prediction list that hasn't
    been (or won't be) persisted as a launch - lets `/predict` (the
    "Preview, not saved" button) show the same official-result
    comparison the History tab shows for an already-launched round,
    per Damiano's Aug 30 2026 request ("official position... both in
    prediction page than history page"). Output shape is identical to
    `score_launch`'s.
    """
    preds_df = pd.DataFrame([asdict(p) for p in predictions])
    return _score_predictions_df(preds_df, year, round_number, targets_path)


def is_scored(record: LaunchRecord, targets_path: Optional[Path] = None) -> bool:
    """True once this round's official qualifying result has actually been
    extracted (see `_score_predictions_df`'s "official_results_available"
    comment for why this is NOT "every driver has has_target=True" - a
    driver who DNS'd/DSQ'd/crashed before a timed lap legitimately, and
    permanently, has has_target=False, and that must never block the round
    itself from ever showing as scored)."""
    scored_df = score_launch(record, targets_path)
    if scored_df.empty:
        return False
    return bool(scored_df["official_results_available"].iloc[0])


def latest_two_launches(
    launches_dir: Optional[Path] = None, targets_path: Optional[Path] = None
) -> List[dict]:
    """The Prediction tab's switcher data: up to the 2 most recently
    launched DISTINCT rounds (newest first), each with its scored table
    and `scored` flag attached. Exactly 2 states the mockup's switcher-bar
    needs: index 0 = the newest launch (usually still awaiting quali),
    index 1 = the previous one (usually already scored) - "move the last
    one to history" falls out for free here since History (below) already
    reads every launch on disk, not just these two.
    """
    records = list_launches(launches_dir)
    records.sort(key=lambda r: r.launched_at_utc, reverse=True)
    out = []
    for record in records[:2]:
        scored_df = score_launch(record, targets_path)
        out.append({
            "year": record.year,
            "round_number": record.round_number,
            "launched_at_utc": record.launched_at_utc,
            "model_trained_at_utc": record.model_trained_at_utc,
            "excluded_test_drivers": record.excluded_test_drivers,
            "scored": is_scored(record, targets_path),
            "rows": scored_df.to_dict(orient="records"),
        })
    return out


def prediction_history(
    season: Optional[int] = None,
    round_number: Optional[int] = None,
    launches_dir: Optional[Path] = None,
    targets_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Every launched round (optionally filtered to one season and/or one
    round number), each scored against whatever official result currently
    exists, concatenated with `year`/`round_number`/`event_name` columns
    attached - the History tab's season/round filter reads directly off
    this. A round still awaiting quali is included with `has_target=False`
    rows rather than being dropped, so a season view can show "not yet
    run" honestly instead of silently omitting it. `event_name` (Damiano,
    Aug 30 2026: "Add the name of the GP not only the round") is looked up
    once per round via `f1qp.config.get_event_name` and repeated onto
    every driver row for that round, same as `year`/`round_number` already
    are - the frontend reads a flat row list either way.
    """
    records = list_launches(launches_dir)
    if season is not None:
        records = [r for r in records if r.year == season]
    if round_number is not None:
        records = [r for r in records if r.round_number == round_number]

    frames = []
    for record in records:
        scored_df = score_launch(record, targets_path)
        if scored_df.empty:
            continue
        scored_df = scored_df.copy()
        scored_df.insert(0, "round_number", record.round_number)
        scored_df.insert(0, "year", record.year)
        scored_df["launched_at_utc"] = record.launched_at_utc
        scored_df["event_name"] = get_event_name(record.year, record.round_number)
        frames.append(scored_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
