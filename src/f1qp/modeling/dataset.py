"""Phase 3, steps 1-3: data assembly, weekend-level split, feature reduction.

Joins `features.parquet` (one row per driver x practice session) with
`qualifying_targets.parquet` (one row per driver x weekend, Q1/Q2/Q3
columns), builds an era-stratified train/val/holdout split at the WEEKEND
level (never the row level - driver rows from the same weekend share
session-level features, so a row-level split would leak), and reduces the
feature set before the XGBoost baseline and LSTM.

Column names are resolved defensively (see `resolve_feature_columns` /
`resolve_target_columns`) rather than hard-coded: this module was written
without the ability to inspect the real `features.parquet` /
`qualifying_targets.parquet` files directly, so a wrong name guess fails
loudly with the real column list instead of silently mis-joining. Run
`scripts/prepare_phase3_dataset.py` against the real files to confirm the
resolved names before trusting any downstream step, and see
`tests/test_dataset.py` for the behaviour this module is expected to have
on synthetic data matching the documented schema.

The gap-target formulation (`add_practice_reference_and_gaps`) produces a
target that is often negative - qualifying is reliably faster than any
practice lap (fresh tyres, low fuel, full push). That's real signal, not an
error. Never compute a percentage-error metric (MAPE) on the raw gap - it's
unstable/undefined near zero. Always reconstruct to absolute time first
(`gap + practice_reference`) and score that; `f1qp.modeling.baseline` does
this consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

RANDOM_SEED = 42
VAL_FRACTION = 0.2
CORRELATION_THRESHOLD = 0.9

# Zandvoort - Phase 5's offline test. Never used here for model selection.
HOLDOUT_YEAR = 2026
HOLDOUT_ROUND = 12

# Confirmed against the REAL features.parquet columns (Aug 23 2026 - see
# phase3-planning.md). Two names in docs/feature_engineering.md's 18-feature
# list didn't match the real build: "braking_events_per_lap" is really
# "braking_events", and "avg_corner_speed" doesn't exist as a corner-isolated
# metric - the real telemetry module only computed a whole-lap "avg_speed"
# (plus "max_speed", not in the original doc at all). That's a broader
# signal than the doc scoped (straight-line pace folded in, not just
# cornering) - kept anyway since it's the real available column, and this
# is exactly what the Step 3 correlation check + Step 4 importance ranking
# exist to arbitrate, not something to guess about upfront.
#
# Also added five real columns that exist in features.parquet but weren't
# in the doc's final enumeration: humidity_mean, wind_speed_mean (complete
# the weather group - the doc's own extract_weather.py pulls the full
# FastF1 weather frame, these just weren't in the final write-up),
# throttle_mean (a whole-lap throttle average alongside throttle_full_pct's
# full-throttle percentage), max_speed, and best_lap_number (raw lap number
# of the best lap - likely correlated with best_lap_session_position's
# normalized version; left for the correlation/importance steps to judge,
# not dropped on a guess).
ALL_FEATURE_COLS = [
    # Pace
    "best_lap_time", "gap_to_session_best", "median_flying_lap_time", "lap_time_std",
    # Run structure
    "n_runs", "n_flying_laps", "avg_run_length", "longest_run_length",
    # Tyre & fuel
    "compound_on_best_lap", "tyre_life_on_best_lap", "fuel_corrected_pace", "long_run_avg_pace",
    # Sector & speed
    "best_sector1_time", "best_sector2_time", "best_sector3_time",
    # Track evolution & conditions
    "best_lap_session_position", "best_lap_number",
    "air_temp_mean", "track_temp_mean", "rainfall_share", "humidity_mean", "wind_speed_mean",
    # Telemetry trend
    "throttle_full_pct", "throttle_mean", "braking_events", "avg_speed", "max_speed",
]

# Dropped now because it's provably identical to best_lap_time on the real
# data (confirmed Aug 23 2026: the Phase 2 fuel-burn reliability guardrail
# correctly rejected the estimate for every row, so fuel_corrected_pace ==
# best_lap_time everywhere) - not merely correlated with it, an exact
# duplicate. Revisit only if the Phase 2 fuel open item
# (docs/feature_engineering.md) gets resolved with a real correction.
DROPPED_FEATURES = ["fuel_corrected_pace"]

# Canonical practice-session order within a weekend, per
# docs/feature_engineering.md: "FP1/FP2/FP3 normal, FP1/SQ or FP1/SS
# sprint". Sprint weekends have only 2 real sessions; the 3rd slot is left
# as missing (NaN) when pivoting to a flat feature vector, not zero-filled
# - `is_sprint` tells the model that's an expected gap, not corrupt data.
SESSION_ORDER = {"FP1": 0, "FP2": 1, "FP3": 2, "SQ": 1, "SS": 1}


@dataclass(frozen=True)
class FeatureColumnMap:
    """Resolved `features.parquet` column names."""

    year: str
    round_number: str
    driver: str
    session: str
    is_sprint: str
    era: str


@dataclass(frozen=True)
class TargetColumnMap:
    """Resolved `qualifying_targets.parquet` column names."""

    year: str
    round_number: str
    driver: str
    q1: str
    q2: str
    q3: str


def resolve_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"Could not find a column for '{label}'. Tried {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def resolve_feature_columns(features_df: pd.DataFrame) -> FeatureColumnMap:
    return FeatureColumnMap(
        year=resolve_column(features_df, ["Year", "year", "Season", "season"], "Year"),
        round_number=resolve_column(
            features_df, ["RoundNumber", "Round", "round_number", "round"], "RoundNumber"
        ),
        driver=resolve_column(features_df, ["Driver", "driver", "Abbreviation"], "Driver"),
        session=resolve_column(
            features_df,
            ["SessionCode", "Session", "session_code", "session_type", "SessionType"],
            "SessionCode",
        ),
        is_sprint=resolve_column(features_df, ["IsSprint", "is_sprint"], "IsSprint"),
        era=resolve_column(features_df, ["Era", "era"], "Era"),
    )


def resolve_target_columns(targets_df: pd.DataFrame) -> TargetColumnMap:
    return TargetColumnMap(
        year=resolve_column(targets_df, ["Year", "year", "Season", "season"], "Year (targets)"),
        round_number=resolve_column(
            targets_df, ["RoundNumber", "Round", "round_number", "round"], "RoundNumber (targets)"
        ),
        driver=resolve_column(targets_df, ["Driver", "driver", "Abbreviation"], "Driver (targets)"),
        q1=resolve_column(targets_df, ["Q1", "q1"], "Q1"),
        q2=resolve_column(targets_df, ["Q2", "q2"], "Q2"),
        q3=resolve_column(targets_df, ["Q3", "q3"], "Q3"),
    )


def coalesce_final_quali_time(targets_df: pd.DataFrame, tcols: TargetColumnMap) -> pd.DataFrame:
    """Return `targets_df` reduced to (Year, RoundNumber, Driver,
    final_quali_time, has_target), with `final_quali_time` coalesced Q3 ->
    Q2 -> Q1 - the EXACT same rule `assemble_dataset` below uses (see its
    docstring for the "why": the segment a driver was actually classified
    in, matching how F1 itself scores an early elimination).

    Factored out (Phase 5, f1qp.serving.history) so anything that needs
    just the single official per-driver result - independent of building
    a full features-joined training frame - shares this one rule instead
    of re-deriving it. `assemble_dataset` itself is left untouched (its
    own inline version is already covered by tests/test_dataset.py); this
    is an additive helper, not a refactor of working code.
    """
    slim = targets_df[
        [tcols.year, tcols.round_number, tcols.driver, tcols.q1, tcols.q2, tcols.q3]
    ].rename(
        columns={
            tcols.year: "Year",
            tcols.round_number: "RoundNumber",
            tcols.driver: "Driver",
            tcols.q1: "Q1",
            tcols.q2: "Q2",
            tcols.q3: "Q3",
        }
    ).copy()
    final = slim["Q3"].where(slim["Q3"].notna(), slim["Q2"])
    final = final.where(final.notna(), slim["Q1"])
    slim["final_quali_time"] = final
    slim["has_target"] = final.notna()
    return slim[["Year", "RoundNumber", "Driver", "final_quali_time", "has_target"]]


def assemble_dataset(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    fcols: FeatureColumnMap,
    tcols: TargetColumnMap,
) -> pd.DataFrame:
    """Join features onto targets on (Year, RoundNumber, Driver).

    Deliberately many-to-one: every practice-session row for a driver that
    weekend gets the same Q1/Q2/Q3 label attached, since the sequence built
    from those practice sessions predicts toward that one label. Output
    columns are always named "Q1"/"Q2"/"Q3"/"has_Q1"/"has_Q2"/"has_Q3"
    regardless of the source casing, so downstream code never has to think
    about resolved names again. Missing Q2/Q3 is structural (the driver was
    eliminated earlier) - masked via has_Q2/has_Q3, never imputed.

    Also adds `final_quali_time` (decided Aug 23 2026, replacing the
    three-way Q1/Q2/Q3 masked-output design): the time that actually
    determined the driver's grid position - Q3 if they reached it,
    otherwise Q2, otherwise Q1. This is exactly how F1 itself classifies a
    driver eliminated in Q1 or Q2: by their time in the segment they were
    knocked out in. Modeling this one coalesced number instead of three
    separate segments removes the masking complexity entirely (every
    driver who took part in qualifying has exactly one final_quali_time,
    no has_Q2/has_Q3 branching needed downstream) while losing nothing real
    - a driver eliminated in Q1 never gets to show a Q3 pace in reality
    either, so there is no "true" Q2/Q3 time being discarded. `has_Q1` is
    reused as `has_target` (true unless the driver has no classified time
    at all - DNS/DSQ/no match). `reached_segment` records which segment
    produced the number, kept only as descriptive metadata - it must never
    be used as a model INPUT feature, since it is only known after
    qualifying happens (using it as a feature would leak the label).
    """
    targets_slim = targets_df[
        [tcols.year, tcols.round_number, tcols.driver, tcols.q1, tcols.q2, tcols.q3]
    ].rename(
        columns={
            tcols.year: fcols.year,
            tcols.round_number: fcols.round_number,
            tcols.driver: fcols.driver,
            tcols.q1: "Q1",
            tcols.q2: "Q2",
            tcols.q3: "Q3",
        }
    )

    merged = features_df.merge(
        targets_slim,
        on=[fcols.year, fcols.round_number, fcols.driver],
        how="left",
        validate="m:1",
    )
    merged["has_Q1"] = merged["Q1"].notna()
    merged["has_Q2"] = merged["Q2"].notna()
    merged["has_Q3"] = merged["Q3"].notna()

    merged["final_quali_time"] = merged["Q3"].where(merged["has_Q3"], merged["Q2"])
    merged["final_quali_time"] = merged["final_quali_time"].where(
        merged["final_quali_time"].notna(), merged["Q1"]
    )
    merged["has_target"] = merged["final_quali_time"].notna()
    merged["reached_segment"] = np.select(
        [merged["has_Q3"], merged["has_Q2"], merged["has_Q1"]],
        ["Q3", "Q2", "Q1"],
        default=None,
    )
    return merged


def build_weekend_split(
    merged: pd.DataFrame,
    fcols: FeatureColumnMap,
    holdout: tuple | None = (HOLDOUT_YEAR, HOLDOUT_ROUND),
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Add a 'split' column: 'train' / 'val' / 'holdout'.

    Split at the WEEKEND level (Year, RoundNumber), stratified by Era. No
    scikit-learn dependency (not in requirements.txt) - done by hand with a
    fixed seed for reproducibility.

    `holdout=None` (Phase 5 addition, Aug 25 2026): fold every weekend into
    the normal train/val split - no round is held out at all. This is the
    mode used for the FINAL pre-Round-13 retrain, once the Zandvoort offline
    test (scripts/evaluate_holdout.py) has already run against the default
    holdout-excluded dataset: Task_List.txt's Phase 5 line is explicit that
    the production model should retrain "on R1-R11 including Olanda" once
    that test is done, not hold Round 12 out forever. Wired via
    scripts/prepare_phase3_dataset.py's `--include-holdout` flag - see that
    script and scripts/evaluate_holdout.py's module docstring for the full
    sequencing. Passing any other tuple still holds out exactly that one
    weekend, unchanged from before this addition.
    """
    year_col, round_col, era_col = fcols.year, fcols.round_number, fcols.era
    weekends = merged[[year_col, round_col, era_col]].drop_duplicates().reset_index(drop=True)

    if holdout is None:
        is_holdout = pd.Series(False, index=weekends.index)
    else:
        holdout_year, holdout_round = holdout
        is_holdout = (weekends[year_col] == holdout_year) & (weekends[round_col] == holdout_round)
    holdout_weekends = weekends.loc[is_holdout].assign(split="holdout")
    trainval_weekends = weekends.loc[~is_holdout].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    split_labels = pd.Series("train", index=trainval_weekends.index)
    for _, group in trainval_weekends.groupby(era_col):
        idx = group.index.to_numpy().copy()  # .to_numpy() can be read-only; shuffle needs writable
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * val_fraction))
        split_labels.loc[idx[:n_val]] = "val"
    trainval_weekends = trainval_weekends.assign(split=split_labels)

    weekend_split = pd.concat([trainval_weekends, holdout_weekends], ignore_index=True)

    out = merged.merge(weekend_split[[year_col, round_col, "split"]], on=[year_col, round_col], how="left")
    assert out["split"].isna().sum() == 0, "Every row should have landed in train/val/holdout"
    return out


def _suggest_similar_columns(name: str, available: Sequence[str], max_suggestions: int = 4) -> list:
    """Cheap fuzzy-match helper for the get_feature_columns error message:
    tokens of `name` (split on '_') that appear in a real column name,
    scored by overlap. Not exact - just enough to point at the likely real
    name instead of leaving a bare 'not found'."""
    name_tokens = set(name.lower().split("_"))
    scored = []
    for col in available:
        col_tokens = set(col.lower().replace("-", "_").split("_"))
        overlap = len(name_tokens & col_tokens)
        if overlap:
            scored.append((overlap, col))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [col for _, col in scored[:max_suggestions]]


def get_feature_columns(
    df: pd.DataFrame,
    all_features: Sequence[str] = ALL_FEATURE_COLS,
    dropped: Sequence[str] = DROPPED_FEATURES,
) -> list:
    missing = [c for c in all_features if c not in df.columns]
    if missing:
        suggestions = {name: _suggest_similar_columns(name, df.columns) for name in missing}
        detail = "; ".join(
            f"'{name}' not found (closest real columns: {suggestions[name] or 'none obvious'})"
            for name in missing
        )
        raise KeyError(
            f"Expected feature columns not found - {detail}. Check "
            f"f1qp/features/build.py or f1qp/features/telemetry.py for the "
            f"real names and update ALL_FEATURE_COLS in "
            f"f1qp/modeling/dataset.py accordingly."
        )
    return [c for c in all_features if c not in dropped]


def compute_correlation_matrix(
    df: pd.DataFrame, feature_cols: Sequence[str], train_mask: pd.Series
) -> pd.DataFrame:
    return df.loc[train_mask, list(feature_cols)].astype(float).corr(method="pearson")


def find_high_correlation_pairs(
    corr_matrix: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD
) -> pd.DataFrame:
    cols = corr_matrix.columns.tolist()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append((cols[i], cols[j], r))
    pairs_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "pearson_r"])
    if len(pairs_df):
        pairs_df = pairs_df.sort_values("pearson_r", key=abs, ascending=False).reset_index(drop=True)
    return pairs_df


def add_practice_reference_and_gaps(df: pd.DataFrame, fcols: FeatureColumnMap) -> pd.DataFrame:
    """Add `practice_reference`, `gap_Q1`/`gap_Q2`/`gap_Q3` (kept for
    reference/diagnostics), and `gap_final` - the primary modeling target,
    computed from `final_quali_time`.

    `practice_reference` is the fastest practice lap posted by anyone that
    weekend - known before Q starts, so legitimate to use at inference
    time (not leakage). Each gap is NaN wherever its mask is False (never
    imputed).
    """
    year_col, round_col = fcols.year, fcols.round_number
    reference = (
        df.groupby([year_col, round_col])["best_lap_time"]
        .min()
        .rename("practice_reference")
        .reset_index()
    )
    out = df.merge(reference, on=[year_col, round_col], how="left")
    for seg in ("Q1", "Q2", "Q3"):
        gap = out[seg] - out["practice_reference"]
        out[f"gap_{seg}"] = gap.where(out[f"has_{seg}"])
    gap_final = out["final_quali_time"] - out["practice_reference"]
    out["gap_final"] = gap_final.where(out["has_target"])
    return out


def pivot_to_weekend_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    fcols: FeatureColumnMap,
    session_order: Dict[str, int] = SESSION_ORDER,
) -> pd.DataFrame:
    """Flatten the long (driver, practice session) table to one row per
    (Year, RoundNumber, Driver), with columns `session{0,1,2}_<feature>`.

    Sessions are ordered by position in the weekend (FP1 first, then
    FP2/SQ, then FP3/SS), not by literal label, since sprint weekends only
    have 2 real sessions. The missing 3rd slot on sprint weekends is left
    as NaN - XGBoost (and the LSTM later) handle missing values / an
    explicit padding indicator rather than a silent zero-fill, per the
    Phase 1 design decision. `is_sprint` tells the model the gap is
    expected, not corrupt data.

    This is the flat analogue of the LSTM's sequence input - same
    information, reshaped for a non-sequential baseline model.
    """
    session_col = fcols.session
    unknown = set(df[session_col].dropna().unique()) - set(session_order)
    if unknown:
        raise KeyError(
            f"Unrecognised session code(s) {sorted(unknown)} - extend "
            f"SESSION_ORDER in f1qp/modeling/dataset.py before pivoting."
        )

    work = df.copy()
    work["_session_slot"] = work[session_col].map(session_order)

    # Built via explicit per-slot merges, NOT pivot_table/unstack: both of
    # those reshape via a MultiIndex product and can silently manufacture
    # rows for (year, round, driver) combinations that never occurred
    # together in the data (caught by tests/test_dataset.py - an earlier
    # pivot_table attempt with dropna=False produced 36 rows for 7
    # weekends x 3 drivers = 21 real driver-weekends, inventing entries
    # like (2023, round 12) which doesn't exist, because round 12 and year
    # 2023 each individually appear elsewhere in the index). An outer merge
    # on the real (Year, RoundNumber, Driver) key can't invent combinations
    # that were never present on either side.
    id_cols = [fcols.year, fcols.round_number, fcols.driver]

    slot_frames = []
    for slot in sorted(set(session_order.values())):
        slot_df = work.loc[work["_session_slot"] == slot, id_cols + list(feature_cols)].copy()
        slot_df = slot_df.rename(columns={feat: f"session{slot}_{feat}" for feat in feature_cols})
        slot_frames.append(slot_df)

    wide = slot_frames[0]
    for other in slot_frames[1:]:
        wide = wide.merge(other, on=id_cols, how="outer")

    passthrough_cols = [
        c for c in [
            fcols.is_sprint, fcols.era, "Q1", "Q2", "Q3", "has_Q1", "has_Q2", "has_Q3",
            "final_quali_time", "has_target", "reached_segment",
            "practice_reference", "gap_Q1", "gap_Q2", "gap_Q3", "gap_final", "split",
        ]
        if c in work.columns
    ]
    passthrough = work[id_cols + passthrough_cols].drop_duplicates(subset=id_cols)
    wide = wide.merge(passthrough, on=id_cols, how="left")
    return wide
