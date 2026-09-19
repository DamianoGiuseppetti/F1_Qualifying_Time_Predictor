"""Shared synthetic-data fixtures for f1qp.modeling tests.

Builds a features.parquet / qualifying_targets.parquet pair matching the
documented schema (docs/feature_engineering.md, HANDOVER.md) - this was
written without being able to inspect the real parquet files directly, so
these fixtures exercise the modeling code against a schema built from the
documentation, not the real files. Re-run scripts/prepare_phase3_dataset.py
against the real files to confirm the resolved column names before
trusting output from f1qp.modeling on real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1qp.modeling.dataset import (
    ALL_FEATURE_COLS,
    add_practice_reference_and_gaps,
    assemble_dataset,
    build_weekend_split,
    get_feature_columns,
    pivot_to_weekend_features,
    resolve_feature_columns,
    resolve_target_columns,
)

DRIVERS = ["VER", "HAM", "LEC"]

# (year, round, era, is_sprint, best_lap_time per driver) - small set, used
# by tests/test_dataset.py where exact row counts are asserted by hand.
WEEKENDS = [
    (2023, 1, 0, False, {"VER": 90.0, "HAM": 90.5, "LEC": 90.2}),
    (2023, 2, 0, False, {"VER": 91.0, "HAM": 91.8, "LEC": 91.1}),
    (2024, 1, 0, False, {"VER": 80.0, "HAM": 80.4, "LEC": 80.9}),
    (2024, 2, 0, True, {"VER": 70.0, "HAM": 70.6, "LEC": 70.3}),  # sprint weekend
    (2025, 1, 1, False, {"VER": 95.0, "HAM": 95.9, "LEC": 95.2}),
    (2025, 2, 1, False, {"VER": 96.0, "HAM": 96.3, "LEC": 96.5}),
    (2026, 12, 1, False, {"VER": 88.0, "HAM": 88.6, "LEC": 88.9}),  # holdout weekend
]

# Larger set - enough weekends per era for XGBoost to fit on a non-trivial
# train split, used by tests/test_baseline.py.
MANY_WEEKENDS = (
    [(2023, r, 0, False, {"VER": 90.0 + r, "HAM": 90.5 + r, "LEC": 90.2 + r}) for r in range(1, 9)]
    + [(2025, r, 1, False, {"VER": 95.0 + r, "HAM": 95.9 + r, "LEC": 95.2 + r}) for r in range(1, 9)]
    + [(2026, 12, 1, False, {"VER": 88.0, "HAM": 88.6, "LEC": 88.9})]  # holdout
)


def make_features_df(weekends) -> pd.DataFrame:
    rows = []
    for year, round_num, era, is_sprint, best_laps in weekends:
        sessions = ["FP1", "SQ"] if is_sprint else ["FP1", "FP2", "FP3"]
        for driver in DRIVERS:
            for s_idx, session in enumerate(sessions):
                row = {
                    "Year": year,
                    "RoundNumber": round_num,
                    "Driver": driver,
                    "SessionCode": session,
                    "IsSprint": is_sprint,
                    "Era": era,
                }
                for feat in ALL_FEATURE_COLS:
                    if feat in ("best_lap_time", "fuel_corrected_pace"):
                        # fuel_corrected_pace is identical to best_lap_time
                        # on the real data too (Phase 2 guardrail outcome).
                        row[feat] = best_laps[driver] + s_idx * 0.2
                    else:
                        row[feat] = float(abs(hash((year, round_num, driver, session, feat))) % 50)
                rows.append(row)
    return pd.DataFrame(rows)


def make_targets_df(weekends) -> pd.DataFrame:
    rows = []
    for year, round_num, _era, _is_sprint, best_laps in weekends:
        for i, driver in enumerate(DRIVERS):
            q1 = best_laps[driver] - 1.0  # Q is faster than the best practice lap
            q2 = q1 - 0.3 if i < 2 else np.nan  # only 2 of 3 drivers reach Q2
            q3 = q2 - 0.2 if i < 1 else np.nan  # only 1 of 3 drivers reaches Q3
            rows.append(
                {"Year": year, "RoundNumber": round_num, "Driver": driver, "Q1": q1, "Q2": q2, "Q3": q3}
            )
    return pd.DataFrame(rows)


@pytest.fixture
def features_df():
    return make_features_df(WEEKENDS)


@pytest.fixture
def targets_df():
    return make_targets_df(WEEKENDS)


def _build_many_weekends_wide_df():
    features = make_features_df(MANY_WEEKENDS)
    targets = make_targets_df(MANY_WEEKENDS)
    fcols = resolve_feature_columns(features)
    tcols = resolve_target_columns(targets)

    merged = assemble_dataset(features, targets, fcols, tcols)
    merged = build_weekend_split(merged, fcols, val_fraction=0.3, seed=0)
    feature_cols = get_feature_columns(merged)
    merged = add_practice_reference_and_gaps(merged, fcols)
    wide_df = pivot_to_weekend_features(merged, feature_cols, fcols)
    return wide_df, feature_cols, fcols


@pytest.fixture
def wide_df_and_feature_cols():
    wide_df, feature_cols, fcols = _build_many_weekends_wide_df()
    wide_df = wide_df[wide_df["split"] != "holdout"].reset_index(drop=True)
    return wide_df, feature_cols, fcols


@pytest.fixture
def wide_df_with_holdout_and_feature_cols():
    """Same as wide_df_and_feature_cols but keeps the holdout (Zandvoort)
    rows in - for tests that specifically check holdout exclusion logic."""
    return _build_many_weekends_wide_df()
