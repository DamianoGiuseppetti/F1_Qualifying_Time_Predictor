"""Phase 4, API step: run the production LSTM against one already-featurized
event and return a point prediction + the shipped conformal interval per
driver.

Deliberately reuses the exact training-time pipeline pieces
(`pivot_to_weekend_features`, `build_lstm_sequences`, `FeatureImputer`,
`FeatureScaler`, `build_interval`) rather than re-deriving a parallel
inference-only version of any of them - the whole point of an inference
endpoint is that it sees data through IDENTICAL preprocessing to what the
model was trained on. A second hand-written implementation of the same
pivot/scale logic here would be exactly the kind of thing that quietly
drifts out of sync the next time the training-side pipeline changes.

Only two things are genuinely new to this module, both because nothing else
in the project needed them before there was a live inference call to serve:

1. Loading the saved artifacts (`lstm_final_production.pt`,
   `final_preprocessing.npz`, `final_model_metadata.json`,
   `conformal_intervals.json`) back into memory - the WRITE side of these
   four files already exists (train_final_lstm.py / calibrate_conformal.py /
   retrain_pipeline.py); `load_production_artifacts` is the read side.
2. Building a wide/sequence input for a weekend that has NO qualifying
   result yet (has_target=False everywhere, by construction - that's the
   whole point of predicting it). `build_lstm_sequences` still expects
   'final_quali_time' / 'gap_final' / 'has_target' / 'split' columns to
   exist on the wide frame (it reads them for supervision/diagnostics on
   the training path), so `predict_event` adds them as inert placeholders
   before calling it. None of the four is ever read for anything at
   inference time - the model only ever sees X / mask / lengths / static.

**Input contract** (decided via AskUserQuestion, Aug 24 2026): the caller
supplies an already-built features slice - same schema as
`data/processed/features.parquet` - for exactly one (year, round_number).
Nothing here downloads or computes features; that stays
`scripts/download_2026.py` + `scripts/build_features.py`'s job, run before
a prediction is ever requested. This keeps inference fast (no FastF1 /
network call in the request path - the <500ms latency target from
Task_List.txt would be unreachable otherwise) and keeps this module honest
about not silently re-deriving data it was never given.

**Partial-weekend predictions fall out for free**: if only FP1 (or FP1+FP2)
has run so far, `event_features_df` simply won't have session2 (or
session1+session2) rows yet - `pivot_to_weekend_features` leaves those
columns NaN for every driver, exactly like a sprint weekend's structurally
missing 3rd session, and `build_lstm_sequences` masks them out the same
way. Not a special case written for it - it's the same mechanism the
sprint-weekend handling already needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from f1qp.modeling.conformal import build_interval
from f1qp.modeling.dataset import pivot_to_weekend_features, resolve_feature_columns
from f1qp.modeling.lstm_model import QualifyingLSTM
from f1qp.modeling.sequences import FeatureImputer, FeatureScaler, build_lstm_sequences

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = Path(os.environ.get("F1QP_MODELS_DIR", REPO_ROOT / "models" / "lstm"))
FEATURES_PATH = Path(
    os.environ.get("F1QP_FEATURES_PATH", REPO_ROOT / "data" / "processed" / "features.parquet")
)

# Same alpha the Phase 3 AskUserQuestion decision shipped for deployment -
# see f1qp.modeling.conformal's module docstring and
# scripts/retrain_pipeline.py's ALPHA_LEVELS ([0.50, 0.32, 0.20, 0.10] =
# 50/68/80/90% coverage). json.dump writes the dict key as str(0.50) ==
# "0.5" (Python's own float->str, not a formatted literal) - matched here
# and in scripts/retrain_pipeline.py's _load_previous_summary the same way.
SHIPPED_ALPHA = 0.50
SHIPPED_INTERVAL_LABEL = "typical range"  # not "confidence interval" - Damiano's Phase 3 decision


class ArtifactsNotFoundError(RuntimeError):
    """Raised when a production artifact file is missing from models/lstm/,
    or the conformal file doesn't yet have the shipped alpha's era_1 entry."""


class EventNotFoundError(ValueError):
    """Raised when the requested (year, round_number) has no rows in features.parquet."""


@dataclass(frozen=True)
class ProductionArtifacts:
    """Everything loaded once at API startup and reused across requests -
    reloading these from disk on every request would be the obvious way to
    blow the <500ms latency target for no reason, since none of it changes
    between requests (only a retrain - a separate manual step - changes it).
    """

    model: QualifyingLSTM
    imputer: FeatureImputer
    scaler: FeatureScaler
    feature_cols: List[str]
    metadata: dict
    deployment_quantile_seconds: float
    deployment_quantile_exact: bool
    coverage_target_pct: float


def _require(path: Path) -> Path:
    if not path.exists():
        raise ArtifactsNotFoundError(
            f"Missing production artifact: {path}. Run "
            f"scripts/train_final_lstm.py (or scripts/retrain_pipeline.py) "
            f"before starting the API."
        )
    return path


def load_production_artifacts(models_dir: Path | None = None) -> ProductionArtifacts:
    """Load the model weights + preprocessing stats + metadata + the
    shipped conformal interval from `models_dir` (defaults to the module-
    level `MODELS_DIR`, read at CALL time rather than bound as a default
    argument value - a default bound at import time can't be monkeypatched
    by tests that override the module-level path). Raises
    `ArtifactsNotFoundError` with a clear, actionable message if any of the
    four files is missing (or the conformal file lacks the shipped alpha's
    era_1 entry), rather than letting torch/json raise a generic
    FileNotFoundError/KeyError deep in a stack trace.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    model_path = _require(models_dir / "lstm_final_production.pt")
    preprocessing_path = _require(models_dir / "final_preprocessing.npz")
    metadata_path = _require(models_dir / "final_model_metadata.json")
    conformal_path = _require(models_dir / "conformal_intervals.json")

    preprocessing = np.load(preprocessing_path, allow_pickle=False)
    feature_cols = [str(c) for c in preprocessing["feature_cols"]]
    imputer = FeatureImputer(median=preprocessing["imputer_median"])
    scaler = FeatureScaler(mean=preprocessing["scaler_mean"], std=preprocessing["scaler_std"])

    model = QualifyingLSTM(n_features=len(feature_cols))
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(conformal_path) as f:
        conformal = json.load(f)

    alpha_block = conformal.get("alphas", {}).get(str(SHIPPED_ALPHA), {})
    era1 = alpha_block.get("era_1", {})
    deployment_quantile_seconds = era1.get("deployment_quantile_seconds")
    if deployment_quantile_seconds is None:
        raise ArtifactsNotFoundError(
            f"{conformal_path} has no alphas['{SHIPPED_ALPHA}']['era_1']"
            f"['deployment_quantile_seconds'] - run "
            f"scripts/calibrate_conformal.py (or scripts/retrain_pipeline.py) "
            f"before starting the API."
        )

    return ProductionArtifacts(
        model=model,
        imputer=imputer,
        scaler=scaler,
        feature_cols=feature_cols,
        metadata=metadata,
        deployment_quantile_seconds=float(deployment_quantile_seconds),
        deployment_quantile_exact=bool(era1.get("deployment_quantile_exact")),
        coverage_target_pct=float(alpha_block.get("coverage_target_pct", (1 - SHIPPED_ALPHA) * 100)),
    )


def load_features_for_event(
    year: int, round_number: int, features_path: Path | None = None
) -> pd.DataFrame:
    """Read `data/processed/features.parquet` and slice it to one weekend.

    Raises `EventNotFoundError` (not a silent empty DataFrame) if the
    weekend isn't in the file yet - the caller almost certainly wants to
    know "run download_2026.py + build_features.py for this round first",
    not get an empty prediction list back with no explanation.

    `features_path` defaults to the module-level `FEATURES_PATH`, read at
    CALL time (same reasoning as `load_production_artifacts`'s `models_dir`
    - keeps this patchable in tests without editing a bound default).
    """
    if features_path is None:
        features_path = FEATURES_PATH
    if not features_path.exists():
        raise ArtifactsNotFoundError(
            f"{features_path} does not exist. Run scripts/download_2026.py "
            f"and scripts/build_features.py for this event first."
        )
    features_df = pd.read_parquet(features_path)
    fcols = resolve_feature_columns(features_df)
    event = features_df[
        (features_df[fcols.year] == year) & (features_df[fcols.round_number] == round_number)
    ].reset_index(drop=True)
    if event.empty:
        raise EventNotFoundError(
            f"No rows for year={year}, round_number={round_number} in "
            f"{features_path}. Run scripts/download_2026.py + "
            f"scripts/build_features.py for this event before predicting it."
        )
    return event


@dataclass(frozen=True)
class DriverPrediction:
    driver: str
    era: int
    is_sprint: bool
    n_practice_sessions: int
    predicted_quali_time_seconds: float
    interval_low_seconds: float
    interval_high_seconds: float
    interval_width_seconds: float
    interval_level_pct: float
    interval_exact: bool
    interval_label: str = SHIPPED_INTERVAL_LABEL


# A driver counts as a real weekend entrant if they have 2+ distinct
# practice sessions, or exactly one session that ISN'T FP1. Verified
# against real Round 11 2026 data (Aug 26 2026): the 5 rows FP1-only in
# features.parquet (ARO, FOR, HER, HIR, VES) are all reserve/test drivers
# who ran a single promotional/test FP1 outing and never entered
# qualifying; the other 22 all have 2-3 sessions (including FP2/FP3) and
# are real entrants. This is the inverse rule, stated the way it's easiest
# to check per-driver: EXCLUDE a driver iff their session-code set is
# exactly {"FP1"} - nothing else needs excluding.
TEST_DRIVER_SESSION_SET = frozenset({"FP1"})


def identify_test_drivers(event_features_df: pd.DataFrame, fcols=None) -> List[str]:
    """Return the sorted list of drivers in `event_features_df` who only
    ran FP1 this weekend (see module-level docstring above for the rule
    and how it was verified). These are reserve/test drivers doing a
    promotional or young-driver FP1 outing, not real qualifying entrants -
    a prediction for them is meaningless (there is no qualifying result to
    ever compare it against) and previously inflated Round 11 2026's
    driver count to 27 instead of the real 22.

    `fcols` is accepted for callers that already resolved
    `FeatureColumnMap` and don't want to pay for it twice; resolved fresh
    from `event_features_df` if omitted.
    """
    if fcols is None:
        fcols = resolve_feature_columns(event_features_df)
    sessions_by_driver = event_features_df.groupby(fcols.driver)[fcols.session].apply(
        lambda s: frozenset(s.dropna().unique())
    )
    test_drivers = sessions_by_driver[sessions_by_driver == TEST_DRIVER_SESSION_SET].index.tolist()
    return sorted(str(d) for d in test_drivers)


def predict_event(
    event_features_df: pd.DataFrame, artifacts: ProductionArtifacts
) -> List[DriverPrediction]:
    """Run the production LSTM over one weekend's already-built practice-
    session feature rows (`event_features_df` should already be filtered to
    a single year/round_number - `load_features_for_event` does that; this
    function trusts its caller and predicts every driver found as-is).

    Mirrors `scripts/train_final_lstm.py`'s pipeline shape exactly (pivot ->
    build sequences -> impute -> scale -> forward pass -> add back
    `practice_reference`), with the calibration/split-only columns replaced
    by inert placeholders - see module docstring.
    """
    fcols = resolve_feature_columns(event_features_df)
    feature_cols = artifacts.feature_cols

    # Drop FP1-only reserve/test-driver rows before anything else touches
    # this frame - see identify_test_drivers()'s docstring. Filtering here
    # (rather than upstream in load_features_for_event) keeps the rule in
    # exactly one place regardless of whether a caller built
    # event_features_df from disk or constructed it itself (tests do the
    # latter).
    test_drivers = identify_test_drivers(event_features_df, fcols)
    if test_drivers:
        event_features_df = event_features_df[
            ~event_features_df[fcols.driver].isin(test_drivers)
        ].reset_index(drop=True)

    wide_df = pivot_to_weekend_features(event_features_df, feature_cols, fcols)

    # Same reference computation as
    # f1qp.modeling.dataset.add_practice_reference_and_gaps: the fastest
    # practice lap posted by anyone in the field that weekend - known
    # before Q starts, so legitimate to use here (not leakage).
    reference = (
        event_features_df.groupby([fcols.year, fcols.round_number])["best_lap_time"]
        .min()
        .rename("practice_reference")
        .reset_index()
    )
    wide_df = wide_df.merge(reference, on=[fcols.year, fcols.round_number], how="left")

    # Inert placeholders build_lstm_sequences expects to find on the wide
    # frame - never read by anything downstream of it at inference time
    # (see module docstring point 2).
    wide_df["final_quali_time"] = np.nan
    wide_df["gap_final"] = np.nan
    wide_df["has_target"] = False
    wide_df["split"] = "predict"

    batch = build_lstm_sequences(wide_df, feature_cols, fcols)

    X_imputed = artifacts.imputer.transform(batch.X, batch.mask)
    X_scaled = artifacts.scaler.transform(X_imputed, batch.mask)

    with torch.no_grad():
        pred_gap = artifacts.model(
            torch.as_tensor(X_scaled, dtype=torch.float32),
            torch.as_tensor(batch.lengths, dtype=torch.int64),
            torch.as_tensor(batch.static, dtype=torch.float32),
        ).numpy()
    pred_abs = pred_gap + batch.practice_reference

    low, high = build_interval(pred_abs, artifacts.deployment_quantile_seconds)
    width = high - low

    drivers = wide_df[fcols.driver].tolist()
    is_sprint_col = wide_df[fcols.is_sprint].to_numpy()
    era_col = wide_df[fcols.era].to_numpy()

    results = []
    for i, driver in enumerate(drivers):
        results.append(
            DriverPrediction(
                driver=str(driver),
                era=int(era_col[i]),
                is_sprint=bool(is_sprint_col[i]),
                n_practice_sessions=int(batch.lengths[i]),
                predicted_quali_time_seconds=float(pred_abs[i]),
                interval_low_seconds=float(low[i]),
                interval_high_seconds=float(high[i]),
                interval_width_seconds=float(width[i]),
                interval_level_pct=artifacts.coverage_target_pct,
                interval_exact=artifacts.deployment_quantile_exact,
            )
        )
    return results


def predict_event_from_disk(
    year: int,
    round_number: int,
    artifacts: ProductionArtifacts,
    features_path: Path | None = None,
) -> List[DriverPrediction]:
    """Convenience wrapper the API layer calls directly: load this event's
    rows from `data/processed/features.parquet`, then predict."""
    event_features_df = load_features_for_event(year, round_number, features_path)
    return predict_event(event_features_df, artifacts)


def test_drivers_for_event_from_disk(
    year: int,
    round_number: int,
    features_path: Path | None = None,
) -> List[str]:
    """The excluded-FP1-only-driver list for one event, without running
    the model - the API layer's /predict response and the dashboard's
    "N excluded (FP1-only)" note both want this alongside the predictions
    themselves, not just implicitly folded into a shorter driver list."""
    event_features_df = load_features_for_event(year, round_number, features_path)
    return identify_test_drivers(event_features_df)
