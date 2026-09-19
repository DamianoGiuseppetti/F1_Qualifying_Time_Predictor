"""Phase 3, LSTM step: learning curve diagnosis + SHAP error analysis
support code.

Task_List.txt: "Learning curves + SHAP error analysis; sanity-check
against the small-sample risk flagged during planning (~1,500 independent
driver-weekend sequences feeding a 64-hidden-unit LSTM)." See
scripts/analyze_lstm.py for the script that actually runs both checks on
real data - this module holds only the pure, fully-unit-testable logic
(no `shap` or `torch` import here at all, deliberately - see below).

**Why SHAP is scoped to non-sprint (3-real-session) rows only:** a sprint
weekend's `lengths=2` is a STRUCTURAL property of that weekend - which
session steps `pack_padded_sequence` ever lets the LSTM see - not a
feature VALUE to perturb. SHAP needs ONE predict function shared across
every instance it explains; mixing sprint and non-sprint rows would need
either a different closure per distinct (lengths, is_sprint) combination,
or pretending a sprint weekend could have had a real 3rd session, neither
of which is sound. Restricting to non-sprint rows keeps `lengths=3` and
`is_sprint=0` fixed and IDENTICAL for every explained instance, so a
single predict function (built in scripts/analyze_lstm.py, not here) is
genuinely correct for all of them. `era` is NOT held fixed like
`is_sprint` is - it doesn't affect the length-based masking logic at all,
only the head's direct input, so it's treated as one more feature SHAP is
free to explain.

**Why the flat layout is session-major with era last**
(`flatten_features_for_shap`/`unflatten_features_from_shap`): SHAP's
KernelExplainer works on a flat 2D (instances x features) array, not the
LSTM's native (instances, sessions, features) tensor - this is the
translation layer between the two, kept as pure array reshaping with no
model-specific logic so it's trivially testable and has nothing to do with
`torch` at all.

**Why explaining the GAP-scale prediction is equivalent to explaining
reconstructed absolute time**: `pred_abs = pred_gap + practice_reference`,
and `practice_reference` is a per-INSTANCE additive constant, not one of
the perturbed input features. Adding a constant to a function's output
shifts SHAP's reported base/expected value for that instance, but leaves
every individual feature's Shapley attribution completely unchanged -
Shapley values are computed relative to how much perturbing each feature
moves the output away from a baseline, and a constant offset moves every
possible coalition's output by the same amount, so it cancels out of every
attribution. This means scripts/analyze_lstm.py can skip reconstruction
for the SHAP explanation itself - reconstruction is still needed
separately to pick the low/high error buckets, since "error" has to be
measured in real seconds, not the unitless-feeling gap.

Deliberately no `shap` (or `torch`) import in this module: only the
pure numpy/pandas logic lives here, so the test suite can fully exercise
it without either dependency needing to be installed/importable, and a
`shap` API surprise (version differences, etc.) is isolated entirely to
scripts/analyze_lstm.py, where Damiano will see it directly if it happens.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd

from f1qp.modeling.sequences import MAX_SESSIONS


@dataclass
class LearningCurveDiagnosis:
    best_epoch: int
    final_epoch: int
    best_val_mape: float
    train_loss_at_best_epoch: float
    train_loss_at_final_epoch: float
    train_loss_kept_falling_after_best: bool
    val_mape_worst_after_best: float
    val_mape_drift_after_best: float
    epochs_trained_past_best: int
    overfitting_signature: bool

    def to_dict(self) -> dict:
        return asdict(self)


def diagnose_learning_curve(history_df: pd.DataFrame) -> LearningCurveDiagnosis:
    """Quantify the overfitting pattern a small-sample LSTM run is expected
    to show (Task_List.txt's own flagged risk): does train loss keep
    falling PAST the epoch where validation performance was best, while
    validation performance itself gets WORSE (not just plateaus)? That
    combination - not merely "training continued after the best epoch",
    which is normal and exactly why early stopping exists - is the actual
    overfitting signature. Early stopping already protects the model that
    ships (it reloads the best epoch's weights), so a `True` verdict here
    confirms the safeguard is doing real work, not that anything is wrong
    with the shipped artifact.

    `history_df` should have columns "epoch", "train_loss", "val_mape"
    (exactly what scripts/train_lstm.py's saved learning_curve.csv has).
    """
    if history_df.empty:
        raise ValueError("learning curve history is empty - nothing to diagnose")

    best_idx = history_df["val_mape"].idxmin()
    best_row = history_df.loc[best_idx]
    final_row = history_df.iloc[-1]

    best_epoch = int(best_row["epoch"])
    final_epoch = int(final_row["epoch"])
    post_best = history_df[history_df["epoch"] > best_epoch]

    train_loss_kept_falling = bool(
        len(post_best) > 0 and post_best["train_loss"].iloc[-1] < best_row["train_loss"]
    )
    if len(post_best) > 0:
        val_mape_worst_after_best = float(post_best["val_mape"].max())
    else:
        val_mape_worst_after_best = float(best_row["val_mape"])
    val_mape_drift = val_mape_worst_after_best - float(best_row["val_mape"])

    return LearningCurveDiagnosis(
        best_epoch=best_epoch,
        final_epoch=final_epoch,
        best_val_mape=float(best_row["val_mape"]),
        train_loss_at_best_epoch=float(best_row["train_loss"]),
        train_loss_at_final_epoch=float(final_row["train_loss"]),
        train_loss_kept_falling_after_best=train_loss_kept_falling,
        val_mape_worst_after_best=val_mape_worst_after_best,
        val_mape_drift_after_best=val_mape_drift,
        epochs_trained_past_best=final_epoch - best_epoch,
        overfitting_signature=bool(train_loss_kept_falling and val_mape_drift > 0),
    )


def flatten_features_for_shap(X: np.ndarray, era: np.ndarray) -> np.ndarray:
    """(n, MAX_SESSIONS, n_features) + (n,) era -> (n, MAX_SESSIONS*n_features + 1).

    Session-major flattening (all of slot 0's features, then slot 1's,
    then slot 2's), era appended as the final column. `X` must already be
    fully numeric (imputed + scaled) - this is pure reshaping, it does not
    handle NaN.
    """
    n, n_sessions, n_features = X.shape
    if n_sessions != MAX_SESSIONS:
        raise ValueError(f"Expected {MAX_SESSIONS} session slots, got {n_sessions}")
    if era.shape[0] != n:
        raise ValueError(f"X has {n} rows but era has {era.shape[0]}")
    flat = X.reshape(n, n_sessions * n_features)
    return np.concatenate([flat, era.reshape(n, 1).astype(np.float32)], axis=1)


def unflatten_features_from_shap(flat: np.ndarray, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of `flatten_features_for_shap`."""
    n = flat.shape[0]
    expected_cols = MAX_SESSIONS * n_features + 1
    if flat.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {flat.shape[1]}")
    X = flat[:, : MAX_SESSIONS * n_features].reshape(n, MAX_SESSIONS, n_features)
    era = flat[:, MAX_SESSIONS * n_features]
    return X, era


def aggregate_shap_by_feature(shap_values: np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
    """Collapse a flat (n_instances, MAX_SESSIONS*n_features + 1) SHAP
    value array (same layout as `flatten_features_for_shap`'s output) into
    one row per BASE feature name - summing/averaging across all
    MAX_SESSIONS session slots - plus one row for 'era'.

    `mean_abs_shap`: mean(|shap|) across instances - the usual "how much
    does this feature matter, regardless of direction" importance measure.
    `mean_signed_shap`: mean(shap), signed - shows whether a feature pushes
    the prediction the same way across instances (a large-magnitude signed
    mean close to the absolute mean) or in different directions depending
    on context (signed mean much smaller than the absolute mean).

    Sorted by `mean_abs_shap` descending, matching
    f1qp.modeling.baseline's gain_importance/permutation_importance
    convention (most important first).
    """
    n_features = len(feature_cols)
    expected_cols = MAX_SESSIONS * n_features + 1
    if shap_values.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {shap_values.shape[1]}")

    rows = []
    for i, feat in enumerate(feature_cols):
        slot_cols = [i + slot * n_features for slot in range(MAX_SESSIONS)]
        values = shap_values[:, slot_cols]  # (n_instances, MAX_SESSIONS)
        rows.append({
            "feature": feat,
            "mean_abs_shap": float(np.mean(np.abs(values))),
            "mean_signed_shap": float(np.mean(values)),
        })
    era_values = shap_values[:, -1]
    rows.append({
        "feature": "era",
        "mean_abs_shap": float(np.mean(np.abs(era_values))),
        "mean_signed_shap": float(np.mean(era_values)),
    })
    return (
        pd.DataFrame(rows)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def select_error_buckets(abs_residuals: np.ndarray, bucket_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of the `bucket_size` smallest ("low error") and
    `bucket_size` largest ("high error") absolute residuals.

    If there are fewer than `2 * bucket_size` residuals, both buckets
    shrink equally to `n // 2` each rather than raising - a
    smaller-than-requested comparison is still informative; silently
    returning empty/overlapping buckets would not be. Raises only when
    there are fewer than 2 residuals total (nothing to compare).
    """
    n = len(abs_residuals)
    if n < 2:
        raise ValueError(f"Need at least 2 residuals to form low/high error buckets, got {n}")
    bucket_size = min(bucket_size, n // 2)
    order = np.argsort(abs_residuals, kind="stable")
    low_idx = order[:bucket_size]
    high_idx = order[-bucket_size:]
    return low_idx, high_idx
