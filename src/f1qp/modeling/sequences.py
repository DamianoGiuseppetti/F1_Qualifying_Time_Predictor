"""Phase 3, LSTM step 0: reshape the pivoted wide table into real sequences.

`dataset.pivot_to_weekend_features` flattens each driver-weekend's practice
sessions into `session{0,1,2}_<feature>` COLUMNS - the right shape for
XGBoost, which has no concept of "step 2 comes after step 1". The LSTM
needs the opposite: an actual (session, feature) AXIS it can walk over in
order, plus an explicit indicator of which steps are real vs padded, so a
sprint weekend's missing 3rd session is something the model is TOLD is
absent rather than something it has to infer from a run of zeros (which
would be indistinguishable from "a very slow, very consistent FP3" -
exactly the silent-zero-padding failure mode Phase 1 ruled out).

This module reuses `pivot_to_weekend_features`'s already-tested wide
output rather than re-deriving the per-slot merges.

**Real-data correction (Aug 23 2026):** the first version of this module
assumed a slot's features are all-present or all-absent together - true on
the synthetic fixtures (which never generate a lone missing feature), false
on the real features.parquet: a session that genuinely happened can still
have one or two individual features come back NaN (e.g. `long_run_avg_pace`
has nothing to compute if a driver never ran a long stint that session,
independent of whether the session itself happened). Confirmed on the real
first run (raised loudly rather than silently mis-masking, per this
module's original design intent - it just drew the line in the wrong
place). Fixed by separating two different kinds of "missing" that used to
be conflated:

1. **Slot absence** (a sprint weekend's FP3): every feature in the slot is
   NaN at once, because the merge in `pivot_to_weekend_features` has no
   source row at all for that (year, round, driver, slot). Detected as
   "ALL features NaN", not "the first feature is NaN" - `mask` now reflects
   this correctly.
2. **Feature absence within a real session** (e.g. no long run happened):
   some features NaN, others present, in a slot the driver did participate
   in. This is real, expected missing data, not a padding signal - it's
   handled by `FeatureImputer` (median fill, fit on train only), applied
   AFTER masking and BEFORE scaling, never by pretending the whole session
   didn't happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

MAX_SESSIONS = 3


@dataclass
class SequenceBatch:
    """Everything one LSTM run needs, aligned row-for-row.

    `X` and `mask` are the sequence input (shape (n, MAX_SESSIONS,
    n_features) and (n, MAX_SESSIONS) respectively). `static` holds the
    per-weekend context features that sit alongside the sequence rather
    than inside it (is_sprint, era - see docs/feature_engineering.md's
    "Global (non-per-timestep) context features"). Everything else is
    needed for supervision or reconstruction, never as a model input.

    `X` may still contain NaN at this stage, at positions where a real
    (mask=1) slot has a genuinely missing individual feature value - see
    module docstring. Padded (mask=0) positions are always 0.0, never NaN.
    Run `FeatureImputer` before `FeatureScaler` to get a fully numeric
    array.
    """

    X: np.ndarray
    mask: np.ndarray
    static: np.ndarray
    static_cols: list
    lengths: np.ndarray
    y_gap: np.ndarray
    y_abs: np.ndarray
    practice_reference: np.ndarray
    has_target: np.ndarray
    split: np.ndarray
    era: np.ndarray
    ids: pd.DataFrame


def build_lstm_sequences(
    wide_df: pd.DataFrame,
    feature_cols: Sequence[str],
    fcols,
    max_sessions: int = MAX_SESSIONS,
) -> SequenceBatch:
    """Reshape `pivot_to_weekend_features`'s wide output into sequences.

    A slot is "real" (mask=1) when AT LEAST ONE of its features is present
    - a slot is only ever fully absent (every feature NaN at once) when the
    driver didn't take part in that session at all, since the wide table's
    per-slot merge has no source row to draw any feature from in that case.
    A real slot's individual feature can still be NaN on its own (see
    module docstring) - that's left as NaN in `X` for `FeatureImputer` to
    handle, not filled with 0.0 here (0.0 isn't a meaningful value for,
    say, a sector time - filling it in blind would quietly bias the scaler
    fit downstream).

    Padded (mask=0) slots ARE zero-filled here - `lengths` (real session
    count) is what actually drives `pack_padded_sequence`, which stops the
    LSTM from processing a padded step at all, so the placeholder value at
    a padded position is never seen by the model regardless of what it is.
    """
    n = len(wide_df)
    n_features = len(feature_cols)
    X = np.full((n, max_sessions, n_features), np.nan, dtype=np.float32)
    mask = np.zeros((n, max_sessions), dtype=np.float32)

    for slot in range(max_sessions):
        cols = [f"session{slot}_{feat}" for feat in feature_cols]
        missing_cols = [c for c in cols if c not in wide_df.columns]
        if missing_cols:
            raise KeyError(
                f"wide_df is missing expected pivoted columns for slot {slot}: "
                f"{missing_cols}. Did you pass the output of "
                f"dataset.pivot_to_weekend_features?"
            )
        # .to_numpy() can return a read-only array (same pitfall hit once
        # already in dataset.py's build_weekend_split) - .copy() before any
        # in-place assignment.
        block = wide_df[cols].to_numpy(dtype=np.float32).copy()
        slot_real = ~np.isnan(block).all(axis=1)
        mask[:, slot] = slot_real.astype(np.float32)
        # Real slots keep their values as-is (including any lone NaN
        # feature, left for FeatureImputer). Padded slots are zeroed - the
        # model never sees them via pack_padded_sequence, but a stray NaN
        # sitting in X would still break loss.backward() if anything ever
        # touched it (e.g. a bug elsewhere), so zero rather than NaN there.
        block[~slot_real] = 0.0
        X[:, slot, :] = block

    lengths = mask.sum(axis=1).astype(np.int64)
    if (lengths == 0).any():
        raise ValueError(
            "Found row(s) with zero real practice sessions - every "
            "driver-weekend should have at least FP1. Check the pivot step."
        )

    static_cols = [fcols.is_sprint, fcols.era]
    static = wide_df[static_cols].to_numpy(dtype=np.float32)

    y_abs = wide_df["final_quali_time"].to_numpy(dtype=np.float64)
    y_gap = wide_df["gap_final"].to_numpy(dtype=np.float64)
    practice_reference = wide_df["practice_reference"].to_numpy(dtype=np.float64)
    has_target = wide_df["has_target"].to_numpy(dtype=bool)
    split = wide_df["split"].to_numpy()
    era = wide_df[fcols.era].to_numpy()
    ids = wide_df[[fcols.year, fcols.round_number, fcols.driver]].reset_index(drop=True)

    return SequenceBatch(
        X=X,
        mask=mask,
        static=static,
        static_cols=static_cols,
        lengths=lengths,
        y_gap=y_gap,
        y_abs=y_abs,
        practice_reference=practice_reference,
        has_target=has_target,
        split=split,
        era=era,
        ids=ids,
    )


@dataclass
class FeatureImputer:
    """Per-feature median fill for real (mask=1) timesteps that still have
    a NaN value in one individual feature (see module docstring - this is
    expected, e.g. `long_run_avg_pace` when a driver ran no long stint that
    session, not a sign the session itself is missing).

    Fit on TRAIN's real timesteps only, same reasoning as `FeatureScaler`:
    a median pulled from val (or from padded placeholder values) would leak
    information the model shouldn't get to see at "training time". Median,
    not mean, because a feature that's missing specifically BECAUSE a
    driver did something unusual that session (skipped a long run, skipped
    representative laps) is exactly the kind of feature likely to have a
    skewed distribution - the median is the more robust fallback of the
    two, consistent with this project's general preference for robust
    statistics over the plain versions (see also: the Huber loss choice in
    f1qp.modeling.baseline / lstm_model).

    `n_imputed_per_feature` is kept on the instance after `fit` purely for
    diagnostics - scripts/train_lstm.py prints it so a feature that's
    missing far more often than the others is visible, not silently
    absorbed.
    """

    median: np.ndarray
    n_imputed_per_feature: np.ndarray = None

    @classmethod
    def fit(cls, X: np.ndarray, mask: np.ndarray) -> "FeatureImputer":
        real = mask.astype(bool)
        # Diagnostic count FIRST, before padded positions get folded into
        # NaN below for the median calc - a padded slot's placeholder isn't
        # a "missing feature value", so it must never be counted as one.
        real_missing = real[:, :, None] & np.isnan(X)
        n_missing = real_missing.reshape(-1, X.shape[-1]).sum(axis=0)

        # nanmedian over real timesteps only; a feature that's NaN in
        # EVERY real train timestep (shouldn't happen, but don't crash on
        # it) falls back to 0.0 rather than producing a NaN median.
        flat = np.where(real[:, :, None], X, np.nan)
        with np.errstate(all="ignore"):
            median = np.nanmedian(flat.reshape(-1, X.shape[-1]), axis=0)
        median = np.nan_to_num(median, nan=0.0)
        return cls(median=median.astype(np.float32), n_imputed_per_feature=n_missing)

    def transform(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(X), self.median, X)
        # Anything outside a real slot stays exactly 0.0 regardless (should
        # already be 0.0 from build_lstm_sequences - this is just a guard).
        return np.where(mask.astype(bool)[:, :, None], filled, 0.0)


@dataclass
class FeatureScaler:
    """Per-feature standardization (mean 0, std 1), fit on TRAIN's real
    (non-padded) timesteps only and applied identically everywhere else.

    Trees (XGBoost) split on thresholds and don't care about feature scale;
    an LSTM's gates and gradients do - a raw `n_runs` value of 2-6 and a raw
    `best_lap_time` of ~80-100 would otherwise let the pace features
    dominate purely because of their units, not their actual importance.
    Padded slots are excluded from fitting for the same reason they're
    excluded from the loss - their values are placeholders, not signal, and
    letting a sea of the pad fill value drag the mean/std around would bias
    the scaling for the real timesteps that actually matter.

    Expects `X` to already be fully numeric (no NaN) - run `FeatureImputer`
    first. Fitting a mean/std over data that still has NaN in it would
    propagate NaN into every downstream computation.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, mask: np.ndarray) -> "FeatureScaler":
        if np.isnan(X).any():
            raise ValueError(
                "FeatureScaler.fit received NaN values - run FeatureImputer "
                "first (fit+transform on the same split) to fill real-slot "
                "missing features before scaling."
            )
        real = mask.astype(bool)
        flat = X[real]  # (n_real_timesteps, n_features)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)  # guard constant features
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        scaled = (X - self.mean) / self.std
        # padded slots are still 0 after scaling isn't guaranteed (0 - mean)
        # / std != 0 in general - re-zero them explicitly so pack_padded's
        # "never processed" guarantee is matched by "never even a stray
        # nonzero value" for anyone who inspects X directly.
        return scaled * mask[:, :, None]
