"""Phase 3, step 4: XGBoost baseline before the LSTM.

Serves two jobs at once (both named in the project brief / Task_List.txt):
a fallback model if the LSTM doesn't converge, and an empirical feature
selector - its gain and permutation importances finalize what goes into the
LSTM, since the Step 3 correlation check alone found no pair extreme enough
to force a drop.

Trains ONE model (simplified Aug 23 2026, replacing an earlier three-way
Q1/Q2/Q3 masked design): the target is `final_quali_time` - the time that
actually determined each driver's grid position (Q3 if they reached it,
otherwise Q2, otherwise Q1). Every driver who took part in qualifying has
exactly one such value, so no per-segment masking is needed.

Builds two target formulations:
  - "absolute": predict final_quali_time directly.
  - "gap": predict (final_quali_time - practice_reference), reconstruct
    absolute time for scoring (see f1qp.modeling.dataset module docstring
    for why the gap itself is never scored directly - it's often
    negative/near-zero).

Uses a Huber-like loss (`reg:pseudohubererror`), not plain squared error:
verified real-data investigation (Aug 23 2026) found the gap target's long
tail is dominated by two whole-weekend rain-affected qualifying sessions
(2024 Sao Paulo, 2025 Las Vegas) plus one genuine one-off mechanical
failure (2023 Saudi Arabia, Sargeant) - real events, not data errors, kept
in training, but without a robust loss they would dominate gradient
updates for the ~97% of ordinary dry weekends.

Only the "train" and "val" splits are ever touched here. Rows with
split == "holdout" (Zandvoort) are excluded structurally (see
train_model) - that weekend is Phase 5's offline test, never used for
model selection.

`leave_one_round_out_cv` is a separate diagnostic (Aug 23 2026): the main
80/20 split puts only ~2 of the 11 2026 weekends into era 1's validation
slice - too thin to trust "does this work on 2026" from alone. It holds
out each 2026 round in turn, trains on everything else, and pools the
results - 11 independent tests instead of one small random split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

RANDOM_SEED = 42
PERMUTATION_REPEATS = 5

TARGET_COL = "final_quali_time"
HAS_TARGET_COL = "has_target"
GAP_COL = "gap_final"

DEFAULT_XGB_PARAMS = dict(
    objective="reg:pseudohubererror",
    huber_slope=1.0,  # errors beyond ~1s are treated as roughly linear (robust)
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
)


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error. Only ever called on absolute lap
    times (always well clear of zero) - never on the gap target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def r_squared(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def permutation_importance(
    model: xgb.XGBRegressor,
    X_val: pd.DataFrame,
    y_val_true: pd.Series,
    practice_reference: pd.Series,
    formulation: str,
    seed: int = RANDOM_SEED,
    n_repeats: int = PERMUTATION_REPEATS,
) -> pd.Series:
    """Permutation importance scored by MAPE increase on reconstructed
    absolute time, implemented directly (no scikit-learn dependency)."""
    rng = np.random.default_rng(seed)

    def to_abs(raw_pred):
        if formulation == "gap":
            return raw_pred + practice_reference.to_numpy()
        return raw_pred

    baseline_score = mape(y_val_true, to_abs(model.predict(X_val)))

    importances = {}
    for col in X_val.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = X_val.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(mape(y_val_true, to_abs(model.predict(shuffled))) - baseline_score)
        importances[col] = float(np.mean(drops))
    return pd.Series(importances).sort_values(ascending=False)


@dataclass
class ModelResult:
    formulation: str
    model: xgb.XGBRegressor
    feature_cols: list
    val_mape: float
    val_r2: float
    val_mape_by_era: Dict = field(default_factory=dict)
    val_r2_by_era: Dict = field(default_factory=dict)
    gain_importance: pd.Series = field(default_factory=pd.Series)
    permutation_importance: pd.Series = field(default_factory=pd.Series)
    n_train: int = 0
    n_val: int = 0


def train_model(
    wide_df: pd.DataFrame,
    formulation: str,
    feature_cols: Sequence[str],
    era_col: str,
    target_col: str = TARGET_COL,
    has_target_col: str = HAS_TARGET_COL,
    gap_col: str = GAP_COL,
    xgb_params: dict = None,
) -> ModelResult:
    """Train one XGBoost regressor on `final_quali_time` (or its gap).

    `feature_cols` are the base feature names (e.g. "best_lap_time") - the
    actual training columns are the pivoted `session{0,1,2}_<feature>`
    columns produced by `dataset.pivot_to_weekend_features`.
    """
    usable = wide_df[wide_df[has_target_col].astype(bool)].copy()
    # Holdout (Zandvoort) is excluded structurally - Phase 5's job, not this one.
    train_df = usable[usable["split"] == "train"]
    val_df = usable[usable["split"] == "val"]

    session_feature_cols = [
        c for c in wide_df.columns
        if any(c == f"session{i}_{feat}" for i in (0, 1, 2) for feat in feature_cols)
    ]

    if formulation == "absolute":
        y_train = train_df[target_col]
        y_val_true = val_df[target_col]
    elif formulation == "gap":
        y_train = train_df[gap_col]
        y_val_true = val_df[target_col]  # always compared in absolute space
    else:
        raise ValueError(f"Unknown formulation: {formulation!r} (expected 'absolute' or 'gap')")

    params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    model = xgb.XGBRegressor(**params)
    model.fit(train_df[session_feature_cols], y_train)

    X_val = val_df[session_feature_cols]
    raw_pred = pd.Series(model.predict(X_val), index=val_df.index)
    if formulation == "gap":
        pred_abs = raw_pred + val_df["practice_reference"]
    else:
        pred_abs = raw_pred

    val_mape_score = mape(y_val_true, pred_abs)
    val_r2_score = r_squared(y_val_true, pred_abs)

    val_mape_by_era, val_r2_by_era = {}, {}
    for era_value, group in val_df.groupby(era_col):
        idx = group.index
        val_mape_by_era[era_value] = mape(y_val_true.loc[idx], pred_abs.loc[idx])
        val_r2_by_era[era_value] = r_squared(y_val_true.loc[idx], pred_abs.loc[idx])

    gain_importance = pd.Series(
        model.feature_importances_, index=session_feature_cols
    ).sort_values(ascending=False)

    perm_importance = permutation_importance(
        model, X_val, y_val_true, val_df["practice_reference"], formulation
    )

    return ModelResult(
        formulation=formulation,
        model=model,
        feature_cols=session_feature_cols,
        val_mape=val_mape_score,
        val_r2=val_r2_score,
        val_mape_by_era=val_mape_by_era,
        val_r2_by_era=val_r2_by_era,
        gain_importance=gain_importance,
        permutation_importance=perm_importance,
        n_train=len(train_df),
        n_val=len(val_df),
    )


def leave_one_round_out_cv(
    wide_df: pd.DataFrame,
    feature_cols: Sequence[str],
    fcols,
    formulation: str = "gap",
    era_value=1,
    target_col: str = TARGET_COL,
    has_target_col: str = HAS_TARGET_COL,
    gap_col: str = GAP_COL,
    xgb_params: dict = None,
) -> dict:
    """Leave-one-round-out CV within one era (2026 by default).

    Motivation (Aug 23 2026): the main 80/20 validation split puts only
    ~2 of the 11 2026 training weekends into era 1's validation slice -
    too thin to trust a claim that the model "works on 2026" (a single
    unlucky or lucky weekend, e.g. the verified Las Vegas rain weekend,
    could swing that number either way). Here every 2026 round is held out
    exactly once and tested after training on everything else - all of
    era 0 plus the other 2026 rounds - which mirrors the actual Round 13
    deployment setup (train on all prior data, predict one unseen round)
    far more closely than a random split does, and gives 11 independent
    test folds instead of one.

    Ignores the 'split' column's train/val distinction entirely (folds are
    defined by round, not by the pre-existing split) but still excludes
    split == 'holdout' rows (Zandvoort) from ever entering training or
    testing here - that stays reserved for the Phase 5 offline test.
    """
    round_col, era_col = fcols.round_number, fcols.era

    pool = wide_df[
        (wide_df["split"] != "holdout") & (wide_df[has_target_col].astype(bool))
    ].copy()
    era_rounds = sorted(pool.loc[pool[era_col] == era_value, round_col].unique())
    if not era_rounds:
        raise ValueError(f"No rounds found for era {era_value!r} in this dataset")

    session_feature_cols = [
        c for c in wide_df.columns
        if any(c == f"session{i}_{feat}" for i in (0, 1, 2) for feat in feature_cols)
    ]

    params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    per_round = {}
    all_true, all_pred = [], []
    for round_number in era_rounds:
        is_test_round = (pool[era_col] == era_value) & (pool[round_col] == round_number)
        train_fold = pool.loc[~is_test_round]
        test_fold = pool.loc[is_test_round]

        if formulation == "absolute":
            y_train = train_fold[target_col]
        elif formulation == "gap":
            y_train = train_fold[gap_col]
        else:
            raise ValueError(f"Unknown formulation: {formulation!r} (expected 'absolute' or 'gap')")
        y_test_true = test_fold[target_col]

        model = xgb.XGBRegressor(**params)
        model.fit(train_fold[session_feature_cols], y_train)
        raw_pred = model.predict(test_fold[session_feature_cols])
        if formulation == "gap":
            pred_abs = raw_pred + test_fold["practice_reference"].to_numpy()
        else:
            pred_abs = raw_pred

        per_round[round_number] = {
            "mape": mape(y_test_true, pred_abs),
            "r2": r_squared(y_test_true, pred_abs),
            "n_test": len(test_fold),
        }
        all_true.append(y_test_true.to_numpy())
        all_pred.append(np.asarray(pred_abs))

    all_true_arr = np.concatenate(all_true)
    all_pred_arr = np.concatenate(all_pred)
    pooled = {
        "mape": mape(all_true_arr, all_pred_arr),
        "r2": r_squared(all_true_arr, all_pred_arr),
        "n_test": len(all_true_arr),
    }

    return {
        "formulation": formulation,
        "era_value": era_value,
        "per_round": per_round,
        "pooled": pooled,
    }


def run_baseline_comparison(wide_df: pd.DataFrame, feature_cols: Sequence[str], fcols) -> dict:
    """Train both formulations, pick ONE winning formulation by validation
    MAPE on reconstructed absolute time."""
    results = {
        formulation: train_model(wide_df, formulation, feature_cols, fcols.era)
        for formulation in ("absolute", "gap")
    }
    winning_formulation = min(results, key=lambda f: results[f].val_mape)

    return {
        "results": results,
        "aggregate_val_mape": {f: results[f].val_mape for f in results},
        "winning_formulation": winning_formulation,
    }
