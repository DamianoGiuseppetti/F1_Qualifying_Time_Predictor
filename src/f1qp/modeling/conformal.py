"""Phase 3, LSTM step: confidence intervals via split-conformal calibration.

Task_List.txt's spec for this step: "Add confidence intervals via
split-conformal calibration on held-out validation residuals, era-
stratified, computed on reconstructed absolute time. Interval width is NOT
chosen in advance - it's whatever the real residual spread says. Check
coverage (does a 90% interval actually contain ~90% of true values?) before
trusting it for Round 13." Everything below is built to satisfy that
literally, using residual sources that already exist in this project rather
than inventing a new split.

**The method (standard split-conformal, absolute-residual score):** given a
pool of calibration residuals from data the model never trained on, and a
target miscoverage rate `alpha` (e.g. 0.10 for 90% coverage), the conformal
quantile is the ceil((n+1)(1-alpha))-th smallest ABSOLUTE residual in that
pool. A new point's interval is `[prediction - q, prediction + q]` -
symmetric, and valid regardless of whether the model's residuals are biased,
because the guarantee comes from directly bounding |actual - predicted|
against the calibration pool's own empirical spread, not from assuming
zero-mean/Gaussian errors. This is why interval width is "whatever the real
residual spread says", not a number picked in advance: `q` falls straight
out of the calibration residuals, nothing else.

**Where the calibration residuals come from, and why they differ by era:**

- **Era 1 (2026)**: `leave_one_round_out_cv_lstm`'s per-round out-of-fold
  predictions (each round's residuals come from a model that never trained
  on that round) - mirrors the actual Round 13 situation (a genuinely new,
  never-seen round) far more closely than a random val split could, and is
  already built and confirmed (1.053%/R²=0.989 pooled MAPE/R² - see
  Task_List.txt). This is the era that actually matters operationally: the
  production model only ever predicts 2026 rounds.
- **Era 0 (2023-2025)**: the standard held-out val-split residuals from the
  standalone 80/20 LSTM run (scripts/train_lstm.py) - there is no
  leave-one-round-out structure for era 0 (it was never built, and never
  needed to be: era 0 is historical training data, not a deployment
  target). Kept purely to satisfy Task_List's "era-stratified" requirement
  and for the model card - not used for any real Round 13 decision.

**Checking coverage without circularity:** fitting a quantile on a
calibration pool and then checking coverage on THAT SAME pool is close to
tautological (the pool was used to pick the threshold, so of course close
to alpha-fraction falls outside it by construction). To get an honest
"does a 90% interval actually contain ~90% of true values" answer, both
check functions below use a further held-out split of the calibration pool
itself:

- `leave_one_round_out_conformal_check`: for each round, calibrate on every
  OTHER round's residuals, check coverage on the held-out round's own
  residuals. Repeats for every round, reports per-round and pooled
  coverage. Reuses the exact round-level grouping already used by
  `leave_one_round_out_cv_lstm` - the natural, deployment-mirroring choice
  for era 1.
- `k_fold_conformal_check`: same idea for era 0, where there's no round-OOF
  structure to reuse - a fixed-seed K-fold split of the flat residual pool
  instead.

In both cases, the quantile ACTUALLY used for deployment (`final_quantile`
on each result) is refit on the FULL calibration pool (not a held-out
subset) - more calibration data makes a tighter, more stable quantile
estimate, and by the time deployment happens the coverage check has already
answered the "can this be trusted" question on subsets, so the full-pool
quantile isn't the one whose coverage was just checked.

Known contributor to a wider-than-otherwise-expected quantile: the verified
real disruption weekends already found during the XGBoost/LSTM work (2024
São Paulo, 2025 Las Vegas, 2023 Saudi Arabia, 2026 Spa) sit in the residual
tail. The Huber-loss training dampens their effect on gradient updates, but
does nothing to shrink their contribution to the RESIDUAL itself - a wide
absolute-residual quantile driven partly by these events is an honest
reflection of real variance, not a sign the calibration is broken.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ConformalQuantile:
    """One split-conformal quantile, plus enough context to know whether
    its finite-sample guarantee actually held.

    `exact=False` means the calibration pool was too small for `alpha` -
    the ceil((n+1)(1-alpha)) order statistic that theory calls for would
    have been *past* the largest available residual, so the maximum
    observed residual was used instead. That's the widest defensible
    interval given what's actually been seen, but it is a fallback, not a
    formally guaranteed (1-alpha) bound - callers should treat `exact=False`
    as "trust this less", not as an error.
    """

    quantile: float
    n_calibration: int
    alpha: float
    exact: bool


def conformal_quantile(abs_residuals, alpha: float = 0.10) -> ConformalQuantile:
    """Standard split-conformal quantile (Vovk et al.): the
    ceil((n+1)(1-alpha))-th smallest value among the calibration pool's
    ABSOLUTE residuals.

    `abs_residuals` may be signed or already-absolute - `np.abs` is applied
    unconditionally, since the score this method needs is |actual -
    predicted|, never the signed residual (a systematic bias in the signed
    residual doesn't need correcting for the interval to be valid - see
    module docstring).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")

    abs_residuals = np.abs(np.asarray(abs_residuals, dtype=float))
    n = len(abs_residuals)
    if n < 1:
        raise ValueError("Need at least 1 calibration residual to compute a conformal quantile")

    sorted_abs = np.sort(abs_residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    exact = k <= n
    k_clamped = min(k, n)
    quantile = float(sorted_abs[k_clamped - 1])  # k is 1-indexed
    return ConformalQuantile(quantile=quantile, n_calibration=n, alpha=alpha, exact=exact)


@dataclass
class LOROConformalResult:
    alpha: float
    coverage_target: float
    per_round: Dict
    pooled_coverage: float
    n_test_total: int
    final_quantile: ConformalQuantile


def leave_one_round_out_conformal_check(
    residuals_by_round: Dict, alpha: float = 0.10
) -> LOROConformalResult:
    """Honest, non-circular coverage check for era 1: for each round,
    calibrate a conformal quantile on every OTHER round's residuals (out-
    of-fold relative to both the original model training AND this
    calibration step), then check whether that round's own residuals fall
    inside it. `final_quantile` - the one to actually deploy - is refit on
    every round's residuals pooled together, since by then the coverage
    check has already answered whether this general recipe can be trusted.

    `residuals_by_round` should be `leave_one_round_out_cv_lstm`'s
    `residuals_by_round` return value: {round_number: array of SIGNED
    residuals (pred_abs - actual_abs), one round already held fully out of
    that round's own training}.
    """
    rounds = sorted(residuals_by_round)
    if len(rounds) < 2:
        raise ValueError(
            "Need residuals from at least 2 rounds to run a leave-one-round-out "
            "coverage check - the round held out needs a separate pool of other "
            "rounds' residuals to calibrate against."
        )
    abs_by_round = {r: np.abs(np.asarray(residuals_by_round[r], dtype=float)) for r in rounds}

    per_round = {}
    total_covered = 0
    total_n = 0
    for held_out in rounds:
        calibration_pool = np.concatenate([abs_by_round[r] for r in rounds if r != held_out])
        cq = conformal_quantile(calibration_pool, alpha=alpha)
        test_abs = abs_by_round[held_out]
        n_test = int(len(test_abs))
        covered = test_abs <= cq.quantile
        per_round[held_out] = {
            "quantile": cq.quantile,
            "n_calibration": cq.n_calibration,
            "n_test": n_test,
            "coverage": float(covered.mean()) if n_test else float("nan"),
            "exact": cq.exact,
        }
        total_covered += int(covered.sum())
        total_n += n_test

    pooled_coverage = (total_covered / total_n) if total_n else float("nan")
    final_quantile = conformal_quantile(
        np.concatenate([abs_by_round[r] for r in rounds]), alpha=alpha
    )

    return LOROConformalResult(
        alpha=alpha,
        coverage_target=1 - alpha,
        per_round=per_round,
        pooled_coverage=pooled_coverage,
        n_test_total=total_n,
        final_quantile=final_quantile,
    )


@dataclass
class KFoldConformalResult:
    alpha: float
    coverage_target: float
    k: int
    per_fold: List[Dict]
    pooled_coverage: float
    n_test_total: int
    final_quantile: ConformalQuantile


def k_fold_conformal_check(
    abs_residuals, alpha: float = 0.10, k: int = 5, seed: int = 42
) -> KFoldConformalResult:
    """Era-0 analogue of `leave_one_round_out_conformal_check`: there's no
    round-level out-of-fold structure for era 0 (never built, never needed
    for actual deployment - see module docstring), so this uses a fixed-
    seed K-fold split of the flat calibration pool instead. Same recipe
    otherwise: calibrate each fold's quantile on the other k-1 folds, check
    coverage on the held-out fold, report per-fold and pooled coverage, and
    a `final_quantile` refit on the whole pool for reference/documentation.

    Deliberately no scikit-learn dependency (consistent with
    f1qp.modeling.dataset's own by-hand split) - a fixed-seed shuffle plus
    `np.array_split` is all a flat K-fold needs.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k!r}")

    abs_residuals = np.abs(np.asarray(abs_residuals, dtype=float))
    n = len(abs_residuals)
    if n < 2 * k:
        raise ValueError(
            f"Only {n} calibration residuals available - need at least {2 * k} "
            f"for a meaningful {k}-fold coverage check (>= 2 residuals per fold)."
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    per_fold = []
    total_covered = 0
    total_n = 0
    for i, test_idx in enumerate(folds):
        calib_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        cq = conformal_quantile(abs_residuals[calib_idx], alpha=alpha)
        test_abs = abs_residuals[test_idx]
        n_test = int(len(test_abs))
        covered = test_abs <= cq.quantile
        per_fold.append({
            "fold": i,
            "quantile": cq.quantile,
            "n_calibration": cq.n_calibration,
            "n_test": n_test,
            "coverage": float(covered.mean()) if n_test else float("nan"),
            "exact": cq.exact,
        })
        total_covered += int(covered.sum())
        total_n += n_test

    pooled_coverage = (total_covered / total_n) if total_n else float("nan")
    final_quantile = conformal_quantile(abs_residuals, alpha=alpha)

    return KFoldConformalResult(
        alpha=alpha,
        coverage_target=1 - alpha,
        k=k,
        per_fold=per_fold,
        pooled_coverage=pooled_coverage,
        n_test_total=total_n,
        final_quantile=final_quantile,
    )


def build_interval(point_pred, quantile: float):
    """Symmetric split-conformal interval around a point prediction:
    `[point_pred - quantile, point_pred + quantile]`. `point_pred` and the
    result are on the same scale the calibration residuals were computed
    on - reconstructed absolute qualifying time (seconds), per this
    module's docstring, never the raw gap.

    Not wired into any script yet - Phase 4's inference endpoint is where a
    real point prediction will exist to apply this to. Kept here as a
    one-line building block so that step doesn't have to re-derive the
    (trivial) interval arithmetic.
    """
    point_pred = np.asarray(point_pred, dtype=float)
    return point_pred - quantile, point_pred + quantile
