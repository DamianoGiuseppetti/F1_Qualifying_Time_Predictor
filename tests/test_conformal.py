"""Tests for f1qp.modeling.conformal. All synthetic, hand-computable data -
real predictive behavior needs the actual model/residuals, which is what
scripts/calibrate_conformal.py exercises on real data. These check the
arithmetic (the ceil((n+1)(1-alpha)) order-statistic formula, the
leave-one-round-out / K-fold coverage-check bookkeeping) is correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from f1qp.modeling.conformal import (
    build_interval,
    conformal_quantile,
    k_fold_conformal_check,
    leave_one_round_out_conformal_check,
)


def test_conformal_quantile_matches_hand_computed_order_statistic():
    # n=19, alpha=0.1 -> k = ceil(20*0.9) = ceil(18.0) = 18 (exact, k<=19).
    # sorted ascending 1..19 (already sorted), 18th smallest (1-indexed) is
    # index 17 (0-indexed) = 18.0.
    abs_residuals = np.arange(1, 20, dtype=float)  # 1..19
    result = conformal_quantile(abs_residuals, alpha=0.10)
    assert result.quantile == pytest.approx(18.0)
    assert result.n_calibration == 19
    assert result.alpha == pytest.approx(0.10)
    assert result.exact is True


def test_conformal_quantile_falls_back_to_max_when_pool_too_small():
    # n=3, alpha=0.1 -> k = ceil(4*0.9) = ceil(3.6) = 4 > 3 -> not exact,
    # falls back to the max observed residual.
    abs_residuals = [1.0, 2.0, 5.0]
    result = conformal_quantile(abs_residuals, alpha=0.10)
    assert result.quantile == pytest.approx(5.0)
    assert result.exact is False
    assert result.n_calibration == 3


def test_conformal_quantile_uses_absolute_value_of_signed_residuals():
    # A negative residual should contribute its magnitude, not be treated
    # as smaller than a positive one of lesser magnitude.
    signed_residuals = [-10.0, 1.0, 2.0]
    result = conformal_quantile(signed_residuals, alpha=0.5)
    # n=3, alpha=0.5 -> k=ceil(4*0.5)=2 -> sorted abs = [1,2,10], index1 = 2.0
    assert result.quantile == pytest.approx(2.0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_conformal_quantile_raises_on_alpha_out_of_range(alpha):
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile([1.0, 2.0, 3.0], alpha=alpha)


def test_conformal_quantile_raises_on_empty_input():
    with pytest.raises(ValueError, match="at least 1"):
        conformal_quantile([], alpha=0.10)


def test_conformal_quantile_is_monotonic_in_coverage_level():
    """Regression guard for the Aug 24 2026 coverage/width tradeoff sweep
    (scripts/calibrate_conformal.py now reports 50/68/80/90% side by side):
    on a FIXED calibration pool, asking for a higher target coverage
    (smaller alpha) must never produce a narrower quantile than a lower
    one - if it did, the tradeoff table's own premise (narrower coverage =
    narrower width) would be broken."""
    rng = np.random.default_rng(1)
    abs_residuals = rng.exponential(scale=1.0, size=150)  # right-skewed, like real residuals

    quantiles = [
        conformal_quantile(abs_residuals, alpha=alpha).quantile
        for alpha in (0.50, 0.32, 0.20, 0.10)  # decreasing alpha = increasing coverage
    ]
    assert quantiles == sorted(quantiles)


def test_leave_one_round_out_conformal_check_hand_computed_example():
    # 4 rounds, each with 10 identical residuals: A=1, B=2, C=3, D=10 (an
    # outlier round). alpha=0.5 (target 50% coverage) chosen so the
    # ceil((n+1)*0.5) order statistic lands on round boundaries exactly -
    # see the module's own worked comment below for each fold's arithmetic.
    residuals_by_round = {
        "A": np.full(10, 1.0),
        "B": np.full(10, 2.0),
        "C": np.full(10, 3.0),
        "D": np.full(10, 10.0),
    }
    result = leave_one_round_out_conformal_check(residuals_by_round, alpha=0.5)

    assert set(result.per_round.keys()) == {"A", "B", "C", "D"}

    # held_out=A: calib pool = B+C+D (30 values: ten 2s, ten 3s, ten 10s).
    # k=ceil(31*0.5)=16 -> 16th smallest (0-idx 15) falls in the "3" block
    # (indices 10-19) -> quantile=3.0. A's own residuals (all 1.0) <= 3.0
    # -> fully covered.
    assert result.per_round["A"]["quantile"] == pytest.approx(3.0)
    assert result.per_round["A"]["coverage"] == pytest.approx(1.0)

    # held_out=B: calib pool = A+C+D -> same arithmetic, quantile=3.0.
    # B's residuals (2.0) <= 3.0 -> fully covered.
    assert result.per_round["B"]["quantile"] == pytest.approx(3.0)
    assert result.per_round["B"]["coverage"] == pytest.approx(1.0)

    # held_out=C: calib pool = A+B+D (ten 1s, ten 2s, ten 10s) -> index15
    # falls in the "2" block -> quantile=2.0. C's residuals (3.0) > 2.0 ->
    # NOT covered.
    assert result.per_round["C"]["quantile"] == pytest.approx(2.0)
    assert result.per_round["C"]["coverage"] == pytest.approx(0.0)

    # held_out=D: calib pool = A+B+C (ten 1s, ten 2s, ten 3s) -> index15
    # falls in the "2" block -> quantile=2.0. D's residuals (10.0) > 2.0 ->
    # NOT covered.
    assert result.per_round["D"]["quantile"] == pytest.approx(2.0)
    assert result.per_round["D"]["coverage"] == pytest.approx(0.0)

    # Pooled: (10+10+0+0) covered out of 40 = 0.5 - matches target exactly
    # in this constructed example (illustrates pooled vs per-round coverage
    # can differ a lot even when pooled matches nominal).
    assert result.pooled_coverage == pytest.approx(0.5)
    assert result.n_test_total == 40

    # final_quantile fit on all 40 pooled: ten 1s, ten 2s, ten 3s, ten 10s.
    # n=40, alpha=0.5 -> k=ceil(41*0.5)=21 -> index20 falls in the "3" block
    # (indices 20-29) -> quantile=3.0.
    assert result.final_quantile.quantile == pytest.approx(3.0)
    assert result.final_quantile.n_calibration == 40
    assert result.final_quantile.exact is True


def test_leave_one_round_out_conformal_check_raises_on_fewer_than_two_rounds():
    with pytest.raises(ValueError, match="at least 2 rounds"):
        leave_one_round_out_conformal_check({"A": np.array([1.0, 2.0])}, alpha=0.10)


def test_k_fold_conformal_check_covers_every_input_exactly_once():
    rng = np.random.default_rng(0)
    abs_residuals = rng.normal(loc=0.0, scale=1.0, size=200)

    result = k_fold_conformal_check(abs_residuals, alpha=0.10, k=5, seed=42)

    assert result.k == 5
    assert len(result.per_fold) == 5
    assert sum(f["n_test"] for f in result.per_fold) == 200
    assert result.n_test_total == 200
    # Sanity bound, not an exact value (fold membership depends on the RNG
    # shuffle) - with 200 i.i.d. points and a 90% target, pooled empirical
    # coverage should land comfortably above a loose floor, never above 1.0.
    assert 0.75 <= result.pooled_coverage <= 1.0
    assert np.isfinite(result.final_quantile.quantile)
    assert result.final_quantile.n_calibration == 200


def test_k_fold_conformal_check_raises_on_k_less_than_two():
    with pytest.raises(ValueError, match="k must be"):
        k_fold_conformal_check(np.ones(20), alpha=0.10, k=1)


def test_k_fold_conformal_check_raises_on_too_few_residuals_for_k():
    with pytest.raises(ValueError, match="Only"):
        k_fold_conformal_check(np.ones(5), alpha=0.10, k=5)  # needs >= 10


def test_build_interval_is_symmetric_around_point_prediction():
    lower, upper = build_interval(90.0, quantile=1.5)
    assert lower == pytest.approx(88.5)
    assert upper == pytest.approx(91.5)


def test_build_interval_works_elementwise_on_arrays():
    lower, upper = build_interval(np.array([90.0, 100.0]), quantile=2.0)
    np.testing.assert_allclose(lower, [88.0, 98.0])
    np.testing.assert_allclose(upper, [92.0, 102.0])
