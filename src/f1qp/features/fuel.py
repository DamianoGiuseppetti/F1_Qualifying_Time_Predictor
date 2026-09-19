"""Fuel burn factor: learned from 2026 data via session-level fixed effects.

Two effects push a driver's lap time in opposite directions as a run goes
on: the car gets lighter as fuel burns off (faster), and the tyres degrade
(slower). `learn_fuel_burn_factor` isolates the fuel-only component by
regressing lap time on how far into the run a lap is (`lap_in_run`),
*controlling for* `TyreLife` and `Compound` so the tyre-wear component
doesn't leak into the fuel coefficient - AND by removing each session's own
baseline pace before fitting (see "Why session-level fixed effects" below),
so circuit-to-circuit and session-to-session pace differences don't leak
into it either.

What this deliberately does NOT claim: an absolute "qualifying fuel load"
correction. We have no ground truth for how much fuel any car is carrying
in a given practice run (that's a team secret, not in the FastF1 data), so
extrapolating "add back the fuel this car would burn between here and an
empty tank" would be exactly the kind of arbitrary assumption the project
brief rules out. Instead, `fuel_corrected_pace` corrects a lap back to
what it would have been at the *start* of its own run, holding tyre wear
fixed - removing the within-run fuel-burn trend so the model sees driving
pace rather than a mix of pace and how much fuel happened to be left when
the lap was set. That's a claim the data actually supports.

## Why session-level fixed effects, not run-level

An earlier version of this function pooled `lap_time_seconds` across every
round with a single global intercept. That's biased whenever a session's
baseline pace correlates with which `lap_in_run` values it tends to
contribute - e.g. a circuit that's easy on tyres runs long green-flag
stints (high `lap_in_run` values are common there) and happens to have a
different baseline pace than a circuit where stints stay short. Verified
against real 2026 practice data: per-round mean lap time swings by 30+
seconds while the true fuel effect is on the order of 0.03-0.08s/lap - more
than enough for that between-session confound to flip the coefficient's
sign.

The fix is NOT to demean within `(Driver, RunId)` (i.e. run-level fixed
effects), even though that's the more obvious "within" transform. Checked
directly against real data: `TyreLife` and `lap_in_run` both just count
laps elapsed on the *same* tyre set, so within a single run they're already
substantially correlated (~0.6, vs ~0.4 pooled) - demeaning by run removes
exactly the between-run variation that lets the fuel and tyre effects be
told apart, and collapses the fuel coefficient to statistical noise (a
point estimate indistinguishable from zero) rather than fixing its sign.

Demeaning by `(Year, RoundNumber, SessionCode)` instead removes the actual
confound (session/circuit baseline pace) while preserving the *between-run,
within-session* variation - different runs in the same session start at
different points in their tyre life and run to different lengths, which is
exactly the variation that separates "more laps of fuel burn" from "more
laps of tyre wear."

## Why this still isn't trusted blindly

The true effect is small relative to lap-time noise and only 11 rounds of
2026 data back it. So `learn_fuel_burn_factor` doesn't just report a
number - it reports a classical standard error/t-stat AND a run-level
cluster bootstrap sign-consistency share (laps within one run aren't
independent, so a naive per-lap SE understates the true uncertainty; the
bootstrap resamples whole runs to account for that). `is_reliable` is only
true if both clear a threshold and the sign is physically sensible.
Downstream code should use `result.effective_seconds_per_lap`, not
`result.seconds_per_lap` directly - it's 0.0 (no correction applied) when
the estimate doesn't pass the guardrail, so an unreliable session's worth
of data degrades to "no correction" instead of injecting noise into every
`fuel_corrected_pace` value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_RUN_LENGTH = 5  # shorter runs don't carry enough within-run fuel burn-off to be informative
COMPOUND_ORDER = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}
DEFAULT_COMPOUND_ORD = 1  # unknown/other compound falls back to "medium" as a neutral midpoint
GREEN_FLAG_TRACK_STATUS = "1"
MIN_LAPS_TO_FIT = 10
FE_GROUP_COLUMNS = ["Year", "RoundNumber", "SessionCode"]

# Reliability guardrail. Deliberately conservative on both counts: the classical
# t-stat is computed from a non-clustered SE that still ignores serial correlation
# within a run, so it understates true uncertainty - a threshold that would be
# unremarkable for an ordinary regression is not conservative enough here.
MIN_ABS_T_STAT = 2.5
MIN_BOOTSTRAP_SIGN_CONSISTENCY = 0.9
N_BOOTSTRAP = 300
BOOTSTRAP_SEED = 0


@dataclass
class FuelBurnResult:
    seconds_per_lap: float  # positive = laps get this many seconds faster per lap of fuel burned
    n_laps_used: int
    n_runs_used: int
    n_sessions_used: int
    standard_error: float  # of seconds_per_lap; classical (non-clustered) - a lower bound on true uncertainty
    t_stat: float  # seconds_per_lap / standard_error
    # fraction of run-level bootstrap resamples agreeing with the point estimate's sign
    bootstrap_sign_consistency: float
    # True only if the sign is physically sensible AND both diagnostics clear their threshold
    is_reliable: bool

    @property
    def effective_seconds_per_lap(self) -> float:
        """What downstream code should actually use: seconds_per_lap if reliable, else 0.0 (no correction)."""
        return self.seconds_per_lap if self.is_reliable else 0.0


def _regression_frame(laps_with_runs: pd.DataFrame, *, min_run_length: int) -> pd.DataFrame:
    required = [
        "Driver", "RunId", "IsFlyingLap", "LapNumber", "LapTime", "TyreLife", "Compound", *FE_GROUP_COLUMNS,
    ]
    missing = [c for c in required if c not in laps_with_runs.columns]
    if missing:
        raise ValueError(f"fuel regression is missing required columns: {missing}")

    df = laps_with_runs
    df = df[df["Year"] == 2026]
    df = df[df["IsFlyingLap"]]
    if "TrackStatus" in df.columns:
        df = df[df["TrackStatus"].astype(str) == GREEN_FLAG_TRACK_STATUS]
    df = df.dropna(subset=["LapTime", "TyreLife"])
    if df.empty:
        return df

    run_len = df.groupby(["Driver", "RunId"])["LapNumber"].transform("count")
    df = df[run_len >= min_run_length].copy()
    if df.empty:
        return df

    df["lap_in_run"] = df.groupby(["Driver", "RunId"])["LapNumber"].rank(method="first")
    df["lap_time_seconds"] = df["LapTime"].dt.total_seconds()
    df["compound_ord"] = (
        df["Compound"].astype(str).str.upper().map(COMPOUND_ORDER).fillna(DEFAULT_COMPOUND_ORD)
    )
    # Unique run id across sessions - the bootstrap resamples at this level, not per-lap,
    # because laps within one run are not independent observations.
    df["_run_key"] = list(zip(df["Year"], df["RoundNumber"], df["SessionCode"], df["Driver"], df["RunId"]))
    return df


def _demean_within_session(df: pd.DataFrame) -> pd.DataFrame:
    """The within/fixed-effects transform: subtract each session's own mean.

    Removes all between-session variation (circuit baseline pace, track
    evolution, conditions) while keeping the between-run variation within a
    session that's needed to separate the fuel and tyre effects.
    """
    g = df.groupby(FE_GROUP_COLUMNS)
    out = df.copy()
    for col in ["lap_in_run", "TyreLife", "compound_ord", "lap_time_seconds"]:
        out[col] = df[col] - g[col].transform("mean")
    return out


def _fit_within_ols(demeaned: pd.DataFrame, n_groups: int) -> tuple[float, float]:
    """OLS on already-demeaned data. Returns (lap_in_run coefficient, its standard error).

    Degrees of freedom for the residual variance are corrected for the fixed
    effects that demeaning implicitly estimates (one mean per session, per
    regressor) - using the naive n - k would understate sigma^2 and report
    a standard error tighter than it really is.
    """
    X = demeaned[["lap_in_run", "TyreLife", "compound_ord"]].to_numpy(dtype=float)
    y = demeaned["lap_time_seconds"].to_numpy(dtype=float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - k - n_groups, 1)
    sigma2 = (resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(np.diag(sigma2 * xtx_inv))[0])
    return float(beta[0]), se


def _bootstrap_sign_consistency(df: pd.DataFrame, point_beta: float, *, n_bootstrap: int, seed: int) -> float:
    """Run-level cluster bootstrap: resample whole runs (with replacement), re-demean
    and refit on each resample, and report the fraction of replicates whose
    coefficient has the same sign as the point estimate.

    This is what actually tests whether the estimate is stable, rather than
    an artifact of exactly which runs happened to land in this dataset - and
    it partially compensates for `_fit_within_ols`'s classical SE ignoring
    serial correlation within a run, since resampling at the run level
    respects that laps within a run move together.
    """
    if point_beta == 0:
        return 0.0
    run_keys = df["_run_key"].unique()
    grouped = dict(tuple(df.groupby("_run_key")))
    rng = np.random.default_rng(seed)

    same_sign = 0
    for _ in range(n_bootstrap):
        sampled_keys = rng.choice(np.asarray(run_keys, dtype=object), size=len(run_keys), replace=True)
        sample = pd.concat([grouped[k] for k in sampled_keys], ignore_index=True)
        demeaned = _demean_within_session(sample)
        n_groups = demeaned.groupby(FE_GROUP_COLUMNS).ngroups
        try:
            beta, _ = _fit_within_ols(demeaned, n_groups)
        except np.linalg.LinAlgError:
            continue
        if np.sign(beta) == np.sign(point_beta):
            same_sign += 1
    return same_sign / n_bootstrap


def learn_fuel_burn_factor(
    laps_with_runs: pd.DataFrame,
    *,
    min_run_length: int = MIN_RUN_LENGTH,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> FuelBurnResult:
    """Fit the fuel-burn effect on 2026 green-flag long runs via session-level fixed effects.

    See this module's docstring for why session-level (not run-level) fixed
    effects, and why the result also carries a reliability guardrail rather
    than being trusted outright. Raises ValueError if there isn't enough
    qualifying data (too few long, clean runs) to fit at all - a distinct
    failure mode from "fit, but not reliable" (`is_reliable=False`).
    """
    df = _regression_frame(laps_with_runs, min_run_length=min_run_length)
    if len(df) < MIN_LAPS_TO_FIT:
        raise ValueError(
            f"Not enough 2026 green-flag long-run laps to learn a fuel burn factor "
            f"(need >= {MIN_LAPS_TO_FIT}, got {len(df)})"
        )

    demeaned = _demean_within_session(df)
    n_sessions = df.groupby(FE_GROUP_COLUMNS).ngroups
    beta, se = _fit_within_ols(demeaned, n_sessions)
    # beta is d(lap_time)/d(lap_in_run); a real fuel effect makes laps FASTER as
    # lap_in_run increases, i.e. beta < 0, so seconds_per_lap = -beta > 0.
    seconds_per_lap = -beta
    t_stat = seconds_per_lap / se if se > 0 else 0.0

    bootstrap_consistency = _bootstrap_sign_consistency(
        df, beta, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )

    is_reliable = (
        seconds_per_lap > 0
        and abs(t_stat) >= MIN_ABS_T_STAT
        and bootstrap_consistency >= MIN_BOOTSTRAP_SIGN_CONSISTENCY
    )

    return FuelBurnResult(
        seconds_per_lap=seconds_per_lap,
        n_laps_used=len(df),
        n_runs_used=int(df.groupby(["Driver", "RunId"]).ngroups),
        n_sessions_used=n_sessions,
        standard_error=se,
        t_stat=t_stat,
        bootstrap_sign_consistency=bootstrap_consistency,
        is_reliable=is_reliable,
    )


def fuel_corrected_pace(lap_time_seconds: float, lap_in_run: int, fuel_burn_seconds_per_lap: float) -> float:
    """Add back the within-run fuel-burn saving so the lap reads as if it were the run's first lap.

    `lap_in_run` is 1-indexed (1 = the run's own out-of-fuel-burn-effect baseline).
    Pass `result.effective_seconds_per_lap`, not `result.seconds_per_lap`, as
    `fuel_burn_seconds_per_lap` - see FuelBurnResult.
    """
    return lap_time_seconds + fuel_burn_seconds_per_lap * (lap_in_run - 1)
