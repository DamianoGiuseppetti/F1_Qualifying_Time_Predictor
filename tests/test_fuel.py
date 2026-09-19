import numpy as np
import pandas as pd
import pytest

from f1qp.features.fuel import fuel_corrected_pace, learn_fuel_burn_factor

TRUE_FUEL_SECONDS_PER_LAP = 0.06
TRUE_TYRE_DEG_SECONDS_PER_LAP = 0.05
BASE_LAP_SECONDS = 90.0


def _laps_from_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["LapTime"] = pd.to_timedelta(df["lap_time"], unit="s")
    df["LapNumber"] = df["lap_in_run"].astype(float)
    df = df.drop(columns=["lap_time"])
    df["IsFlyingLap"] = True
    df["TrackStatus"] = "1"
    return df


def _synthetic_single_session_long_runs(n_runs: int = 8, laps_per_run: int = 10, seed: int = 0) -> pd.DataFrame:
    """Laps shaped like real green-flag long runs, all from ONE session, with a KNOWN
    fuel effect baked in. With only one session there's nothing for session-level
    fixed effects to remove - this is the direct analogue of the original
    single-intercept test and should still recover the true effect.
    """
    rng = np.random.default_rng(seed)
    rows = []
    compounds = ["SOFT", "MEDIUM", "HARD"]
    for run_id in range(n_runs):
        driver = f"D{run_id % 3}"
        compound = compounds[run_id % 3]
        tyre_life_start = rng.integers(1, 5)
        for lap_in_run in range(1, laps_per_run + 1):
            tyre_life = tyre_life_start + lap_in_run
            lap_time = (
                BASE_LAP_SECONDS
                - TRUE_FUEL_SECONDS_PER_LAP * lap_in_run
                + TRUE_TYRE_DEG_SECONDS_PER_LAP * tyre_life
                + rng.normal(0, 0.02)
            )
            rows.append({
                "Driver": driver, "RunId": run_id, "lap_in_run": lap_in_run, "lap_time": lap_time,
                "TyreLife": float(tyre_life), "Compound": compound,
                "Year": 2026, "RoundNumber": 1, "SessionCode": "FP2",
            })
    return _laps_from_rows(rows)


def _synthetic_two_session_confound(seed: int = 7) -> pd.DataFrame:
    """Two sessions with a REAL between-session confound baked in on purpose: the
    slower-baseline session (115s) also happens to run much longer stints (higher
    lap_in_run values) than the faster-baseline session (85s) - a 30s baseline swing
    mirroring what real 2026 practice data showed across circuits. A single-intercept
    pooled fit gets fooled by this (composition effect dominates the tiny true slope);
    session-level fixed effects should not be.
    """
    rng = np.random.default_rng(seed)
    rows = []
    sessions = [
        {"RoundNumber": 1, "baseline": 85.0, "run_lengths": [4, 5, 4, 5, 4, 5]},
        {"RoundNumber": 2, "baseline": 115.0, "run_lengths": [12, 13, 12, 13, 12, 13]},
    ]
    run_id = 0
    for s in sessions:
        for rl in s["run_lengths"]:
            run_id += 1
            driver = f"D{run_id % 3}"
            tyre_start = rng.integers(1, 8)  # some runs continue an already-worn set
            for lap_in_run in range(1, rl + 1):
                tyre_life = float(tyre_start + lap_in_run - 1)
                lap_time = (
                    s["baseline"]
                    - TRUE_FUEL_SECONDS_PER_LAP * lap_in_run
                    + 0.04 * tyre_life
                    + rng.normal(0, 0.05)
                )
                rows.append({
                    "Driver": driver, "RunId": run_id, "lap_in_run": lap_in_run, "lap_time": lap_time,
                    "TyreLife": tyre_life, "Compound": "MEDIUM",
                    "Year": 2026, "RoundNumber": s["RoundNumber"], "SessionCode": "FP2",
                })
    return _laps_from_rows(rows)


def _naive_pooled_fit(laps: pd.DataFrame) -> float:
    """The OLD (buggy) method: single global intercept, no fixed effects at all -
    reproduced here directly (not via fuel.py) purely to demonstrate the confound
    this test's data is designed to expose, independent of whatever fuel.py does.
    """
    df = laps.copy()
    df["lap_time_seconds"] = df["LapTime"].dt.total_seconds()
    df["lap_in_run"] = df.groupby(["Driver", "RunId"])["LapNumber"].rank(method="first")
    X = np.column_stack([df["lap_in_run"], df["TyreLife"], np.ones(len(df))])
    y = df["lap_time_seconds"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return -float(beta[0])  # seconds_per_lap convention


def _synthetic_noise_only(seed: int = 3) -> pd.DataFrame:
    """Two small sessions, no real fuel or tyre effect at all - pure noise well within
    what a real session's lap-to-lap scatter looks like. Nothing here should pass the
    reliability guardrail.
    """
    rng = np.random.default_rng(seed)
    rows = []
    sessions = [
        {"RoundNumber": 1, "baseline": 90.0, "run_lengths": [5, 6]},
        {"RoundNumber": 2, "baseline": 92.0, "run_lengths": [5, 6]},
    ]
    run_id = 0
    for s in sessions:
        for rl in s["run_lengths"]:
            run_id += 1
            driver = f"D{run_id % 3}"
            tyre_start = rng.integers(1, 4)
            for lap_in_run in range(1, rl + 1):
                tyre_life = float(tyre_start + lap_in_run - 1)
                lap_time = s["baseline"] + rng.normal(0, 0.6)
                rows.append({
                    "Driver": driver, "RunId": run_id, "lap_in_run": lap_in_run, "lap_time": lap_time,
                    "TyreLife": tyre_life, "Compound": "MEDIUM",
                    "Year": 2026, "RoundNumber": s["RoundNumber"], "SessionCode": "FP1",
                })
    return _laps_from_rows(rows)


def test_learn_fuel_burn_factor_recovers_known_effect_single_session():
    laps = _synthetic_single_session_long_runs()
    result = learn_fuel_burn_factor(laps, n_bootstrap=50)
    assert result.seconds_per_lap == pytest.approx(TRUE_FUEL_SECONDS_PER_LAP, abs=0.015)
    assert result.n_laps_used == len(laps)
    assert result.n_runs_used == 8
    assert result.n_sessions_used == 1
    assert result.is_reliable


def test_learn_fuel_burn_factor_session_fixed_effects_survive_a_real_confound():
    laps = _synthetic_two_session_confound()

    naive_estimate = _naive_pooled_fit(laps)
    assert naive_estimate < 0, "sanity check on the test data itself: the confound should fool a single-intercept fit"

    result = learn_fuel_burn_factor(laps, n_bootstrap=200, bootstrap_seed=0)
    assert result.seconds_per_lap == pytest.approx(TRUE_FUEL_SECONDS_PER_LAP, abs=0.03)
    assert result.seconds_per_lap > 0
    assert result.n_sessions_used == 2
    assert result.is_reliable


def test_learn_fuel_burn_factor_flags_noisy_estimate_as_unreliable():
    laps = _synthetic_noise_only()
    result = learn_fuel_burn_factor(laps, min_run_length=4, n_bootstrap=200, bootstrap_seed=0)
    assert not result.is_reliable
    assert result.effective_seconds_per_lap == 0.0


def test_fuel_burn_result_effective_seconds_per_lap_matches_point_estimate_when_reliable():
    laps = _synthetic_single_session_long_runs()
    result = learn_fuel_burn_factor(laps, n_bootstrap=50)
    assert result.is_reliable
    assert result.effective_seconds_per_lap == result.seconds_per_lap


def test_learn_fuel_burn_factor_ignores_non_2026_data():
    laps = _synthetic_single_session_long_runs()
    laps["Year"] = 2025  # everything is historical - project constraint says 2026 only
    with pytest.raises(ValueError, match="Not enough 2026"):
        learn_fuel_burn_factor(laps)


def test_learn_fuel_burn_factor_ignores_short_runs():
    laps = _synthetic_single_session_long_runs(laps_per_run=3)  # below MIN_RUN_LENGTH default of 5
    with pytest.raises(ValueError, match="Not enough 2026"):
        learn_fuel_burn_factor(laps)


def test_learn_fuel_burn_factor_excludes_non_green_flag_laps():
    laps = _synthetic_single_session_long_runs()
    laps.loc[laps.index[::3], "TrackStatus"] = "4"  # safety car on a third of laps
    result = learn_fuel_burn_factor(laps, n_bootstrap=50)
    assert result.n_laps_used < len(laps)
    assert result.seconds_per_lap == pytest.approx(TRUE_FUEL_SECONDS_PER_LAP, abs=0.03)


def test_learn_fuel_burn_factor_raises_with_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        learn_fuel_burn_factor(pd.DataFrame({"Driver": ["VER"]}))


def test_learn_fuel_burn_factor_raises_without_fixed_effect_group_columns():
    laps = _synthetic_single_session_long_runs().drop(columns=["RoundNumber"])
    with pytest.raises(ValueError, match="missing required columns"):
        learn_fuel_burn_factor(laps)


def test_fuel_corrected_pace_adds_back_within_run_saving():
    # A lap set on lap 5 of its run, with a 0.06s/lap fuel effect, should be
    # corrected UP by 4 * 0.06s to read as if it were lap 1 of that run.
    corrected = fuel_corrected_pace(lap_time_seconds=88.5, lap_in_run=5, fuel_burn_seconds_per_lap=0.06)
    assert corrected == pytest.approx(88.5 + 4 * 0.06)


def test_fuel_corrected_pace_is_a_no_op_on_the_runs_first_lap():
    assert fuel_corrected_pace(90.0, lap_in_run=1, fuel_burn_seconds_per_lap=0.06) == pytest.approx(90.0)


def test_fuel_corrected_pace_is_a_no_op_with_zero_factor():
    # This is exactly what happens downstream when is_reliable is False -
    # effective_seconds_per_lap is 0.0, so the "correction" must do nothing.
    assert fuel_corrected_pace(90.0, lap_in_run=5, fuel_burn_seconds_per_lap=0.0) == pytest.approx(90.0)
