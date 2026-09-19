# Feature Engineering — Phase 2

## Two gaps found in the Phase 1 data before any feature could be built

Phase 1 saved `session.laps` to `data/raw/<year>/*.parquet` and nothing else,
even though `f1qp.data.loader.load_session` calls
`session.load(laps=True, telemetry=..., weather=True, messages=False)` for
every session. Two consequences, both discovered while inspecting real
downloaded files at the start of Phase 2:

1. **Weather was fetched but never saved.** `session.weather_data` (a
   ~1/min time series of `AirTemp`, `TrackTemp`, `Humidity`, `Pressure`,
   `Rainfall`, `WindDirection`, `WindSpeed`) lives on the `Session` object
   FastF1 returns, but `download_2026.py` / `download_historical.py` only
   ever pulled `.laps` off it before writing to parquet. None of the 81
   raw files contain a single weather column.
2. **The qualifying target itself was never saved.** The model's actual
   prediction target is each driver's official Q1/Q2/Q3 time, which FastF1
   exposes via `session.results` (columns `Q1`, `Q2`, `Q3`, one row per
   driver) after loading a `Q` session. Only the raw lap-by-lap rows for
   the `Q` session were saved - not the classified segment times. Raw laps
   alone don't reliably tell you where Q1 ends and Q2 begins (that's a
   timing-system decision, not a fixed lap count), so this can't be
   reconstructed after the fact from what's on disk.

Both are fixed the same way: a small extraction script that reopens each
already-cached session (`fastf1.get_session(...).load(...)`) and pulls the
one extra thing off it. Because FastF1's disk cache (`data/cache/`, ~400MB)
already holds every session's raw API responses from the Phase 1 backfill,
re-loading weather or results is a **cache hit - no new network calls**,
and is fast even across all 82 training weekends. Telemetry is the
exception: Phase 1 deliberately never requested it
(`telemetry=False` everywhere), so `scripts/extract_telemetry.py` is the
one Phase 2 script that needs the network, and only for a small
representative subset of laps (see below), not a full backfill.

Weather is intentionally **not** merged into the lap-level parquet files.
It's a session-level time series (~90-120 samples per session vs.
15-30 laps per driver); joining it onto every lap row would repeat the
same handful of session aggregates dozens of times for no benefit. It's
extracted straight to per-session aggregates in
`data/processed/weather.parquet`, keyed by `(Year, RoundNumber,
SessionCode)`, and joined onto the feature table at assembly time.

## Practice-session input: what one LSTM timestep looks like

The model's input per weekend is a sequence of practice sessions (up to 3:
FP1/FP2/FP3 normal, FP1/SQ or FP1/SS sprint - see `f1qp.config`). Each
sequence step is one feature vector **per driver, per practice session**.
The sprint-weekend case has only 2 real sessions; the third slot is padded
using an explicit `session_type` indicator per `docs/data_strategy.md`'s
existing design decision, so the model can tell "no FP3 this weekend" apart
from "an FP3 that happened to look empty."

## Feature list (18 features per practice-session step)

Grouped by what they capture, all derived from columns that actually exist
in the saved laps (`Driver, LapTime, LapNumber, Stint, Compound, TyreLife,
Sector1-3Time, SpeedI1/I2/FL/ST, IsPersonalBest, FreshTyre, TrackStatus,
Position, Deleted, IsAccurate`) plus the two extracted sources above.

**Pace (4)**
1. `best_lap_time` - fastest clean flying lap (see run identification) in
   the session, in seconds.
2. `gap_to_session_best` - `best_lap_time` minus the best lap set by anyone
   in that session. Absolute lap time swings with fuel load, track
   evolution and weather; the gap to the field's benchmark is the more
   stable signal and is what actually correlates with qualifying pace.
3. `median_flying_lap_time` - robust central pace, less sensitive than the
   single best lap to one exceptional out-of-nowhere effort.
4. `lap_time_std` - consistency across flying laps; a tight session and a
   scrappy one can share the same best lap.

**Run structure (4)**
5. `n_runs` - distinct runs identified by `f1qp.features.runs` (stint
   changes + in-stint time-gap breaks, e.g. a red flag that doesn't force a
   pit stop).
6. `n_flying_laps` - laps that are neither an out-lap, an in-lap, nor
   deleted/inaccurate; the actual signal-bearing laps in the session.
7. `avg_run_length` - laps per run; short choppy runs vs. long green-flag
   runs are different session shapes.
8. `longest_run_length` - length of the single longest run, a proxy for
   how much long-run/race-sim work vs. pure qualifying-sim work happened.

**Tyre & fuel state (4)**
9. `compound_on_best_lap` - encoded compound (ordinal: hard=0, medium=1,
   soft=2) of the fastest flying lap; softer compound partly explains a
   quicker time on its own.
10. `tyre_life_on_best_lap` - laps on that tyre when the best lap was set;
    a banker lap late in a long stint is not the same signal as lap 2 on a
    fresh set.
11. `fuel_corrected_pace` - `best_lap_time` adjusted by
    `f1qp.features.fuel`'s learned seconds-per-lap fuel effect times the
    estimated fuel burned off since the run started. This is the feature
    that most directly answers "how fast would this lap have been at
    qualifying (near-empty tank) fuel load."
12. `long_run_avg_pace` - mean lap time over the longest run's flying laps,
    tyre-life-adjusted; a race-pace signal that's informative context even
    though the target is qualifying pace, not race pace.

**Sector & speed (3)**
13. `best_sector1_time`, 14. `best_sector2_time`, 15. `best_sector3_time` -
    best sector times don't have to come from the same lap as
    `best_lap_time` (a driver can lose a lap to traffic in one sector while
    still setting personal-best splits elsewhere); the theoretical-best
    composite is a cleaner true-pace signal than any single timed lap.

**Track evolution & conditions (2)**
16. `best_lap_session_position` - where in the session (0=start, 1=end) the
    best lap fell; track grip typically improves through a session, so
    *when* the best lap was set matters, not just what it was.
17. `air_temp_mean` / `track_temp_mean` / `rainfall_share` - collapsed into
    one row here but three real columns from `f1qp.features.weather`;
    track temp in particular has a well-known, large effect on tyre grip
    and lap time.

**Telemetry trend (1, aggregate of 3 sub-features)**
18. `throttle_full_pct`, `braking_events_per_lap`, `avg_corner_speed` (from
    `SpeedI1`/`SpeedI2`, the two intermediate speed traps, as a
    low-cost proxy alongside the telemetry-derived ones) on the session's
    fastest lap - how the lap was actually driven, not just how fast it
    was. Computed only for the representative lap per run (fastest lap per
    run, not every lap), per the Phase 1 telemetry-cost decision.

Global (non-per-timestep) context features carried alongside the sequence,
one per weekend rather than one per practice session:
- `is_sprint` (already in the raw data as `IsSprint`)
- `era` - 0 for 2023-2025, 1 for 2026, addressing the regulation-shift risk
  flagged in `docs/data_strategy.md`; cheap to add now, and Phase 3 needs
  season-stratified validation to trust an aggregate MAPE across both eras.

That's 15 practice-session-step features + 3 weather sub-features + the 3
telemetry sub-features counted as one line above = 18 distinct numeric
columns per (driver, session) row, plus the 2 global context columns - in
the 15-20 range the project brief calls for.

## Run identification (`f1qp.features.runs`)

A **run** is a stretch of consecutive laps on one set of tyres with no
break long enough to change the driving context (not a fresh pit-lane
stint, and not a red-flag/session-break pause while still out on track).
Two independent boundary rules, either one starts a new run:

- `Stint` changes (a real pit stop - the raw data already carries this).
- The gap between one lap's start time and the previous lap's start time
  is more than `GAP_MULTIPLIER` (default 2.5x) the session's own median
  lap time, and it isn't already explained by `PitOutTime`/`PitInTime` on
  the boundary lap. This catches a red flag or a long blocked-track pause
  that doesn't show up as a stint change (cars stay out on Stint N but sit
  stationary for minutes).

Within a run, a lap is a **flying lap** only if all of: it isn't the run's
first lap (out-lap), it isn't followed immediately by `PitInTime` on the
*next* lap (in-lap), `Deleted` is not true, and `IsAccurate` is true. This
mirrors the exact out-lap/in-lap reasoning already validated in Phase 1's
null-`LapTime` investigation (`docs/data_strategy.md`), applied at the
per-lap level instead of the per-session aggregate level.

## Fuel burn factor (`f1qp.features.fuel`)

Learned, not assumed, from 2026 data only (`Year == 2026`), per the
project's explicit constraint. Method: take every flying lap inside a run
of at least `MIN_RUN_LENGTH` (default 5) laps, under green-flag conditions
(`TrackStatus == "1"`), and regress `LapTime` on `lap_in_run` (1, 2, 3...)
with `TyreLife` and `Compound` as controls, **after removing each
session's own baseline pace** (`(Year, RoundNumber, SessionCode)`
fixed effects, via demeaning). The `lap_in_run` coefficient is
contaminated by three things pulling in different directions: fuel
burn-off (car gets lighter, laps get faster - the thing we want),
tyre degradation (rubber wears, laps get slower - controlled for via
`TyreLife`), and circuit/session baseline pace (which has nothing to do
with either, but swings 30+ seconds round to round on real 2026 data -
easily enough to swamp a ~0.03-0.08s/lap true effect and even flip its
sign if left in). The session-level fixed effects remove that third
source.

**This module's first version didn't have that third control** - it
regressed on the pooled data with a single global intercept, and did
produce a sign-flipped, physically-backwards coefficient on real 2026
practice data (traced and confirmed against actual downloaded laps, not
just suspected from theory). The obvious-looking fix - demean within
`(Driver, RunId)` (run-level fixed effects) instead - turned out to make
things worse, not better: `TyreLife` and `lap_in_run` both just count laps
elapsed on the same tyre set, so within a single run they're already
substantially correlated, and removing all between-run variation collapses
the fuel coefficient to statistical noise. Session-level fixed effects
remove the actual confound (baseline pace) while preserving the
between-run, within-session variation that's what lets the fuel and tyre
effects be told apart at all. Full derivation, including the real
correlation numbers that ruled out the run-level fix, is in
`f1qp.features.fuel`'s module docstring - read it before changing this
regression again.

Because the true effect is small and only 11 rounds of 2026 data back it,
the result also carries a reliability guardrail (a t-stat threshold and a
run-level cluster bootstrap sign-consistency check, both deliberately
conservative) rather than being trusted on its point estimate alone.
`FuelBurnResult.effective_seconds_per_lap` - not `.seconds_per_lap` - is
what downstream code uses: it's `0.0` (no correction applied) when the
estimate doesn't clear the guardrail, so an unreliable fit degrades to "no
correction" for every row instead of injecting a wrong number into all of
them. On the two 2026 rounds available in this session, the guardrail
correctly rejects the estimate (still sign-flipped even after the
session-level fix, on just 2 circuits) rather than reporting it - that's
the guardrail doing its job, not a remaining bug; it should be re-checked
once `build_features.py` runs against the full 11-round dataset, which has
enough between-run, within-session variation for this to plausibly clear
the bar.

This only needs laps data that's already on disk - no re-download.

## What still needs to run outside this session

Everything above is implemented and unit-tested against synthetic data
matching FastF1's real schemas, plus sanity-checked against the two 2026
lap files already available in this session. Four scripts need to run in
Damiano's own terminal (network + the full local dataset, same pattern as
Phase 1's downloads):

- `scripts/extract_weather.py` - cache hit, all 82 training weekends + R12,
  should take well under a minute.
- `scripts/extract_qualifying_targets.py` - cache hit, same speed.
- `scripts/extract_telemetry.py` - the one script that needs fresh network
  calls, for the representative-lap subset only (one fastest lap per run,
  not every lap of every session). Budget real time for this one.
- `scripts/build_features.py` - pure local computation once the three
  outputs above exist; joins everything into
  `data/processed/features.parquet`.
