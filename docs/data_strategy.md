# Data Strategy

## Sources

All data comes from FastF1, which wraps the F1 live-timing API and the
Ergast historical archive. Every pull goes through `f1qp.data.loader`,
which enables FastF1's disk cache first (`f1qp.data.cache.enable_cache`)
so nothing is fetched twice.

## Train / test split

| Split | Weekends | Purpose |
|---|---|---|
| Train | 2023, 2024, 2025 full seasons (72 weekends) + 2026 R1-R11 (11 weekends) | Model fitting |
| Offline test | 2026 R12 - Dutch GP, Zandvoort (Aug 21-23, 2026) | Held-out validation before touching Round 13 |
| Live | 2026 R13 - Italian GP, Monza (Sep 4-6, 2026) | Real deployment target |

Round 12 is a genuine held-out round, not cross-validation - it's excluded
from training entirely and only scored once after the fact. It also
happens to be a sprint weekend, so it doubles as the sprint-format test
case; Round 13 is a standard weekend, so the live path is the simpler of
the two formats.

## Normal vs. sprint weekends

FastF1's own event schedule is the source of truth for which format a
round uses (`f1qp.config.get_event_info`) - round-to-format is not
hardcoded, because the calendar has already changed once this season
(Bahrain and Saudi Arabia were dropped). A normal weekend contributes
FP1/FP2/FP3 as the practice sequence; a sprint weekend contributes
FP1/Sprint-Qualifying, which is shorter and run under different rules
(parc fermé applies earlier, less long-run data). The model's practice-
sequence input is padded to a fixed length either way, but padding uses an
explicit session-type flag rather than silent zeros - a zero-padded FP3
slot with no indicator risks teaching the LSTM that sprint weekends look
like incomplete normal ones, rather than a distinct, expected pattern.

## What gets pulled, and what doesn't

Every session load defaults to `laps=True, weather=True, telemetry=False`.
Full car telemetry (throttle/brake/speed channels at several Hz, per car,
per lap) is the single biggest driver of both download time and disk
usage across 83+ weekends, and only Phase 2's telemetry-trend feature
needs it - for a subset of laps, not the full historical backfill. That
call site turns `telemetry=True` on explicitly; everything else stays
lightweight.

The historical backfill (`scripts/download_historical.py`) is written to
run as a background job across a day or two rather than inside one
sitting: it's idempotent (already-saved rounds are skipped) and logs and
skips individual failures instead of aborting the whole run.

## Known data risks

The ones specific to this phase:

- **Session irregularities.** Red-flagged or weather-shortened sessions can
  leave missing Q3 times or unusually thin qualifying fields for some
  drivers. `f1qp.data.schema.validate_laps` flags any qualifying session
  with fewer than 15 drivers represented, and `scripts/validate_schema.py`
  turns every flag into a line in `docs/data_quirks.md` after each
  download batch - the point is to have these written down before feature
  engineering starts, not discovered mid-Phase-2.
- **2023-25 vs. 2026 regulation shift.** 72 of 82 training weekends predate
  the 2026 car regulations. Every saved lap row carries `Year` and
  `RoundNumber`, so a season/era feature and season-stratified validation
  are straightforward to add in Phase 2 - worth doing before trusting an
  aggregate MAPE.

## Calibrating "is this session clean"

The null-`LapTime` check went through three iterations before landing on
something trustworthy, because a lap with no `LapTime` isn't a defect -
out-laps, in-laps, and laps cut short by a red flag or safety car never
complete a timed lap:

1. **Flat 5% bar.** Wrong - flagged 81/81 downloaded files. Normal running
   produces far more untimed laps than 5% almost everywhere.
2. **Fixed per-session-type bar (Q ≤35%, FP ≤55%).** Still a guess, just a
   less wrong one - flagged 40/302 sessions, but 34 of those were Q
   sessions clustered tightly at 35-43% across every season. That turned
   out to be Q's *actual* normal range (every push lap brackets an
   out-lap and an in-lap), not an anomaly.
3. **Data-driven (current).** Two layers: `f1qp.data.schema.validate_laps`
   keeps a loose fixed ceiling (FP/SQ ≤75%, Q ≤55%) used only at download
   time, one session at a time, before there's a dataset to compare
   against - it exists purely to catch something obviously dead.
   `scripts/validate_schema.py` runs after a full batch and computes each
   session type's own normal range from the data just downloaded (Tukey's
   rule: flag past Q3 + 1.5×IQR), printing the computed ceiling into
   `docs/data_quirks.md` for transparency. This is the version to trust.

**Lesson for later phases:** don't hardcode a data-quality threshold from
intuition - derive it from the distribution once enough samples exist.
Two guessed numbers here were both wrong; the data-driven one wasn't.

## Confirmed session disruptions (Phase 1)

Running the final validator against all 81 downloaded rounds (302
sessions) left exactly 10 flagged. Every one maps to a real, identifiable
event rather than a data problem:

| Session | Null LapTime | Likely cause |
|---|---|---|
| 2023 Hungarian GP, FP1 | 48.6% | Marginal over the computed ceiling (45.0%) |
| 2023 Belgian GP, FP1 | 63.2% | Spa - weather/red-flag prone |
| 2023 Las Vegas GP, FP1 | 45.6% | The water-valve-cover incident that red-flagged the session |
| 2024 Japanese GP, FP2 | 49.3% | Marginal over the computed ceiling (45.0%) |
| 2024 Dutch GP, FP3 | 65.1% | Zandvoort - weather/red-flag prone |
| 2024 Azerbaijan GP, FP1 | 48.9% | Baku - barrier-lined circuit, crash-prone |
| 2025 Japanese GP, FP2 | 63.2% | Suzuka - session disruption |
| 2025 Azerbaijan GP, FP1 | 49.8% | Baku, same weekend as the Q flag below |
| 2025 Azerbaijan GP, Q | 60.9% | Baku - the only flagged **qualifying** session |
| 2026 Belgian GP, FP2 | 53.6% | Spa - consistent with the 2023 pattern |

Spa and Zandvoort recurring across multiple seasons, and Baku showing up
twice in the same weekend, is corroborating evidence the flagging is
finding real signal, not noise.

**Resolved:** the 2025 Azerbaijan Q flag mattered because Q is the model's
*target*, not an input feature - a heavily disrupted qualifying session
can leave a driver with no representative lap time for a segment at all,
which is a missing/corrupted label, not just noisy input. Verified
(Aug 2026): every driver has at least one timed lap in that session, via

```python
df = pd.read_parquet("data/raw/2025/r17_Azerbaijan_Grand_Prix.parquet")
q = df[df["SessionCode"] == "Q"]
q.groupby("Driver")["LapTime"].apply(lambda s: s.notna().sum())
```

No row needs dropping or imputing because of this - the 60.9% null rate is
out/in-lap noise like everywhere else, not a missing label. Phase 1 is
clear on this front.

## Where things land

```
data/
  cache/        FastF1's own cache - safe to delete, will be rebuilt
  raw/2026/     One parquet per 2026 round, all sessions concatenated
  raw/<year>/   Same, for 2023/2024/2025
  processed/    Phase 2 output - feature tables, not raw laps
```

`data/` is gitignored except for `.gitkeep` placeholders - the parquet
files are regenerable from the scripts in a few hours and don't belong in
version control.

See `docs/feature_engineering.md` for Phase 2: it turned up two things
this document's "what gets pulled" section didn't anticipate - weather was
requested here (`weather=True`) but never actually saved, and neither was
`session.results` (the Q1/Q2/Q3 classification the model needs as its
target). Both get fixed by a separate Phase 2 extraction step rather than
changing anything in `download_2026.py` / `download_historical.py`
retroactively.
