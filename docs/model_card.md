# Model card

## What it predicts

One number per driver per weekend: `final_quali_time` — the time that
actually set their grid position (Q3 if they reached it, else Q2, else
Q1, matching how F1 itself classifies an early elimination). The model
never sees Q1/Q2/Q3 as separate targets; the simplification happened
before training and is why there's one head, not three.

Input is that weekend's practice sessions (FP1-FP3, or FP1+SQ on a
sprint weekend) — lap times, run/stint structure, tyre compound,
telemetry trends, weather — plus two static flags, `is_sprint` and
`era` (0 for 2023-2025, 1 for 2026). 26 features in total; the full list
and how each was derived is in `docs/feature_engineering.md`.

## Architecture

Single-layer LSTM, 64 hidden units, packed/padded sequences so a
sprint weekend's missing third session is never zero-filled — the
model simply never processes a step that didn't happen. The final
hidden state is concatenated with the two static flags and passed
through a small feed-forward head (Linear → ReLU → Dropout(0.4) →
Linear) to one scalar output.

The model doesn't predict the lap time directly. It predicts the gap
to `practice_reference` (the fastest practice lap posted by anyone that
weekend), which gets added back to reconstruct the absolute time. Same
formulation an XGBoost baseline confirmed first (1.090% MAPE, R² 0.952)
before the LSTM was built on top of it — predicting an absolute lap
time directly failed outright (656,522% MAPE) because it has to relearn
each circuit's own baseline pace from scratch instead of just the
delta.

Trained with Huber loss (delta=1.0), same reasoning as the baseline:
a few real weekends in the training data are extreme outliers (see
Known limitations) and a squared-error loss would let them dominate.

## Training data

2023-2025 (era 0) plus 2026 rounds 1 through 12 (era 1), 1,624
driver-weekend rows after coalescing to `final_quali_time` and
dropping rows with no target (DNS/DSQ). Round 12 (Zandvoort) was
trained on only after it had already been used once as a genuine
held-out offline test — see Results below.

Retraining happens by hand, after each new round's official result is
available (`scripts/retrain_pipeline.py`) — not on a schedule or an
automatic trigger. Every retrain re-runs leave-one-round-out
cross-validation on era 1 first, so the reported metrics are always
freshly computed against whatever data exists at that point, never
numbers carried over from an earlier run.

## Results

Leave-one-round-out cross-validation, era 1 (2026), pooled across all
12 rounds — every round held out and predicted once, trained on the
other 11 plus all of era 0:

- MAPE: 0.929%, R²: 0.991
- 50% interval: ±0.613s · 68%: ±0.902s · 80%: ±1.219s · 90%: ±1.729s
  (split-conformal, recalibrated on every retrain)

Per-round MAPE ranges from about 0.46% to 3.89%. The one clear outlier
is round 10, Belgian GP (Spa) 2026: MAPE 3.89%, R² -5.95 (worse than
predicting the mean). Rain hit Friday practice and Hamilton crashed in
FP3 — a rushed repair before qualifying — so `practice_reference` for
that weekend reflects different conditions than qualifying itself,
breaking the gap formulation's core assumption. Kept in training rather
than removed (a real race weekend, not corrupted data).

Round 12 (Zandvoort) offline test — held out of training entirely,
predicted once, scored against the real result: 22/22 drivers, MAPE
0.517%, R² 0.825, 21/22 inside the 50% interval. The R² here looks
worse than the pooled number despite a much lower MAPE — that's a
low-variance artifact, not a worse prediction: Zandvoort's whole field
spanned only 3.4 seconds that weekend, so R²'s denominator (total
variance) was unusually small and a normal-sized error ate a bigger
share of it than it would on a normal-spread round.

Round 13 (Monza) — the first live, non-backtested prediction, launched
and scored through the deployed app: average error 0.4s, 13/22 drivers
inside the typical range. Monza qualifying is shaped by slipstream
trains down the straights; free practice running doesn't reproduce that
the same way, which is the likeliest reason the hit rate came in lower
than the calibration numbers above. One live round isn't enough to
confirm this as a pattern — worth watching at Spa and other
slipstream-heavy circuits going forward.

Round 14 (Spanish Grand Prix) — the second live round: 22 drivers
launched, 20 with an official time to compare against (2 had no
classified time to compare — the expected shape for a DNS/no-lap
driver, not a scoring bug, so the round still counts as fully checked
against the official result). Average
absolute error 0.62s across those 20, 15/20 (75%) inside the 50%
interval — closer to the pooled calibration numbers than Monza was.
Closest call: LIN, 1ms off. Biggest miss: BOT, 1.72s off. Two live
rounds isn't enough to call a trend, but the interval coverage landing
nearer target here is consistent with Monza's low hit rate being a
real, circuit-specific (slipstream) effect rather than the interval
being miscalibrated in general.

## Known limitations

**Per-round interval coverage is uneven.** The pooled 50% interval
sits right at its 51.2% empirical target, but individual 2026 rounds
swing from 13.6% to 95.5% coverage at that same level. The interval is
well-calibrated in aggregate, not round by round — a single weekend's
result being outside it isn't on its own evidence of a problem.

**`fuel_corrected_pace` is computed but unused.** The fuel-burn
regression it depends on was checked against the full dataset and
came back unreliable (a built-in guardrail rejects it rather than
shipping a bad correction silently), so this feature currently equals
raw `best_lap_time` for every row. Open, not resolved.

**Three known disruption weekends live in the training data, kept
deliberately:** Sao Paulo and Las Vegas (rain during qualifying itself
distorting the target) and Spa round 10 above (rain during practice
only, distorting the baseline instead). All three stay in training —
real race weekends, not something to filter out — but they're the
likely source of the model's worst per-round numbers.

**No timing check before fetching official results, only before
fetching practice data.** On a sprint weekend the session right after
practice is a Sprint race, not Qualifying, so a simple "next session"
heuristic would be wrong there. Rather than ship a heuristic that's
confidently wrong on sprint weekends, the app just asks for manual
confirmation before that particular fetch.

**Built for one user, not production multi-tenant serving.**
In-memory background jobs, a local SQLite MLflow store, no
authentication. Fine for what this is — a single person's live
prediction each race weekend — not a template for serving this to
multiple users at once.

## Intended use

A portfolio project, and Damiano's own live prediction for each 2026 GP
as it happens. Not built or validated for wagering, commercial
forecasting, or any other decision where a wrong prediction has real
financial consequences.
