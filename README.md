# F1 Qualifying Predictor

**Live app: https://f1qp-render-deploy.onrender.com**

An LSTM that predicts each driver's final qualifying time, whichever
segment actually set their grid position (Q1, Q2 or Q3), from that
weekend's practice session data. It runs against the real, live 2026
season rather than a fixed historical dataset: the point is to predict a
round *before* qualifying happens, then check the prediction against the
real result once it does.

I've followed F1 long enough that "who's fastest in FP2" already feels
like a real question by Saturday morning, not just a warm-up for the
race. This project is my attempt to answer it with something more
rigorous than a hunch: a sequence model trained on three seasons of
practice data, served through a FastAPI + React app that fetches new
rounds on its own, retrains itself after every weekend, and keeps a
human in the loop before any new model actually goes live.

FastAPI backend, React (Vite) frontend, MLflow tracking, all served from
one Docker image, currently deployed free on Render.com.

## Data

Every session comes from [FastF1](https://github.com/theOehrly/Fast-F1),
which wraps the F1 live-timing API and the Ergast historical archive.
`f1qp.data.loader` enables FastF1's own disk cache before anything is
pulled, so nothing gets fetched twice, and every session load skips full
telemetry by default (`laps=True, weather=True, telemetry=False`) since
per-car, per-lap telemetry at several Hz is the single biggest driver of
download time and disk usage across 80+ weekends, and only one feature
below actually needs it.

| Split | Weekends | Purpose |
|---|---|---|
| Train | 2023-2025 full seasons (72 weekends) + every 2026 round with a confirmed result | Model fitting |
| Offline test | 2026 Round 12, Dutch GP (Zandvoort) | Held out completely, scored once, then folded into training |
| Live | 2026 Round 13 (Monza) onward | Real deployment, checked against results as each weekend happens |

Two things got missed the first time the season's data was pulled, and
both are worth knowing about if you're replicating this: weather was
requested from the API but never actually written to disk, and neither
was `session.results`, the official Q1/Q2/Q3 classification the model
needs as its *target*. Both get pulled from FastF1's own cache (a cache
hit, no repeat network calls) by two small extraction scripts
(`scripts/extract_weather.py`, `scripts/extract_qualifying_targets.py`)
rather than changing the original download scripts after the fact.

A sprint weekend (FP1 + a sprint qualifying session) is a genuinely
different shape from a normal one (FP1/FP2/FP3), not just a shorter
version of it, so the model's input sequence is padded to a fixed length
using an explicit session-type flag rather than silent zeros. Which
format a given round uses is read from FastF1's own event schedule, not
hardcoded, since the 2026 calendar has already changed once this season.

Every download batch is checked for irregularities: `scripts/validate_schema.py`
computes each session type's own normal range for missing lap times
(Tukey's rule, not a guessed threshold) and writes every real anomaly to
`docs/data_quirks.md`. Ten sessions across 81 rounds get flagged, and all
ten map to an identifiable real event (Spa and Zandvoort's usual weather
and red-flag pattern, the Las Vegas water-valve-cover incident, Baku's
barriers), not a data problem. Full derivation in `docs/data_strategy.md`.

## Feature engineering

Each practice session contributes one feature vector per driver, so a
weekend's model input is a short sequence (up to three steps: FP1/FP2/FP3,
or FP1 + one sprint session on a sprint weekend). 18 numeric features per
step, grouped by what they capture:

- **Pace** (4): fastest clean lap, gap to the session's best lap, median
  flying-lap time, lap-time standard deviation.
- **Run structure** (4): number of distinct runs, number of real flying
  laps, average and longest run length.
- **Tyre & fuel state** (4): compound on the best lap, tyre age when it
  was set, a fuel-corrected pace estimate, and long-run average pace.
- **Sector & speed** (3): best sector 1/2/3 times, which don't have to
  come from the same lap as the overall best.
- **Track evolution & conditions** (2): where in the session the best lap
  landed (grip usually improves through a session), plus air/track
  temperature and rainfall share.
- **Telemetry trend** (1, from three sub-signals): throttle-full percent,
  braking events per lap, and average corner speed on the fastest lap
  only, not every lap, to keep the one feature that needs real telemetry
  cheap.

Two global, once-per-weekend features ride alongside the sequence:
whether it's a sprint weekend, and an `era` flag (0 for 2023-2025, 1 for
2026) so the model can separate the pre- and post-2026-regulation cars
instead of averaging across a real rules change.

**Known limitation, stated plainly:** the fuel-correction feature is
currently a no-op. It's designed to convert a practice lap into "what
this lap would have looked like at qualifying fuel load," but the
underlying regression carries a deliberately conservative reliability
guardrail (a t-stat threshold plus a cluster-bootstrap sign-consistency
check), and on the 2026 data available so far it hasn't cleared that bar
(the coefficient still comes out sign-flipped on too few rounds). The
guardrail is doing its job, not failing quietly: every row just falls
back to the uncorrected lap time until there's enough 2026 data for the
fuel effect to actually be told apart from tyre degradation and circuit
baseline pace. Full derivation, including the run-level fix that made it
worse rather than better, is in `f1qp.features.fuel`'s own module
docstring and `docs/feature_engineering.md`.

## Model & results

An LSTM over the practice-session sequence, with split-conformal
prediction intervals recalibrated on every retrain. An XGBoost baseline
(1.09% MAPE, R² 0.952) was built first and beaten; a naive model that
tries to predict lap time directly, with no sequence structure, fails
outright (656,522% MAPE) because it has to relearn circuit-to-circuit
pace swings from nothing every time. Current production numbers,
verified live against the deployed app:

- Trained on 1,666 driver-weekend rows: 2023-2025 plus every 2026 round
  with a confirmed result, through Round 14.
- Pooled leave-one-round-out MAPE: 0.917%. Pooled R²: 0.990.
- 50% prediction interval: ±0.650s (split-conformal, recalibrated every
  retrain).

Two live rounds since deployment, both launched and scored through the
app itself, not backtested:

- **Round 13, Monza**: average error 0.4s, 13/22 drivers inside the
  typical range. The lower hit rate here is probably not noise: Monza
  qualifying is shaped by slipstream trains down the straights, which
  practice running doesn't reproduce the same way.
- **Round 14, Spanish GP**: 22 drivers launched, 20 with an official time
  to compare against (2 had no classified time, a normal DNS/no-lap
  outcome, not a scoring bug). Average absolute error 0.62s across those
  20, 15/20 (75%) inside the 50% interval, closer to the pooled
  calibration numbers than Monza was. Closest call: LIN, 1ms off.
  Biggest miss: BOT, 1.72s off.

Round 12 (Zandvoort) was held out of training entirely and scored once
as a genuine offline test before being folded back in: 22/22 drivers,
MAPE 0.517%, 21/22 inside the 50% interval.

**Known limitations:** per-round interval coverage is uneven even though
the pooled number sits right at its empirical target (individual 2026
rounds swing from 13.6% to 95.5% coverage at the same confidence level),
and one round is a clear outlier: Round 10, Belgian GP (Spa), at 3.89%
MAPE and R² -5.95. Full round-by-round breakdown, the fuel-feature
no-op above, and everything else known to still be off:
`docs/model_card.md`.

## MLOps

**Tracking and versioning.** Every retrain logs one MLflow run (local
SQLite-backed store, no server to run) and registers the model under a
single registered name, so MLflow bumps the version automatically on
each successful run. This sits alongside, not instead of, a plain
filesystem archive of every past model under `models/lstm/history/`.

**Stage, don't serve.** Checking a round's official result no longer
requires a manual retrain: it automatically chains into rebuilding the
dataset and retraining, in-process (not a subprocess, deliberately, so a
512MB free-tier instance doesn't have to hold two full copies of
torch/pandas at once mid-retrain). But an automatic retrain does not
mean an automatic promotion. The output always lands in
`models/lstm/staged/<timestamp>/` with a before/after comparison
(training rows, MAPE, R², interval width) attached, and the app keeps
serving whatever it was already serving until someone clicks **Promote
to production**. That one explicit action swaps the live model, archives
the previous one into `models/lstm/history/`, and reloads the running
process's weights immediately, no restart needed.

**A separate, manual retrain.** `scripts/retrain_from_hf.py` is
one command, run on my own machine: it pulls the current data down from
Hugging Face, rebuilds the dataset, retrains, and pushes the staged
candidate and its MLflow run back up. It verifies the push actually
landed before touching anything, and only then deletes its own local
copies, so nothing accumulates on disk between retrains. Full walkthrough
in `docs/Model_Retrain_Runbook.pdf`.

**Durable runtime storage.** Render's own disk doesn't survive a
redeploy or restart, so everything the app writes while running
(launched predictions, fetched features and targets, staged and
promoted models, the MLflow store) mirrors to a private Hugging Face
dataset repo. It's a no-op unless both `HF_DATASET_REPO` and `HF_TOKEN`
are set, so local `docker-compose` runs exactly as before this existed.

**Access control.** A single `ADMIN_TOKEN` gates Launch, on-demand data
fetches, and Promote. Every read (Preview, History, Performance) stays
open to anyone; a deployment with no token set behaves exactly like the
gate isn't there at all.

**CI.** GitHub Actions runs lint, test, and a Docker build on every push
to `main`.

## The application

Three tabs, sharing one FastAPI backend:

- **Prediction** — preview a round (nothing saved) or launch it for real
  (which is what makes it show up in History, scored automatically once
  the official result exists).
- **History** — every launched round against its official result, sorted
  by predicted position or official position side by side. This is what
  the app opens to by default, showing the latest launched round's
  predictions to anyone who visits, guest or not, since Prediction alone
  starts blank until someone with write access launches something.
- **Performance** — current production model info, the full retrain
  history timeline from MLflow, and (when one exists) the staged
  candidate's before/after comparison with its own Promote button.

**New-round predictions, without touching a terminal.** The first
version of this required running two scripts by hand before a brand-new
round could be predicted at all. The app now fetches a round's practice
data itself: click Preview or Launch on a round it hasn't seen yet, and
if there's nothing to predict from, it offers to fetch it there, with a
progress indicator and a pre-flight check that warns if it's asked to
fetch before there's likely been enough time for the session to actually
be published. Once qualifying happens, "Check for official result" pulls
the real times in, scores the round, and (see MLOps above) automatically
stages a retrain candidate for review.

## Reproducing it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Data and features (add `--round N` to any of these three for a round
past what's already downloaded):

```bash
python scripts/download_historical.py         # 2023-2025, one-time
python scripts/download_2026.py               # current season
python scripts/build_features.py              # data/processed/features.parquet
python scripts/extract_qualifying_targets.py  # official results, once Q has run
```

Train and evaluate:

```bash
python scripts/prepare_phase3_dataset.py --include-holdout
python scripts/retrain_pipeline.py   # LORO check, retrain, recalibrate intervals, logs to MLflow
python -m pytest                     # the full test suite
```

Run the app:

```bash
docker compose build
docker compose up
```

API and frontend are both served at `http://localhost:8000`.

## Layout

```
src/f1qp/    model, feature, and serving code
scripts/     one entry point per pipeline step
frontend/    React app (Vite), built into the API image
tests/       pytest suite
docs/        model card, data strategy, feature engineering notes, runbooks
deploy/      free-hosting build scripts (Render, Hugging Face)
```

No `data/` or `models/lstm/staged/` or `models/lstm/history/` in this
repo: all three are regenerable (from FastF1, or from a retrain) rather
than checked in. The currently-promoted model under `models/lstm/` is
tracked, so the app works out of the box without training anything
first.
