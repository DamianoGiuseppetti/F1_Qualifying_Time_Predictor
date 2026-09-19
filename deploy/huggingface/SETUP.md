# Hosting this app on Hugging Face Spaces

Prepared Sep 13 2026, in response to the "final version of the
application" storage/hosting requirement. This is the setup for hosting
option 1 (see the Sep 13 2026 analysis in Task_List.txt / this
conversation) - a free, no-spend host, reachable via a link from the
GitHub repo once it's public.

## Why a separate Dockerfile

Hugging Face Spaces builds one Dockerfile with no external volumes - it
can't mount `./models` or `./data/processed` from a host the way
`docker-compose.yml` does locally. `deploy/huggingface/Dockerfile` bakes
those in instead (they're tiny - under 5MB combined). Nothing about your
local `docker-compose.yml` / root `Dockerfile` workflow changes; this is
an entirely separate, additional deployment target.

## One-time setup

1. **Create a free Hugging Face account** (huggingface.co) if you don't
   have one.
2. **Create the Space**: on huggingface.co, "New Space" ->
   - Owner: your username
   - Space name: e.g. `f1-qualifying-predictor`
   - License: your choice (matches whatever you pick for the GitHub repo)
   - **Select the Docker SDK**, template "Blank"
   - Hardware: **CPU basic** (free)
   - Visibility: **Public** - it has to be, for the "guests can view via
     a link from the repo" requirement to mean anything.
3. **Get a push credential**: Hugging Face -> your profile -> Settings ->
   Access Tokens -> New token, role **Write**. Git will ask for this as
   the *password* the first time you push (username: your HF username) -
   Hugging Face no longer accepts your actual account password for git
   over HTTPS, only a token like this one. Treat it like any other
   credential - store it in your own password manager, not in this repo.
   **Also install Git LFS** if you don't already have it (macOS:
   `brew install git-lfs`) - `sync_and_push.sh` (see below) needs it to
   push the binary files (`.parquet`, `.pt`/`.pth`, `mlruns/mlflow.db`)
   this Space bakes in; Hugging Face's own pre-receive hook rejects a
   plain push containing those outright without it.
4. **Point this project at your Space**:
   ```
   cp deploy/huggingface/space_remote.txt.example deploy/huggingface/space_remote.txt
   ```
   then edit that file to your Space's actual git URL (shown on the
   Space's page, "..." menu -> "Clone repository" - looks like
   `https://huggingface.co/spaces/<you>/<space-name>`).
5. **Set your admin token** (Sep 13 2026 addition - point 4's gating is
   now built, see "Both things are now resolved" below for what it does):
   on the Space's page, Settings -> "Variables and secrets" -> New secret
   -> name it `ADMIN_TOKEN`, value your own choice (a long random string -
   your password manager can generate one). Hugging Face injects it as a
   normal environment variable at runtime, exactly like the app's other
   `F1QP_*` settings. Paste that SAME value into the app's own admin panel
   (the "GUEST" chip in the status bar, under the tabs - not top-right)
   the first time you open the Space yourself, so your browser remembers
   it (localStorage) and unlocks
   Launch/data-fetch from then on - anyone without it only ever sees
   "GUEST" and Preview.
6. **Create a private Dataset repo for durable storage** (Sep 13 2026
   addition, same day - "storage should be into hugging face without be
   in my local storage anymore"; see "Both things are now resolved"
   below): on huggingface.co, "New Dataset" -> owner your username, name
   e.g. `f1qp-space-data`, visibility **Private** (it only ever holds
   your own launched predictions and fetched round data - no reason to
   make it public). Then add TWO new secrets to the Space (same
   "Variables and secrets" page as step 5):
   - `HF_DATASET_REPO` - the dataset's id, e.g. `your-username/f1qp-space-data`
   - `HF_TOKEN` - a Write-scope access token (same kind as step 3's push
     credential - reusing that exact token is fine, or make a new one)

   You can skip this step entirely if you'd rather not bother - the app
   works exactly as it did before this existed, it just goes back to not
   surviving a redeploy (see "Both things are now resolved" below).

## Every time you want to (re)deploy

```
./deploy/huggingface/sync_and_push.sh
```

This rebuilds `deploy/huggingface/space_build/` from whatever is
currently in `models/lstm/`, `data/processed/`, `mlruns/`,
`data/predictions/`, `src/`, `frontend/` on your machine, and pushes it
to the Space. Git will prompt for the username/token from step 3 above
the first time (and again whenever the token expires). The Space
rebuilds itself automatically on receiving the push - watch its "Logs"
tab on huggingface.co; a build takes a few minutes, mostly installing
torch.

Once it finishes, open the Space's URL and check `/health`,
`/docs`, and that the frontend itself loads at `/`. This same script IS
the "promote" action from the retrain-automation plan (point 3): running
it after a retrain is what actually puts the new model in front of
anyone using this Space.

## Both things are now resolved

**1. Gating is built (Sep 13 2026).** `f1qp/api/main.py` now has a
`require_admin` dependency: Preview and every GET route (History,
model info, etc.) stay open to everyone; `POST /predict/.../launch`,
`POST /data/fetch/...`, and `POST /data/fetch-results/...` all require an
`X-Admin-Token` header matching the `ADMIN_TOKEN` secret (step 5 above).
If `ADMIN_TOKEN` isn't set at all (local docker-compose, nothing to do
there), gating is a no-op - zero change to the local workflow. The
frontend has a matching admin panel (GUEST/ADMIN chip in the status
bar, under the tabs) - pasting the right token there unlocks Launch and
the fetch buttons on that browser; anyone without it just doesn't see
those actions offered.
Covered by `tests/test_auth.py` - run the usual `python -m pytest` to
confirm, same as any other change in this project.

**2. Runtime storage now lives on Hugging Face, not just your Mac
(Sep 13 2026, same day).** Damiano: "storage should be into hugging face
without be in my local storage anymore." A real Launch, a `/data/fetch`
job, and a "check for official result" click all used to write only to
the container's writable layer - gone on the next push or an occasional
platform-side restart, which meant your Mac (via `sync_and_push.sh`) was
the only thing keeping that data alive. `f1qp/serving/hf_sync.py` closes
that gap: every launch JSON, `features.parquet`, and
`qualifying_targets.parquet` this app writes at runtime is now also
pushed straight to the private Dataset repo from step 6 above, and pulled
back down from there at startup (`f1qp/api/main.py`'s `lifespan`) -
BEFORE the Space's own baked-in copy is even looked at. Practically:
   - A Launch made directly against the Space, or a fetch/result-check
     click made there, now survives the NEXT redeploy on its own -
     `sync_and_push.sh` and your Mac are no longer the only copy.
   - Both env vars (step 6) are optional and independent of everything
     else - leave them unset and this is a complete no-op, same as
     `ADMIN_TOKEN` above (and exactly how local docker-compose, which
     already has its own real read-write volume mount, stays configured
     - see its own comment in `docker-compose.yml`).
   - `models/lstm/` and `mlruns/` are NOT part of this - those only ever
     change via a manual retrain + `sync_and_push.sh` (Aug 24 2026
     decision) and still work exactly as before; this is only for what
     the RUNNING container itself produces.
   - **A real bug was fixed alongside this**, not just a storage
     mechanism: `scripts/build_features.py` used to silently DROP every
     round it couldn't find raw laps for on disk - harmless on your full
     local checkout, but the Space never bakes in the (huge,
     training-only) raw-lap cache, so its very first on-demand fetch
     would have wiped every other round's features. It now carries
     forward whatever was already in `features.parquet` for a round it
     can't rebuild, instead of dropping it - see that script's own
     module docstring ("Carry-forward for missing raw laps"). A sibling
     fix in `scripts/extract_qualifying_targets.py` (see its own
     docstring, "Incremental when --round is passed") stops the "check
     for official result" button from re-fetching literally every
     historical season from FastF1 over the network on every click, which
     could otherwise overrun the 15-minute per-step job timeout on a Space
     with a cold cache.
   - Covered by `tests/test_hf_sync.py`, `tests/test_history_hf_sync_wiring.py`,
     `tests/test_data_fetch_hf_sync_wiring.py`, `tests/test_build_features_merge.py`,
     and `tests/test_extract_qualifying_targets_incremental.py` - again,
     run `python -m pytest` yourself to confirm; `huggingface_hub` (added
     to `pyproject.toml`/`requirements.txt`) needs to actually be
     installed in your environment (`pip install -e .` or
     `pip install -r requirements.txt`) for the full suite to collect,
     though the hf_sync tests themselves fake it out and don't need it.

No further storage work is planned - both items this file used to flag
as open are done.
