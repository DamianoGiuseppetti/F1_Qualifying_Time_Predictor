# Hosting this app on Render.com (free)

Prepared Sep 13 2026, same day as `deploy/huggingface/`, after Hugging
Face's own account-level bug ("You've reached your CPU Basic quota
limit across all your Spaces!") paused Damiano's Space before he could
ever use it - a widely-reported issue on brand-new HF accounts with no
reliable self-service fix (only workaround reported to consistently
work: emailing HF support and waiting). Render is a second, independent
free host for the exact same Docker image, picked because:

- No credit card required for the free tier.
- A fresh account/provider, so it can't be affected by whatever is
  wrong with the Hugging Face one.
- Docker-native, so almost all of the work already done for the HF
  Space (the Dockerfile shape, and critically `f1qp/serving/hf_sync.py`
  for durable storage - see below) carries over unchanged.

The `deploy/huggingface/` target is left completely in place - nothing
here removes it. If HF support ever clears the quota bug, both can run
side by side, or the HF one can just be resumed later.

## Why a separate Dockerfile from deploy/huggingface/'s

Render, like a HF Docker Space, builds one Dockerfile with no external
bind mounts, so `deploy/render/Dockerfile` bakes in `models/lstm`,
`data/processed`, and `mlruns` exactly like the HF one does (see that
file's own comments for the full reasoning) - it's a near-duplicate,
kept separate so each target reads start-to-finish on its own. The only
real difference is the port (10000, Render's own zero-config default,
vs. 7860 for HF).

## One-time setup

1. **Create a new PRIVATE GitHub repo** just for this deploy target -
   e.g. `f1qp-render-deploy` - separate from the public GitHub repo
   point 5 of the "final version" plan still has queued for later.
   Render needs *a* git repo to watch; it doesn't need to be the public
   project write-up repo, and keeping them separate means point 5 stays
   exactly as deferred as Damiano already said it should be.
   - github.com -> "New repository" -> Private -> no README/gitignore
     (this project's own sync script creates the whole tree itself).
2. **Point this project at that repo**:
   ```
   cp deploy/render/space_remote.txt.example deploy/render/space_remote.txt
   ```
   then edit that file to your new repo's actual git URL (its own page
   -> green "Code" button -> HTTPS - looks like
   `https://github.com/<you>/<repo-name>.git`).
   - Git will ask for a username/password the first time you push to
     it, same as any GitHub push over HTTPS - GitHub also no longer
     accepts your account password here, only a **Personal Access
     Token** (github.com -> Settings -> Developer settings -> Personal
     access tokens -> generate one with `repo` scope) or an SSH remote
     URL + key if you already have one set up. Treat the token like any
     other credential - your password manager, not this repo.
3. **Push the first snapshot**:
   ```
   bash ./deploy/render/sync_and_push.sh
   ```
   (note: `bash`, not `python3` - it's a shell script). This assembles
   `deploy/render/repo_build/` and pushes it to the GitHub repo from
   step 1. No Git LFS needed here (unlike the HF target) - GitHub
   accepts these few-MB binary files with a completely ordinary push.
4. **Create the Render Web Service**: on dashboard.render.com -> "New +"
   -> "Web Service" -> connect your GitHub account if you haven't
   already -> pick the repo from step 1 -> Render should auto-detect
   the `Dockerfile` at its root ("Environment: Docker"). Confirm:
   - **Instance Type: Free**
   - Nothing else needs changing - the Dockerfile already `EXPOSE`s
     and listens on Render's own default port (10000), so there's no
     port field to set.
   -> "Create Web Service". The first build takes a few minutes (mostly
      installing torch) - watch it under that service's "Logs" tab.
5. **Set your admin token** (same mechanism as the HF Space's SETUP.md
   step 5 - `f1qp/api/main.py`'s `require_admin` gating is host-agnostic):
   on the service's page, "Environment" tab -> "Add Environment
   Variable" -> key `ADMIN_TOKEN`, value your own choice (a long random
   string - your password manager can generate one). Paste that SAME
   value into the app's own admin panel (the "GUEST" chip in the status
   bar, under the tabs - not top-right) the first time you open the
   deployed URL yourself, so your browser remembers it and unlocks
   Launch/data-fetch from then on.
6. **Point it at the same private Hugging Face Dataset repo for durable
   storage** (this is the SAME `HF_DATASET_REPO`/`HF_TOKEN` mechanism as
   the HF Space's SETUP.md step 6 - `f1qp/serving/hf_sync.py` doesn't
   know or care which compute host is running it, only that these two
   env vars are set): same "Environment" tab, add:
   - `HF_DATASET_REPO` - the dataset's id, e.g. `DaamG/F1_Qualifying_Predictor`
   - `HF_TOKEN` - the same Write-scope access token used for the HF
     Space (or a fresh one)

   Skip this step entirely if you'd rather not bother - the app works
   exactly as it did before this existed, it just goes back to not
   surviving a redeploy.

Render redeploys automatically whenever it detects an environment
variable change, so adding step 5/6's variables triggers its own
rebuild - no extra action needed.

## Every time you want to (re)deploy

```
bash ./deploy/render/sync_and_push.sh
```

This rebuilds `deploy/render/repo_build/` from whatever is currently in
`models/lstm/`, `data/processed/`, `mlruns/`, `data/predictions/`,
`src/`, `frontend/` on your machine, and pushes it to the private
GitHub repo. Render watches that repo and rebuilds automatically on
every push - no dashboard click needed. Once it finishes, open the
service's `.onrender.com` URL and check `/health`, `/docs`, and that
the frontend itself loads at `/`. This same script IS the "promote"
action from the retrain-automation plan (point 3): running it after a
retrain is what actually puts the new model in front of anyone using
this deployment.

## Free tier limits worth knowing

- **750 free instance-hours/month** per Render account - far more than
  a personal-portfolio-traffic app will use even running continuously
  for a full month.
- **Spins down after 15 minutes with no traffic**, then takes about a
  minute to wake back up on the next request (a loading state is shown
  to the visitor) - same trade-off as any other free-tier host, and a
  complete non-issue for a portfolio piece.
- **Ephemeral filesystem** - anything written at runtime is lost on
  redeploy/restart/spin-down, exactly like the HF Space. This is
  already handled: see step 6 above and `f1qp/serving/hf_sync.py`.
- No credit card is required to sign up or to create a Free-tier web
  service.
