# Render.com deployment image for the F1 Qualifying Predictor.
#
# Sep 13 2026 - added as a SECOND free hosting target alongside
# deploy/huggingface/, after Hugging Face's own account-level "CPU
# Basic quota limit" bug paused the Space before Damiano could ever use
# it (a widely-reported issue on brand-new HF accounts with zero
# self-service fix - see deploy/huggingface/SETUP.md's own note on it).
# This file is a near-duplicate of deploy/huggingface/Dockerfile - same
# reasoning throughout, just re-hosted - kept as its own copy rather
# than a shared/parameterized file so each deploy target's Dockerfile
# can be read start-to-finish on its own, same spirit as having a
# separate Dockerfile from the repo root one in the first place.
#
# Why this is a SEPARATE Dockerfile from the repo root one (Phase 4)
# and from deploy/huggingface/Dockerfile: Render, like a HF Docker
# Space, builds one Dockerfile with no external bind mounts, so
# everything docker-compose.yml normally mounts read-only from the host
# (models/lstm, data/processed, mlruns) has to be baked into the image
# instead. All three combined are under 5MB as of Sep 2026 (see
# docs/model_card.md) - copying them in costs nothing.
#
# This file expects to sit at the ROOT of whatever gets pushed to the
# private GitHub repo Render is connected to - deploy/render/sync_and_push.sh
# assembles that flat build context (this Dockerfile + a fresh copy of
# pyproject.toml, src/, frontend/, models/lstm/, data/processed/,
# mlruns/, data/predictions/) and pushes it - see SETUP.md in this same
# folder for the one-time Render setup steps and the repeatable update
# flow.
#
# CAVEAT: same ephemeral-filesystem situation as the HF Space -
# anything THIS container writes at runtime (data/predictions/
# launches/*.json from a real Launch, or a /data/fetch job's output)
# would live only in the container's writable layer and reset on every
# redeploy or platform-side restart. That gap is closed at the
# APPLICATION level, not here: f1qp.serving.hf_sync mirrors every
# runtime write out to (and restores it from) a Hugging Face DATASET
# repo (a plain storage repo, unrelated to whichever compute host is
# actually running the app) via the HF_DATASET_REPO / HF_TOKEN env vars
# - see SETUP.md. This is exactly why that mechanism was built against
# a Dataset repo rather than anything HF-Space-specific: it works
# unchanged no matter which free host actually runs the container.
# Nothing to change in this Dockerfile for that - both are plain Render
# environment variables, injected at runtime exactly like ADMIN_TOKEN,
# and f1qp.serving.hf_sync is a no-op whenever they're unset.

# --- Stage 1: build the React frontend - same approach as the root
# Dockerfile's own frontend-build stage, see its comments for the
# reasoning (caching order, no Node needed on the host). ---
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: the API image ---
FROM python:3.11-slim
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
# Sep 16 2026 fix (Damiano testing Round 14: the "Fetch data now" job
# failed with "can't open file '/usr/local/lib/python3.11/scripts/
# download_2026.py'"): f1qp.serving.data_fetch runs scripts/download_2026.py,
# build_features.py and extract_qualifying_targets.py as subprocesses -
# this image never copied scripts/ in at all, so that feature could never
# have worked here. See the F1QP_REPO_ROOT line below for the other half
# of this fix (why the path was wrong even ignoring the missing COPY) -
# also added to deploy/render/sync_and_push.sh's build-context assembly.
COPY scripts ./scripts

# CPU-only torch explicitly, installed BEFORE `pip install .` - a plain
# `pip install .` would resolve torch from PyPI's default (CUDA) build,
# multiple GB for a GPU this free CPU host doesn't have. The CPU wheel
# is a fraction of the size; this LSTM is tiny (see docs/model_card.md),
# so inference speed is unaffected. `pip install .` afterward sees
# torch>=2.2 already satisfied and installs everything else
# (fastapi, xgboost, mlflow, ...) around it - if a future dependency
# bump ever causes pip to silently pull a second (CUDA) torch anyway,
# `pip show torch` in Render's build logs is the first thing to check.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

# Built frontend only (frontend/dist/ - static HTML/JS/CSS), not the
# React source - same reasoning as the root Dockerfile's own comment on
# this same line.
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Model/data artifacts baked in (see the module-level caveat above for
# why there's no volume mount here instead). Kept as separate COPY
# layers, not one combined layer, so a data-only update after a retrain
# invalidates only these last few layers, not the much slower
# dependency-install layers above.
COPY models/lstm ./models/lstm
COPY data/processed ./data/processed
COPY mlruns ./mlruns
COPY data/predictions ./data/predictions

ENV F1QP_MODELS_DIR=/app/models/lstm \
    F1QP_FEATURES_PATH=/app/data/processed/features.parquet \
    F1QP_TARGETS_PATH=/app/data/processed/qualifying_targets.parquet \
    F1QP_PREDICTIONS_DIR=/app/data/predictions/launches \
    F1QP_MLRUNS_DIR=/app/mlruns \
    F1QP_REPO_ROOT=/app

# 10000 is Render's own zero-config default port for a web service (its
# docs: "the default port is 10000" unless a PORT env var overrides it)
# - using it here means nothing extra needs setting in Render's
# dashboard beyond connecting the repo; Render just routes to it.
EXPOSE 10000
CMD ["uvicorn", "f1qp.api.main:app", "--host", "0.0.0.0", "--port", "10000"]
