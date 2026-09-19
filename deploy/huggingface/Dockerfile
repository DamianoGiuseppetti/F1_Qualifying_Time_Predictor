# Hugging Face Space deployment image for the F1 Qualifying Predictor.
#
# Why this is a SEPARATE Dockerfile from the repo root one (Phase 4):
# HF Spaces builds a single Dockerfile with no external bind mounts, so
# everything docker-compose.yml normally mounts read-only from the host
# (models/lstm, data/processed, mlruns) has to be baked into the image
# instead. All three combined are under 5MB as of Sep 2026 (see
# docs/model_card.md) - copying them in costs nothing. The root
# Dockerfile stays exactly as it is for local docker-compose use; this
# one only exists for the Space.
#
# This file expects to sit at the ROOT of whatever gets pushed to the
# Space's own git repo - HF Docker Spaces don't support a Dockerfile in
# a subdirectory. deploy/huggingface/sync_and_push.sh assembles that
# flat build context (this Dockerfile + README.md + a fresh copy of
# pyproject.toml, src/, frontend/, models/lstm/, data/processed/,
# mlruns/, data/predictions/) and pushes it - see SETUP.md in this same
# folder for the one-time Space creation steps and the repeatable
# update flow.
#
# CAVEAT, UPDATED Sep 13 2026 - read before relying on this for real
# Launch history:
# Hugging Face deprecated free persistent storage for Spaces, so by
# itself, anything THIS container writes at runtime (data/predictions/
# launches/*.json from a real Launch, or a /data/fetch job's output)
# would live only in the container's writable layer and reset whenever
# the Space's container is recreated - a new push (the normal update path
# below) or, less predictably, a platform-side restart.
#
# That gap is now closed at the APPLICATION level instead of the
# infrastructure level: f1qp.serving.hf_sync mirrors every runtime write
# out to (and restores it from) a separate, free Hugging Face DATASET
# repo, via the HF_DATASET_REPO / HF_TOKEN secrets - see SETUP.md's setup
# steps. Nothing to change in this Dockerfile for that: both are plain
# Space secrets, injected as env vars at runtime exactly like ADMIN_TOKEN
# already is, and f1qp.serving.hf_sync is a no-op whenever they're unset.

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
# Sep 16 2026 fix (same bug found testing Round 14 on the Render target -
# see deploy/render/Dockerfile's matching comment): f1qp.serving.data_fetch
# runs scripts/download_2026.py, build_features.py and
# extract_qualifying_targets.py as subprocesses - this image never copied
# scripts/ in at all, so "Fetch data now" could never have worked here
# either. See F1QP_REPO_ROOT below for the other half of this fix - also
# added to deploy/huggingface/sync_and_push.sh's build-context assembly.
COPY scripts ./scripts

# CPU-only torch explicitly, installed BEFORE `pip install .` - a plain
# `pip install .` would resolve torch from PyPI's default (CUDA) build,
# multiple GB for a GPU this free CPU Space doesn't have. The CPU wheel
# is a fraction of the size; this LSTM is tiny (see docs/model_card.md),
# so inference speed is unaffected. `pip install .` afterward sees
# torch>=2.2 already satisfied and installs everything else
# (fastapi, xgboost, mlflow, ...) around it - if a future dependency
# bump ever causes pip to silently pull a second (CUDA) torch anyway,
# `pip show torch` in the Space's build logs is the first thing to check.
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

# 7860 is Hugging Face's own Docker Space default (must match this
# repo's README.md `app_port:` frontmatter) - kept as 7860 rather than
# reusing the root Dockerfile's 8000 purely so this file matches what a
# Spaces reader expects at a glance.
EXPOSE 7860
CMD ["uvicorn", "f1qp.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
