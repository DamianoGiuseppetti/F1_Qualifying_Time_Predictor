# Phase 4: containerized FastAPI inference service.
#
# Build context is the `repo/` directory (this file's own directory) - see
# docker-compose.yml. Installs from pyproject.toml's `dependencies` list
# (`pip install .`), NOT requirements.txt: requirements.txt is the full
# dev environment (matplotlib, ipykernel, pytest, ...) meant for VSCode,
# most of which the running API never touches. Keep pyproject.toml in sync
# if f1qp.api / f1qp.serving / f1qp.modeling start importing something new
# at module load time (see the NOTE below for a real example of this
# already being non-obvious).
#
# Model artifacts (models/lstm/*) and features (data/processed/*) are
# DELIBERATELY NOT copied into the image - see docker-compose.yml's volume
# mounts instead. The project's retrain trigger stays a manual script
# (scripts/retrain_pipeline.py, run on the host - Aug 24 2026
# AskUserQuestion decision), so a freshly retrained model should reach the
# running API via a container restart, not an image rebuild.

# --- Stage 1: build the React frontend (Aug 30 2026 rebuild - Damiano's
# own choice via AskUserQuestion, replacing the Aug 26 2026 plain HTML/
# CSS/vanilla-JS frontend). A separate Node stage so nobody needs Node
# installed on the host - `docker compose build` is still the only command
# Damiano has to run, same as every phase before this one. Only
# frontend/package.json copied first so this layer (and `npm install`)
# caches across rebuilds that don't touch frontend/'s dependencies; no
# package-lock.json is committed yet, so this resolves each dependency's
# latest version satisfying package.json's ^ranges - fine for now, but if
# a rebuild ever pulls in a breaking transitive update, committing
# frontend/package-lock.json (from a local `npm install`) and switching
# this to `npm ci` pins it.
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: the actual API image ---
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
# Sep 16 2026 fix (Damiano testing Round 14: data-fetch subprocess call
# failed with "can't open file '.../scripts/download_2026.py'"):
# f1qp.serving.data_fetch runs scripts/download_2026.py, build_features.py
# and extract_qualifying_targets.py as subprocesses - they were never
# copied into ANY of this project's images (this one, deploy/render/,
# deploy/huggingface/), so the on-demand "Fetch data now" feature could
# never have worked in a container, only via local `uvicorn --reload` run
# straight from a source checkout. See docker-compose.yml's matching
# F1QP_REPO_ROOT addition for the other half of this fix.
COPY scripts ./scripts

# Aug 30 2026: only the BUILT frontend (frontend/dist/ - static HTML/JS/
# CSS, no Node needed at runtime) is copied in, from stage 1 above - not
# the React source. See src/f1qp/api/main.py's static-mount comment for
# why this still ends up served by this same image/service.
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# NOTE (module-import gotcha, worth knowing if this ever fails to import):
# f1qp.serving.predict imports f1qp.modeling.lstm_model, which imports
# `from f1qp.modeling.baseline import mape, r_squared` at module load time
# - and baseline.py imports xgboost. So xgboost is a REAL transitive
# runtime dependency of the API even though f1qp.api/f1qp.serving never
# call any XGBoost function directly - it's why xgboost is listed in
# pyproject.toml's dependencies, not left as a dev-only extra.
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "f1qp.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
