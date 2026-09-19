#!/usr/bin/env bash
# Assembles a fresh, flat build context for the private GitHub repo
# Render is connected to, and pushes it - see SETUP.md in this same
# folder for the one-time setup steps this depends on. Run this from
# anywhere; it finds the repo root itself.
#
# Sep 13 2026 - this is deploy/render's counterpart to
# deploy/huggingface/sync_and_push.sh (same reasoning throughout,
# re-hosted after HF's own account-level quota bug blocked that Space
# before Damiano could ever use it). One real difference from the HF
# version: no Git LFS setup here. Hugging Face's own push hook rejects
# ANY plain-git binary file outright (its newer Xet-storage policy);
# GitHub has no such blanket rule - it only hard-rejects a single file
# over 100MB, and everything this script copies in is a few MB combined
# (see docs/model_card.md), so a completely ordinary `git push` just
# works.
#
# What it does, every time you run it:
#   1. Rebuilds deploy/render/repo_build/ from scratch (never edited by
#      hand - always a fresh copy of the current repo state).
#   2. Copies in only what the Dockerfile needs: itself, pyproject.toml,
#      src/, scripts/ (Sep 16 2026 addition - see the matching fix below),
#      frontend/ (source, not node_modules/dist - Render's own
#      Docker build runs `npm install && npm run build` itself),
#      models/lstm/, data/processed/, mlruns/, data/predictions/.
#   3. Commits that snapshot and force-pushes it to the "render" remote
#      you configure once in space_remote.txt (see SETUP.md step 2) -
#      a private GitHub repo. Render auto-detects the push and rebuilds
#      the Space on its own; no separate "deploy" click needed once the
#      GitHub connection is set up.
#
# This IS the "promote to production" step for this deployment target
# (point 3 of the Sep 13 2026 "final version" plan): whatever is
# currently in models/lstm/ + data/processed/ + mlruns/ on your Mac when
# you run this is what Render serves after its next build finishes -
# there is no separate volume to restart into the way local
# docker-compose has.
#
# Never run automatically / on a schedule - a human (you) runs this
# on purpose, after checking the retrain comparison, same spirit as
# every other manual step in this project.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/repo_build"
REMOTE_FILE="$SCRIPT_DIR/space_remote.txt"

if [[ ! -f "$REMOTE_FILE" ]] || [[ -z "$(tr -d '[:space:]' < "$REMOTE_FILE")" ]]; then
  echo "Missing $REMOTE_FILE - put your private GitHub repo's git URL in it first." >&2
  echo "(e.g. https://github.com/<you>/<repo-name>.git - see SETUP.md step 1-2)" >&2
  exit 1
fi
RENDER_REMOTE="$(tr -d '[:space:]' < "$REMOTE_FILE")"

echo "Repo root:      $REPO_ROOT"
echo "GitHub remote:  $RENDER_REMOTE"
echo "Build dir:      $BUILD_DIR"
echo

echo "-- Rebuilding $BUILD_DIR from scratch --"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cp "$SCRIPT_DIR/Dockerfile" "$BUILD_DIR/Dockerfile"
cp "$REPO_ROOT/pyproject.toml" "$BUILD_DIR/pyproject.toml"
cp -R "$REPO_ROOT/src" "$BUILD_DIR/src"
# Sep 16 2026 fix: the Dockerfile now COPYs scripts/ in (Damiano's Round
# 14 test found "Fetch data now" failing with a missing-file error -
# f1qp.serving.data_fetch runs these as subprocesses and they were never
# part of this build context at all) - has to land here too, or the
# Dockerfile's own COPY scripts ./scripts finds nothing to copy.
cp -R "$REPO_ROOT/scripts" "$BUILD_DIR/scripts"
cp -R "$REPO_ROOT/models/lstm" "$BUILD_DIR/models_lstm_tmp" && mkdir -p "$BUILD_DIR/models" && mv "$BUILD_DIR/models_lstm_tmp" "$BUILD_DIR/models/lstm"
mkdir -p "$BUILD_DIR/data"
cp -R "$REPO_ROOT/data/processed" "$BUILD_DIR/data/processed"
cp -R "$REPO_ROOT/data/predictions" "$BUILD_DIR/data/predictions"
cp -R "$REPO_ROOT/mlruns" "$BUILD_DIR/mlruns"

# frontend/: source only - node_modules and dist (if present locally
# from `npm run dev`/`npm run build`) must NOT be copied in; Render's
# own Docker build runs `npm install && npm run build` itself.
mkdir -p "$BUILD_DIR/frontend"
cp "$REPO_ROOT/frontend/package.json" "$BUILD_DIR/frontend/package.json"
cp "$REPO_ROOT/frontend/index.html" "$BUILD_DIR/frontend/index.html"
cp "$REPO_ROOT/frontend/vite.config.js" "$BUILD_DIR/frontend/vite.config.js"
cp -R "$REPO_ROOT/frontend/src" "$BUILD_DIR/frontend/src"

echo "-- Verifying no node_modules/dist slipped in --"
if find "$BUILD_DIR/frontend" -maxdepth 1 -name "node_modules" -o -maxdepth 1 -name "dist" | grep -q .; then
  echo "Found node_modules or dist under $BUILD_DIR/frontend - aborting." >&2
  exit 1
fi

echo "-- Committing and pushing --"
cd "$BUILD_DIR"
git init -q
git config user.email "damiano9801@gmail.com"
git config user.name "Damiano Giuseppetti"
git checkout -q -B main
git remote remove render 2>/dev/null || true
git remote add render "$RENDER_REMOTE"
git add -A
git commit -q -m "Sync from repo: $(date -u +%Y-%m-%dT%H:%M:%SZ)" --allow-empty
git push --force render main

echo
echo "Pushed. Render auto-detects the push and rebuilds - check the"
echo "service's own 'Events'/'Logs' tab on dashboard.render.com; a fresh"
echo "build typically takes a few minutes (installing torch is the slow"
echo "step). It stays on the same URL across every redeploy."
