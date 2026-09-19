#!/usr/bin/env bash
# Assembles a fresh, flat build context for the Hugging Face Space and
# pushes it - see SETUP.md in this same folder for the one-time Space
# creation steps this depends on. Run this from anywhere; it finds the
# repo root itself.
#
# What it does, every time you run it:
#   1. Rebuilds deploy/huggingface/space_build/ from scratch (never
#      edited by hand - always a fresh copy of the current repo state).
#   2. Copies in only what the Space's Dockerfile needs: itself,
#      README.md, pyproject.toml, src/, scripts/ (Sep 16 2026 addition -
#      see the matching fix below), frontend/ (source, not
#      node_modules/dist - the Space's own Dockerfile builds those),
#      models/lstm/, data/processed/, mlruns/, data/predictions/.
#   3. Commits that snapshot and force-pushes it to the "space" remote
#      you configure once in space_remote.txt (see SETUP.md step 2).
#
# This IS the "promote to production" step for this deployment target
# (point 3 of the Sep 13 2026 "final version" plan): whatever is
# currently in models/lstm/ + data/processed/ + mlruns/ on your Mac when
# you run this is what the Space serves after its next build finishes -
# there is no separate volume to restart into the way local
# docker-compose has.
#
# Never run automatically / on a schedule - a human (you) runs this
# on purpose, after checking the retrain comparison, same spirit as
# every other manual step in this project.
#
# Git LFS (Sep 13 2026 addition): Hugging Face now rejects a plain git
# push that contains binary files outright ("Your push was rejected
# because it contains binary files... use Xet storage") - every
# .parquet/.pt/.pth file this script copies in, plus mlruns/mlflow.db,
# has to be tracked through Git LFS before the commit that adds them, or
# the push is bounced by the Space's own pre-receive hook. See
# https://huggingface.co/docs/hub/xet/using-xet-storage#git - LFS (not
# the newer Xet CLI) was picked here since it's the one already
# preinstalled with a normal `git` on most Macs / brew installs, and HF
# still fully supports it via its LFS bridge.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/space_build"
REMOTE_FILE="$SCRIPT_DIR/space_remote.txt"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs isn't installed - install it first (macOS: brew install git-lfs)" >&2
  echo "then re-run this script. Hugging Face rejects a plain push of the" >&2
  echo "binary files this script copies in (parquet/pt/pth/mlflow.db) without it." >&2
  exit 1
fi

if [[ ! -f "$REMOTE_FILE" ]] || [[ -z "$(tr -d '[:space:]' < "$REMOTE_FILE")" ]]; then
  echo "Missing $REMOTE_FILE - put your Space's git URL in it first." >&2
  echo "(Space page -> the ... menu -> 'Clone repository', or" >&2
  echo " https://huggingface.co/spaces/<your-username>/<space-name>)" >&2
  exit 1
fi
SPACE_REMOTE="$(tr -d '[:space:]' < "$REMOTE_FILE")"

echo "Repo root:     $REPO_ROOT"
echo "Space remote:  $SPACE_REMOTE"
echo "Build dir:     $BUILD_DIR"
echo

echo "-- Rebuilding $BUILD_DIR from scratch --"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cp "$SCRIPT_DIR/Dockerfile" "$BUILD_DIR/Dockerfile"
cp "$SCRIPT_DIR/README.md" "$BUILD_DIR/README.md"
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
# from `npm run dev`/`npm run build`) must NOT be copied in; the Space's
# own Dockerfile runs `npm install && npm run build` itself.
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
git remote remove space 2>/dev/null || true
git remote add space "$SPACE_REMOTE"

# Track every binary type this build dir can contain BEFORE the first
# `git add` - LFS only intercepts files matching .gitattributes at the
# moment they're staged, so this has to exist first, not be added
# alongside the files it's meant to cover.
git lfs install --local
{
  echo "*.parquet filter=lfs diff=lfs merge=lfs -text"
  echo "*.pt filter=lfs diff=lfs merge=lfs -text"
  echo "*.pth filter=lfs diff=lfs merge=lfs -text"
  echo "mlruns/mlflow.db filter=lfs diff=lfs merge=lfs -text"
} > .gitattributes

git add -A
git commit -q -m "Sync from repo: $(date -u +%Y-%m-%dT%H:%M:%SZ)" --allow-empty
git push --force space main

echo
echo "Pushed. Check the Space's own 'Logs' tab on huggingface.co for the build -"
echo "a fresh build typically takes a few minutes (installing torch is the slow step)."
