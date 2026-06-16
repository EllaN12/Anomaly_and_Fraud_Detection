#!/usr/bin/env bash
# push.sh — authenticate with a GitHub PAT and push to origin/main
#
# Usage (pick one):
#   ./push.sh                         # Keychain, .github_pat file, or prompt
#   export GH_PAT='your_token'; ./push.sh
#
# Easiest if paste fails in the terminal:
#   1. Create .github_pat in THIS folder (one line = your token). File is gitignored.
#   2. Run: ./push.sh
#
# One-time Keychain setup (macOS):
#   security add-generic-password -a "$USER" -s "github-pat" -w 'YOUR_TOKEN' -U

set -euo pipefail

REPO="EllaN12/Anomaly_and_Fraud_Detection"
ROOT="$(cd "$(dirname "$0")" && pwd)"
PAT_FILE="$ROOT/.github_pat"
PUSH_URL="https://${GH_PAT:-}@github.com/${REPO}.git"

# ── Resolve token ─────────────────────────────────────────────────────────────
if [ -z "${GH_PAT:-}" ] && [ -f "$PAT_FILE" ]; then
  GH_PAT="$(tr -d '[:space:]' < "$PAT_FILE")"
  echo "Using token from .github_pat"
fi

if [ -z "${GH_PAT:-}" ] && command -v security &>/dev/null; then
  GH_PAT=$(security find-generic-password -a "$USER" -s "github-pat" -w 2>/dev/null || true)
  [ -n "$GH_PAT" ] && echo "Using token from macOS Keychain"
fi

if [ -z "${GH_PAT:-}" ]; then
  echo "Paste often fails in Cursor's hidden prompt. Options:"
  echo "  • Save token to .github_pat (one line, no quotes) and rerun ./push.sh"
  echo "  • Or run:  export GH_PAT=   then paste token after = and press Enter"
  echo
  read -rp "GitHub PAT (visible as you paste): " GH_PAT
fi

if [ -z "$GH_PAT" ]; then
  echo "ERROR: no token provided." >&2
  exit 1
fi

PUSH_URL="https://${GH_PAT}@github.com/${REPO}.git"

# ── Push ─────────────────────────────────────────────────────────────────────
git fetch origin
REMOTE_SHA="$(git rev-parse origin/main)"
echo "Remote main is at ${REMOTE_SHA:0:7}; pushing rewritten history…"

# Explicit lease required when pushing to a URL (not "origin" remote name).
git push --force-with-lease="refs/heads/main:${REMOTE_SHA}" "$PUSH_URL" main

echo "✓ Pushed to github.com/${REPO}"

if command -v security &>/dev/null && [ -f "$PAT_FILE" ]; then
  read -rp "Save token to macOS Keychain and delete .github_pat? [y/N] " SAVE
  if [[ "$SAVE" =~ ^[Yy]$ ]]; then
    security add-generic-password -a "$USER" -s "github-pat" -w "$GH_PAT" -U
    rm -f "$PAT_FILE"
    echo "✓ Saved to Keychain; removed .github_pat"
  fi
fi
