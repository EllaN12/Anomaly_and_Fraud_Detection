#!/usr/bin/env bash
# push.sh — authenticate with a GitHub PAT and push to origin/main
#
# Usage:
#   ./push.sh                   # reads GH_PAT from env or prompts
#   GH_PAT=ghp_xxx ./push.sh   # inline token
#
# One-time setup:
#   1. Go to GitHub → Settings → Developer settings → Personal access tokens → Fine-grained
#   2. Generate a token with Contents: Read & Write for this repo
#   3. Add to your shell profile:  export GH_PAT=ghp_yourtoken
#      OR store in macOS Keychain (see below)

set -euo pipefail

REPO="EllaN12/Anomaly_and_Fraud_Detection"

# ── Resolve token ─────────────────────────────────────────────────────────────
if [ -z "${GH_PAT:-}" ]; then
  # Try macOS Keychain first (no plaintext token in env)
  if command -v security &>/dev/null; then
    GH_PAT=$(security find-generic-password -a "$USER" -s "github-pat" -w 2>/dev/null || true)
  fi
fi

if [ -z "${GH_PAT:-}" ]; then
  read -rsp "GitHub PAT (input hidden): " GH_PAT
  echo
fi

if [ -z "$GH_PAT" ]; then
  echo "ERROR: no token provided." >&2
  exit 1
fi

# ── Push ─────────────────────────────────────────────────────────────────────
# --force-with-lease: safe after history rewrite (e.g. removing large files).
# Remove this flag once local and remote histories match again.
git push --force-with-lease "https://${GH_PAT}@github.com/${REPO}.git" main

echo "✓ Pushed to github.com/${REPO}"

# ── Optional: save token to macOS Keychain for next time ─────────────────────
if command -v security &>/dev/null; then
  read -rp "Save token to macOS Keychain for next run? [y/N] " SAVE
  if [[ "$SAVE" =~ ^[Yy]$ ]]; then
    security add-generic-password -a "$USER" -s "github-pat" -w "$GH_PAT" -U
    echo "✓ Saved to Keychain (service=github-pat)"
  fi
fi
