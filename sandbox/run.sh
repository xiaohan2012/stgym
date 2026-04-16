#!/bin/bash
# Launch the agent sandbox container.
# Automatically extracts Claude Code OAuth token from macOS Keychain
# and GH_TOKEN from gh CLI auth.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Ensure the host-side telemetry dir exists before docker compose mounts it.
# Otherwise Docker creates it as root on Linux (ownership headaches).
mkdir -p "$SCRIPT_DIR/telemetry"

# Extract Claude Code OAuth refresh token from macOS Keychain
export CLAUDE_CODE_OAUTH_REFRESH_TOKEN=$(
  security find-generic-password -s "Claude Code-credentials" -w \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['claudeAiOauth']['refreshToken'])"
)

# Extract GH_TOKEN from gh CLI auth
export GH_TOKEN=$(gh auth token)

echo "Starting agent sandbox..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" build
docker compose -f "$SCRIPT_DIR/docker-compose.yml" run --rm --service-ports agent
