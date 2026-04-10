#!/bin/bash
# Start Claude Code in the sandbox with all permissions auto-accepted.
#
# Telemetry capture is handled out-of-band by the otel-collector sidecar
# (see sandbox/docker-compose.yml and sandbox/otel-collector-config.yaml).
# No stdio wrapping needed here.
#
# Usage: bash sandbox/claude.sh "your prompt here"
#   or:  bash sandbox/claude.sh   (interactive mode)

set -euo pipefail

if [ $# -gt 0 ]; then
    claude -p --dangerously-skip-permissions "$@"
else
    claude --dangerously-skip-permissions
fi
