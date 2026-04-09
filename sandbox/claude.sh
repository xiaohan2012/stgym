#!/bin/bash
# Start Claude Code in the sandbox with all permissions auto-accepted.
# Usage: bash sandbox/claude.sh "your prompt here"
#   or:  bash sandbox/claude.sh   (interactive mode)

set -euo pipefail

if [ $# -gt 0 ]; then
    claude -p --dangerously-skip-permissions "$@"
else
    claude --dangerously-skip-permissions
fi
