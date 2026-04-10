#!/bin/bash
# Start Claude Code in the sandbox with all permissions auto-accepted.
# Captures OpenTelemetry output (metrics + events) to a per-invocation JSONL
# file under /var/log/claude-telemetry (bind-mounted to sandbox/telemetry/
# on the host).
#
# Usage: bash sandbox/claude.sh "your prompt here"
#   or:  bash sandbox/claude.sh   (interactive mode)

set -euo pipefail

TELEMETRY_DIR=/var/log/claude-telemetry
LOG_FILE="$TELEMETRY_DIR/otel-$(date -u +'%Y%m%dT%H%M%SZ').jsonl"

mkdir -p "$TELEMETRY_DIR"

# Inline secret scrub: redact known secret patterns before they hit disk.
# Applied to every telemetry line. Belt-and-braces vs. an agent ever
# inlining an env var into a bash command / prompt. Patterns cover common
# token formats; extend as new ones appear.
scrub() {
    sed -E \
        -e 's/ghp_[A-Za-z0-9]{20,}/[REDACTED:ghp]/g' \
        -e 's/gho_[A-Za-z0-9]{20,}/[REDACTED:gho]/g' \
        -e 's/ghu_[A-Za-z0-9]{20,}/[REDACTED:ghu]/g' \
        -e 's/ghs_[A-Za-z0-9]{20,}/[REDACTED:ghs]/g' \
        -e 's/ghr_[A-Za-z0-9]{20,}/[REDACTED:ghr]/g' \
        -e 's/sk-ant-[A-Za-z0-9_-]{20,}/[REDACTED:anthropic]/g' \
        -e 's/sk-[A-Za-z0-9]{32,}/[REDACTED:openai]/g' \
        -e 's/-----BEGIN [A-Z ]*PRIVATE KEY-----/[REDACTED:private-key]/g' \
        -e 's/xox[baprs]-[A-Za-z0-9-]{10,}/[REDACTED:slack]/g' \
        -e 's/AKIA[0-9A-Z]{16}/[REDACTED:aws-access-key]/g'
}

# Claude Code writes telemetry to stderr via the console exporter. Fork
# stderr through the scrub filter into the log file while preserving
# terminal visibility.
if [ $# -gt 0 ]; then
    claude -p --dangerously-skip-permissions "$@" \
        2> >(scrub | tee -a "$LOG_FILE" >&2)
else
    claude --dangerously-skip-permissions \
        2> >(scrub | tee -a "$LOG_FILE" >&2)
fi
