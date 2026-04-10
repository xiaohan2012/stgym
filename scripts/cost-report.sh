#!/bin/bash
# Weekly cost & observability summary over sandbox telemetry logs.
#
# Reads sandbox/telemetry/otel-*.jsonl (produced by sandbox/claude.sh)
# and prints per-session totals: turn count, cost, tokens, first user
# prompt. Used for the weekly cost review — see CLAUDE.md § Cost &
# Observability.
#
# Usage: bash scripts/cost-report.sh [days]
#   days defaults to 7

set -euo pipefail

DAYS="${1:-7}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
TELEMETRY_DIR="$REPO_ROOT/sandbox/telemetry"

if [ ! -d "$TELEMETRY_DIR" ]; then
    echo "No telemetry directory at $TELEMETRY_DIR — nothing to report." >&2
    exit 0
fi

# Find JSONL files modified in the last N days.
FILES=$(find "$TELEMETRY_DIR" -name 'otel-*.jsonl' -mtime "-$DAYS" | sort)

if [ -z "$FILES" ]; then
    echo "No telemetry files in the last $DAYS day(s)." >&2
    exit 0
fi

echo "=== Cost & Observability Report — last $DAYS day(s) ==="
echo

# Extract events, aggregate per session.
# Each line is an OTel log record; we care about user_prompt and api_request
# events. Skip lines that aren't JSON (console exporter mixes in framing).
cat $FILES \
  | jq -c 'select(type == "object" and .attributes?)' 2>/dev/null \
  | jq -s -r '
    # Group events by session.id
    group_by(.attributes["session.id"] // "unknown")
    | map({
        session:       (.[0].attributes["session.id"] // "unknown"),
        turns:         ([.[] | select(.attributes["event.name"] == "api_request")] | length),
        cost_usd:      ([.[] | select(.attributes["event.name"] == "api_request") | (.attributes.cost_usd      | tonumber? // 0)] | add // 0),
        input_tokens:  ([.[] | select(.attributes["event.name"] == "api_request") | (.attributes.input_tokens  | tonumber? // 0)] | add // 0),
        output_tokens: ([.[] | select(.attributes["event.name"] == "api_request") | (.attributes.output_tokens | tonumber? // 0)] | add // 0),
        errors:        ([.[] | select(.attributes["event.name"] == "api_error")] | length),
        first_prompt:  ([.[] | select(.attributes["event.name"] == "user_prompt") | .attributes.prompt // empty] | first // "(no prompt logged)")
      })
    | sort_by(-.cost_usd)
    | .[]
    | "session \(.session[0:12])  turns=\(.turns)  cost=$\(.cost_usd)  tok=\(.input_tokens)/\(.output_tokens)  err=\(.errors)\n  first: \(.first_prompt[0:80])"
  '

echo
echo "=== Totals ==="
cat $FILES \
  | jq -c 'select(type == "object" and .attributes?)' 2>/dev/null \
  | jq -s -r '
    {
      sessions:           ([.[] | (.attributes["session.id"] // "unknown")] | unique | length),
      api_requests:       ([.[] | select(.attributes["event.name"] == "api_request")] | length),
      api_errors:         ([.[] | select(.attributes["event.name"] == "api_error")] | length),
      total_cost_usd:     ([.[] | select(.attributes["event.name"] == "api_request") | (.attributes.cost_usd      | tonumber? // 0)] | add // 0),
      total_input_tokens: ([.[] | select(.attributes["event.name"] == "api_request") | (.attributes.input_tokens  | tonumber? // 0)] | add // 0),
      total_output_tokens:([.[] | select(.attributes["event.name"] == "api_request") | (.attributes.output_tokens | tonumber? // 0)] | add // 0)
    }
    | "sessions=\(.sessions)  requests=\(.api_requests)  errors=\(.api_errors)  cost=$\(.total_cost_usd)  tokens=\(.total_input_tokens) in / \(.total_output_tokens) out"
  '
