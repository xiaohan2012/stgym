#!/bin/bash
# Weekly cost & observability summary over sandbox telemetry logs.
#
# Reads sandbox/telemetry/otel.jsonl* (produced by the OTel collector
# sidecar in sandbox/docker-compose.yml) and prints per-session totals:
# turn count, cost, tokens, first user prompt.
#
# Pipeline: JSONL → secret scrub → flatten OTLP envelopes → jq aggregation.
# See sandbox/README.md § Cost & Observability.
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

# Gather JSONL files modified in the last N days. The collector writes
# otel.jsonl and rotates to otel.jsonl.1, otel.jsonl.2, etc.
FILES=$(find "$TELEMETRY_DIR" -name 'otel.jsonl*' -mtime "-$DAYS" 2>/dev/null | sort)

if [ -z "$FILES" ]; then
    echo "No telemetry files in the last $DAYS day(s)." >&2
    exit 0
fi

# Redact common secret patterns before anything else sees the data.
# Belt-and-braces vs. an agent having inlined a token into a command.
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

# jq program: flatten OTLP protocol JSON into per-event records, then
# group by session.id and summarize.
#
# OTLP JSON structure (one object per exporter batch):
#   { resourceLogs: [ { resource:{attributes:[...]}, scopeLogs:[ { logRecords:[ { attributes:[{key,value:{stringValue|intValue|...}}] } ] } ] } ] }
#
# attr($k): look up attribute value by key, unwrapping the typed value.
JQ_PROGRAM=$(cat <<'JQ'
def attr($k):
  (.attributes // [])
  | map(select(.key == $k))
  | first
  | .value
  | (.stringValue // .intValue // .doubleValue // .boolValue // null);

# Unwrap all log records from a single OTLP batch object.
def log_records:
  (.resourceLogs // [])[]
  | (.scopeLogs // [])[]
  | (.logRecords // [])[];

# Flatten all records across all batches, extract fields we care about.
[ .[]
  | log_records
  | {
      event_name:    attr("event.name"),
      session_id:    (attr("session.id") // "unknown"),
      cost_usd:      attr("cost_usd"),
      input_tokens:  attr("input_tokens"),
      output_tokens: attr("output_tokens"),
      duration_ms:   attr("duration_ms"),
      prompt:        attr("prompt"),
      error:         attr("error")
    }
]
| group_by(.session_id)
| map({
    session:       (.[0].session_id),
    turns:         ([.[] | select(.event_name == "api_request")] | length),
    cost_usd:      ([.[] | select(.event_name == "api_request") | ((.cost_usd // "0") | tostring | tonumber? // 0)] | add // 0),
    input_tokens:  ([.[] | select(.event_name == "api_request") | ((.input_tokens // 0) | tostring | tonumber? // 0)] | add // 0),
    output_tokens: ([.[] | select(.event_name == "api_request") | ((.output_tokens // 0) | tostring | tonumber? // 0)] | add // 0),
    errors:        ([.[] | select(.event_name == "api_error")] | length),
    first_prompt:  ([.[] | select(.event_name == "user_prompt") | .prompt // empty] | first // "(no prompt logged)")
  })
| sort_by(-.cost_usd)
JQ
)

echo "=== Cost & Observability Report — last $DAYS day(s) ==="
echo

SUMMARY=$(cat $FILES | scrub | jq -c 'select(type == "object")' 2>/dev/null | jq -s "$JQ_PROGRAM" 2>/dev/null || echo "[]")

if [ "$SUMMARY" = "[]" ] || [ -z "$SUMMARY" ]; then
    echo "(no parseable events found in telemetry files)"
    echo
    echo "Raw file sizes:"
    ls -lh $FILES
    exit 0
fi

echo "$SUMMARY" | jq -r '
    .[] |
    "session \(.session[0:12])  turns=\(.turns)  cost=$\(.cost_usd)  tok=\(.input_tokens)/\(.output_tokens)  err=\(.errors)\n  first: \(.first_prompt[0:80])"
'

echo
echo "=== Totals ==="
echo "$SUMMARY" | jq -r '
    {
      sessions:            length,
      api_requests:        ([.[] | .turns] | add // 0),
      api_errors:          ([.[] | .errors] | add // 0),
      total_cost_usd:      ([.[] | .cost_usd] | add // 0),
      total_input_tokens:  ([.[] | .input_tokens] | add // 0),
      total_output_tokens: ([.[] | .output_tokens] | add // 0)
    }
    | "sessions=\(.sessions)  requests=\(.api_requests)  errors=\(.api_errors)  cost=$\(.total_cost_usd)  tokens=\(.total_input_tokens) in / \(.total_output_tokens) out"
'
